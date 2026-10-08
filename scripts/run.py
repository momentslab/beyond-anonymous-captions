#!/usr/bin/env python3
"""
BAC inference — identity QA or captioning, coord_face, Qwen 2b/4b/8b.

Usage:
    python3 scripts/run.py qa  --size 2b                  # base QA
    python3 scripts/run.py qa  --size 8b --lora auto      # LoRA QA (default LoRA path)
    python3 scripts/run.py qa  --size 4b --lora /path/to/adapter
    python3 scripts/run.py cap --size 2b                  # base captioning
    python3 scripts/run.py cap --size 8b --lora auto      # LoRA captioning
    python3 scripts/run.py qa  --size 2b --limit 5        # quick test
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import cv2
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from grounding.annotate import build_frame_grid
from grounding.generate import generate_multiframe_batch
from grounding.one_prompt import (
    apply_qa_prompt,
    encode_questions_with_p_indices,
    prepare_one_prompt_inputs,
    prepare_one_prompt_qa_inputs,
)
from config import LORA_HF_REPO, LORA_PRESETS, MAX_NEW_TOKENS, MODEL_PRESETS, QA_MAX_NEW_TOKENS
from prompts import IDENTITY_CAPTION_PROMPT, IDENTITY_QA_PROMPT
from utils.bbox import estimate_body_bbox_from_face
from utils.captioner import load_batched_hf_captioner
from utils.names import replace_p_indices
from utils.qa import (
    compare_names_from_qa,
    decode_p_answers,
    merge_separate_qa_result,
    parse_qa_output_or_plain,
    parsed_answer_ids,
    write_identity_qa_accuracies,
)
from utils.text_clean import clean_awkward_phrasings

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
QA_MODE = "identity_qa_coord_face"
QA_RESULT = "result_identity_qa_coord_face.json"
CAP_MODE = "caption_coord_face"
CAP_RESULT = "result_caption_coord_face.json"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def discover_shots(ann_root: Path) -> list[Path]:
    return sorted(p.parent for p in ann_root.glob("*/*/shot*/input_sequence.json"))


def shot_output_folder(output_root: Path, shot_folder: Path, input_data: dict) -> Path:
    movie = input_data.get("movie") or shot_folder.parent.parent.name
    clip = input_data.get("source_clip") or shot_folder.parent.name
    shot = shot_folder.name
    return output_root / movie / clip / shot


def _extract_frames_metadata(prepared):
    frames = prepared.get("frames") or prepared["input_data"]["frames"]
    image_w = prepared.get("image_w") or prepared["input_data"]["frame_width"]
    image_h = prepared.get("image_h") or prepared["input_data"]["frame_height"]
    p_to_full = prepared.get("p_to_full") or {}
    out = []
    for frame in frames:
        persons_prompt = frame.get("_persons_for_prompt", [])
        frame_persons = []
        for p in persons_prompt:
            face = p.get("face_bbox") or p["bbox"]
            body_est = estimate_body_bbox_from_face(
                face, k_h=6.5, k_w=2.5, k_offset=0.15,
                image_width=image_w, image_height=image_h,
            )
            frame_persons.append({
                "label": p["name"],
                "actor": p_to_full.get(p["name"], p["name"]),
                "face_bbox": face,
                "body_bbox_estimated": body_est,
                "used_bbox": p["bbox"],
            })
        out.append({
            "frame_order": frame.get("frame_order", 0),
            "image": frame.get("image", ""),
            "persons": frame_persons,
        })
    return out


def save_viz(output_folder: Path, frames, annotated_images, suffix: str):
    output_folder.mkdir(parents=True, exist_ok=True)
    for frame, img in zip(frames, annotated_images):
        cv2.imwrite(
            str(output_folder / f"frame{frame['frame_order']}_{suffix}_annotated.jpg"), img,
        )
    grid = build_frame_grid(annotated_images, frames=frames)
    if grid is not None:
        cv2.imwrite(str(output_folder / f"sequence_{suffix}_annotated.jpg"), grid)


def resolve_lora(lora_arg: str | None, size: str) -> Path | None:
    if lora_arg is None:
        return None
    if lora_arg == "auto":
        default = LORA_PRESETS.get(size)
        if default is None:
            raise ValueError(f"No default LoRA preset for size {size}")
        p = Path(default)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        if not (p / "adapter_config.json").exists():
            from huggingface_hub import snapshot_download
            print(f"Downloading LoRA {LORA_HF_REPO}/{size} -> {p}")
            snapshot_download(LORA_HF_REPO, allow_patterns=[f"{size}/*"], local_dir=p.parent)
        return p
    return Path(lora_arg)


# ---------------------------------------------------------------------------
# QA
# ---------------------------------------------------------------------------

def _expected_qids(entry: dict) -> set:
    return {
        str(q.get("id") or "").strip()
        for q in (entry.get("questions") or [])
        if str(q.get("id") or "").strip()
    }


def build_qa_jobs(shots, output_root: Path, resume: bool):
    jobs = []
    skipped_q = 0
    skipped_complete = 0
    for shot_folder in shots:
        entry = json.loads((shot_folder / "input_sequence.json").read_text())
        questions = entry.get("questions") or []
        if not questions:
            continue
        movie = entry.get("movie") or shot_folder.parent.parent.name
        clip = entry.get("source_clip") or shot_folder.parent.name
        shot = shot_folder.name
        result_path = output_root / movie / clip / shot / QA_RESULT
        done_ids = parsed_answer_ids(result_path) if resume else set()
        expected = _expected_qids(entry)
        if resume and expected and expected <= done_ids:
            skipped_complete += 1
            skipped_q += len(expected)
            continue
        for question in questions:
            qid = str(question.get("id") or "").strip()
            if not qid or not str(question.get("question") or "").strip():
                continue
            if resume and qid in done_ids:
                skipped_q += 1
                continue
            jobs.append({
                "shot_folder": shot_folder,
                "qa_single": {**entry, "questions": [question]},
                "qa_full": entry,
            })
    return jobs, skipped_q, skipped_complete


def run_qa(args, shots, output_root: Path, model_name: str, batch_size: int,
           lora_path: Path | None) -> int:
    jobs, skipped_q, skipped_complete = build_qa_jobs(shots, output_root, args.resume)

    print(f"Questions:  {len(jobs)} remaining")
    print(f"Skipped:    {skipped_q} questions ({skipped_complete} complete shots)")
    print("=" * 60)

    if not jobs:
        print("Nothing to process.")
        return 0

    output_root.mkdir(parents=True, exist_ok=True)
    captioner = load_batched_hf_captioner(
        model_name, use_compile=not args.no_compile,
        lora_path=str(lora_path) if lora_path else None,
    )

    succeeded = failed = 0
    skipped = skipped_q
    errors: list[dict] = []
    total_infer = 0.0
    wall_start = time.time()
    cache: dict[Path, dict] = {}

    for start in range(0, len(jobs), batch_size):
        batch_jobs = jobs[start:start + batch_size]
        prepared_batch = []
        for job in batch_jobs:
            sf = job["shot_folder"]
            try:
                if sf not in cache:
                    cache[sf] = prepare_one_prompt_qa_inputs(
                        sf, job["qa_full"], prompt_template=IDENTITY_QA_PROMPT,
                        use_p_indices=True, box_source="face", draw_boxes=True,
                    )
                prepared = apply_qa_prompt(
                    cache[sf], job["qa_single"],
                    prompt_template=IDENTITY_QA_PROMPT, use_p_indices=True,
                )
                prepared["shot_folder"] = sf
                prepared["qa_entry_full"] = job["qa_full"]
                prepared_batch.append(prepared)
            except Exception as exc:
                skipped += 1
                errors.append({"shot": str(sf), "error": str(exc), "stage": "prepare"})
                print(f"SKIP {sf.name}: {exc}", flush=True)

        if not prepared_batch:
            continue

        try:
            t0 = time.time()
            outputs_raw = generate_multiframe_batch(
                captioner,
                [p["annotated_images"] for p in prepared_batch],
                [p["prompt"] for p in prepared_batch],
                max_new_tokens=args.max_new_tokens,
            )
            dt = time.time() - t0
            total_infer += dt
            per_run = dt / len(prepared_batch)
        except Exception as exc:
            failed += len(prepared_batch)
            traceback.print_exc()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            for p in prepared_batch:
                errors.append({"shot": str(p["shot_folder"]), "error": str(exc), "stage": "generate"})
            continue

        for prepared, output_text in zip(prepared_batch, outputs_raw):
            sf = prepared["shot_folder"]
            try:
                idata = prepared["input_data"]
                out_dir = shot_output_folder(output_root, sf, idata)
                out_dir.mkdir(parents=True, exist_ok=True)
                rpath = out_dir / QA_RESULT
                qa_entry = prepared["qa_entry"]
                qid = str((qa_entry.get("questions") or [{}])[0].get("id") or "Q1")
                parsed = parse_qa_output_or_plain(output_text, qid)
                p_to_compare = compare_names_from_qa(
                    prepared.get("qa_entry_full") or qa_entry,
                    prepared.get("p_to_short") or {},
                    prepared.get("variant_to_p") or {},
                )
                parsed = decode_p_answers(parsed, p_to_compare)
                extra = {
                    "p_to_full": prepared.get("p_to_full") or {},
                    "p_indices": True, "box_source": "face",
                    "draw_boxes": True, "grounding": "coord_face",
                    "frames_metadata": _extract_frames_metadata(prepared),
                }
                result = merge_separate_qa_result(
                    rpath,
                    movie_name=idata.get("movie") or sf.parent.parent.name,
                    clip_id=(idata.get("source_clip") or sf.parent.name) + "/" + sf.name,
                    mode=QA_MODE,
                    full_questions=(prepared.get("qa_entry_full") or qa_entry).get("questions") or [],
                    single_questions=qa_entry.get("questions") or [],
                    parsed_answers=parsed, output_text=output_text,
                    actors=prepared["actors"],
                    num_frames=len(prepared["frames"]),
                    per_clip_time=per_run, model_name=model_name,
                    questions_p=encode_questions_with_p_indices(
                        (prepared.get("qa_entry_full") or qa_entry).get("questions") or [],
                        prepared.get("variant_to_p") or {},
                    ),
                    extra=extra,
                )
                tmp = rpath.with_name(f"{rpath.stem}.{os.getpid()}.tmp.json")
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                tmp.replace(rpath)
                if args.visualize:
                    save_viz(out_dir, prepared["frames"], prepared["annotated_images"], "qa_coord_face")
                succeeded += 1
                answer = ""
                if parsed:
                    ans = (parsed.get("answers") or [{}])[0] if parsed.get("answers") else {}
                    answer = str(ans.get("answer") or "")
                    if ans.get("answer_p"):
                        answer = f"{ans['answer_p']}->{answer}"
                print(f"[{succeeded+failed}/{len(jobs)}] {sf.parent.name}/{sf.name} {qid} ({per_run:.2f}s) — {answer or output_text[:60]}", flush=True)
            except Exception as exc:
                failed += 1
                errors.append({"shot": str(sf), "error": str(exc), "stage": "save"})
                print(f"FAILED {sf.name}: {exc}", flush=True)
                traceback.print_exc()

        live = {j["shot_folder"] for j in jobs[start + batch_size:]}
        cache = {k: v for k, v in cache.items() if k in live}

    wall = time.time() - wall_start
    _write_timing(output_root, QA_MODE, model_name, lora_path, args.size,
                  len(jobs), succeeded, skipped, failed, total_infer, wall)
    if errors:
        with open(output_root / "errors.json", "w", encoding="utf-8") as f:
            json.dump(errors, f, indent=2, ensure_ascii=False)
    acc_path = write_identity_qa_accuracies(output_root, mode=QA_MODE, result_filename=QA_RESULT)
    acc = json.loads(acc_path.read_text(encoding="utf-8"))

    print()
    print("=" * 60)
    print(f"Succeeded: {succeeded}")
    print(f"Skipped:   {skipped}")
    print(f"Failed:    {failed}")
    print(f"Inference: {total_infer:.1f}s  |  Wall: {wall:.1f}s")
    print(f"Accuracy:  {acc.get('accuracy_percent')}% "
          f"({acc.get('canonical_match')}/{acc.get('questions')})")
    print(f"  named:   {acc.get('named',{}).get('accuracy_percent')}%")
    print(f"  UNID:    {acc.get('unidentified',{}).get('accuracy_percent')}%")
    print(f"Output:    {output_root}")
    print(f"Accuracy:  {acc_path}")
    print("=" * 60)
    return 1 if failed and failed == len(jobs) else 0


# ---------------------------------------------------------------------------
# Captioning
# ---------------------------------------------------------------------------

def run_caption(args, shots, output_root: Path, model_name: str, batch_size: int,
                lora_path: Path | None) -> int:
    if args.resume:
        remaining = []
        skip_n = 0
        for sf in shots:
            entry = json.loads((sf / "input_sequence.json").read_text())
            if (shot_output_folder(output_root, sf, entry) / CAP_RESULT).exists():
                skip_n += 1
            else:
                remaining.append(sf)
        print(f"Skipping {skip_n} already processed shots")
        shots = remaining

    print(f"Shots:      {len(shots)}")
    print("=" * 60)

    if not shots:
        print("Nothing to process.")
        return 0

    output_root.mkdir(parents=True, exist_ok=True)
    captioner = load_batched_hf_captioner(
        model_name, use_compile=not args.no_compile,
        lora_path=str(lora_path) if lora_path else None,
    )

    succeeded = skipped = failed = 0
    errors: list[dict] = []
    total_infer = 0.0
    wall_start = time.time()

    for start in range(0, len(shots), batch_size):
        batch_folders = shots[start:start + batch_size]
        prepared_batch = []
        for sf in batch_folders:
            try:
                prepared = prepare_one_prompt_inputs(
                    sf, prompt_template=IDENTITY_CAPTION_PROMPT,
                    use_p_indices=True, box_source="face", draw_boxes=True,
                )
                prepared["shot_folder"] = sf
                prepared_batch.append(prepared)
            except Exception as exc:
                skipped += 1
                errors.append({"shot": str(sf), "error": str(exc), "stage": "prepare"})
                print(f"SKIP {sf.name}: {exc}", flush=True)

        if not prepared_batch:
            continue

        try:
            t0 = time.time()
            captions_raw = generate_multiframe_batch(
                captioner,
                [p["annotated_images"] for p in prepared_batch],
                [p["prompt"] for p in prepared_batch],
                max_new_tokens=args.max_new_tokens,
            )
            dt = time.time() - t0
            total_infer += dt
            per_clip = dt / len(prepared_batch)
        except Exception as exc:
            failed += len(prepared_batch)
            traceback.print_exc()
            for p in prepared_batch:
                errors.append({"shot": str(p["shot_folder"]), "error": str(exc), "stage": "generate"})
            continue

        for prepared, caption_raw in zip(prepared_batch, captions_raw):
            sf = prepared["shot_folder"]
            try:
                p_to_short = prepared.get("p_to_short") or {}
                caption = clean_awkward_phrasings(replace_p_indices(caption_raw, p_to_short))
                idata = prepared["input_data"]
                out_dir = shot_output_folder(output_root, sf, idata)
                out_dir.mkdir(parents=True, exist_ok=True)
                result = {
                    "mode": CAP_MODE,
                    "model": model_name,
                    "lora": str(lora_path) if lora_path else None,
                    "num_frames": len(prepared["frames"]),
                    "frame_width": prepared["image_w"],
                    "frame_height": prepared["image_h"],
                    "actors": prepared["actors"],
                    "caption_raw": caption_raw,
                    "caption": caption,
                    "p_to_full": prepared.get("p_to_full") or {},
                    "inference_time_seconds": per_clip,
                }
                for key in ("movie", "source_clip", "fps", "total_video_frames"):
                    if key in idata:
                        result[key] = idata[key]
                result.update({
                    "p_indices": True, "box_source": "face",
                    "draw_boxes": True, "grounding": "coord_face",
                    "frames_metadata": _extract_frames_metadata(prepared),
                })
                rpath = out_dir / CAP_RESULT
                tmp = rpath.with_name(f"{rpath.stem}.{os.getpid()}.tmp.json")
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                tmp.replace(rpath)
                if args.visualize:
                    save_viz(out_dir, prepared["frames"], prepared["annotated_images"], "cap_coord_face")
                succeeded += 1
                print(f"[{succeeded+skipped+failed}/{len(shots)}] {sf.parent.name}/{sf.name} ({per_clip:.2f}s) — {caption[:80]}", flush=True)
            except Exception as exc:
                failed += 1
                errors.append({"shot": str(sf), "error": str(exc), "stage": "save"})
                print(f"FAILED {sf.name}: {exc}", flush=True)
                traceback.print_exc()

    wall = time.time() - wall_start
    _write_timing(output_root, CAP_MODE, model_name, lora_path, args.size,
                  len(shots), succeeded, skipped, failed, total_infer, wall)
    if errors:
        with open(output_root / "errors.json", "w", encoding="utf-8") as f:
            json.dump(errors, f, indent=2, ensure_ascii=False)

    print()
    print("=" * 60)
    print(f"Succeeded: {succeeded}")
    print(f"Skipped:   {skipped}")
    print(f"Failed:    {failed}")
    print(f"Inference: {total_infer:.1f}s  |  Wall: {wall:.1f}s")
    print(f"Output:    {output_root}")
    print("=" * 60)
    return 1 if failed == len(shots) else 0


# ---------------------------------------------------------------------------
# Shared
# ---------------------------------------------------------------------------

def _write_timing(output_root, mode, model_name, lora_path, size,
                  total, succeeded, skipped, failed, total_infer, wall):
    timing = {
        "mode": mode, "grounding": "coord_face",
        "model": model_name,
        "lora": str(lora_path) if lora_path else None,
        "size": size,
        "total": total, "processed": succeeded,
        "skipped": skipped, "failed": failed,
        "inference_seconds": total_infer,
        "wall_seconds": wall,
        "avg_seconds": (total_infer / succeeded) if succeeded else 0,
    }
    path = output_root / f"{mode}_timing.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(timing, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="BAC inference — identity QA or captioning (coord_face, Qwen)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  python3 scripts/run.py qa  --size 2b
  python3 scripts/run.py qa  --size 8b --lora auto
  python3 scripts/run.py cap --size 4b --lora /path/to/adapter
  python3 scripts/run.py qa  --size 2b --limit 5 --visualize""",
    )
    parser.add_argument("task", choices=["qa", "cap"], help="qa = question answering, cap = captioning")
    parser.add_argument("--size", choices=sorted(MODEL_PRESETS), default="2b", help="Model size (default: 2b)")
    parser.add_argument("--lora", default=None,
                        help="LoRA adapter: 'auto' for models/lora/{size} (downloaded from HF if missing), or explicit dir")
    parser.add_argument("--model", default=None, help="Override HF model id")
    parser.add_argument("--output_dir", type=Path, default=None, help="Override output root")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=None,
                        help=f"Max tokens (default: {QA_MAX_NEW_TOKENS} for qa, {MAX_NEW_TOKENS} for cap)")
    parser.add_argument("--limit", type=int, default=None, help="Process only first N shots")
    parser.add_argument("--resume", action="store_true", help="Skip already-done shots/questions")
    parser.add_argument("--visualize", action="store_true", help="Save annotated frame images")
    parser.add_argument("--no_compile", action="store_true", help="Disable torch.compile")
    args = parser.parse_args()

    if args.max_new_tokens is None:
        args.max_new_tokens = QA_MAX_NEW_TOKENS if args.task == "qa" else MAX_NEW_TOKENS

    ann_root = PROJECT_ROOT / "data" / "shot_captions" / "test"
    if not ann_root.exists():
        print(f"Error: annotation dir not found: {ann_root}")
        return 1

    lora_path = resolve_lora(args.lora, args.size)
    if lora_path and not (lora_path / "adapter_config.json").exists():
        print(f"Error: LoRA adapter not found: {lora_path}")
        return 1

    preset = MODEL_PRESETS[args.size]
    model_name = args.model or preset["model"]
    batch_size = args.batch_size or preset["batch_size"]
    lora_tag = "_lora" if lora_path else ""
    task_tag = "qa" if args.task == "qa" else "cap"
    if args.output_dir:
        output_root = args.output_dir
    else:
        output_root = PROJECT_ROOT / "output" / f"Qwen3_{args.size}{lora_tag}_{task_tag}" / "test"

    shots = discover_shots(ann_root)
    if args.limit:
        shots = shots[:args.limit]

    print("=" * 60)
    print(f"Task:       {args.task}")
    print(f"Size:       {args.size}")
    print(f"Model:      {model_name}")
    if lora_path:
        print(f"LoRA:       {lora_path}")
    print(f"Output:     {output_root}")
    print(f"Shots:      {len(shots)}")
    print(f"Batch:      {batch_size}")
    print(f"Tokens:     {args.max_new_tokens}")
    print(f"Method:     coord_face + P-indices")

    if args.task == "qa":
        return run_qa(args, shots, output_root, model_name, batch_size, lora_path)
    else:
        return run_caption(args, shots, output_root, model_name, batch_size, lora_path)


if __name__ == "__main__":
    raise SystemExit(main())
