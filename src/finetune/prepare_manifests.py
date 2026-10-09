"""Build train/eval manifests from BAC shot folders.

Default: put every valid training shot in train and leave eval empty
(one-epoch fine-tune over the full set). Pass ``eval_movies`` to hold out
those movies for evaluation (movie-level split: all their shots go to eval).

A shot is kept if it has ``caption_pre`` and at least one extracted frame.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


def _has_frames(shot_folder: Path) -> bool:
    frames_dir = shot_folder / "frames"
    if not frames_dir.is_dir():
        return False
    return any(frames_dir.glob("frame*.jpg"))


def collect_records(
    ann_root: Path,
    target_field: str = "caption_pre",
) -> Tuple[List[Dict], Dict[str, int]]:
    """Scan shot folders and return (records, skip_counts)."""
    records: List[Dict] = []
    counts = {
        "total_shots": 0,
        "skipped_no_json": 0,
        "skipped_bad_json": 0,
        "skipped_no_caption": 0,
        "skipped_no_frames": 0,
    }

    for json_path in sorted(ann_root.glob("*/*/shot*/input_sequence.json")):
        counts["total_shots"] += 1
        shot_folder = json_path.parent
        clip_folder = shot_folder.parent
        movie_folder = clip_folder.parent

        try:
            with json_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            counts["skipped_bad_json"] += 1
            continue

        caption = str(data.get(target_field) or "").strip()
        if not caption:
            counts["skipped_no_caption"] += 1
            continue

        if not _has_frames(shot_folder):
            counts["skipped_no_frames"] += 1
            continue

        records.append({
            "movie": movie_folder.name,
            "clip": clip_folder.name,
            "shot": shot_folder.name,
            "target_caption": caption,
        })

    return records, counts


def write_jsonl(path: Path, records: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_manifests(
    ann_root: Path,
    output_dir: Path,
    eval_movies: Optional[Iterable[str]] = None,
    target_field: str = "caption_pre",
) -> Dict[str, object]:
    """Write train_manifest.jsonl and eval_manifest.jsonl.

    Without ``eval_movies`` (default) every sample goes to train and the
    eval file is empty. With ``eval_movies``, every shot of those movies
    goes to eval and all other movies to train.
    """
    held_out = set(eval_movies or [])
    missing = sorted(m for m in held_out if not (ann_root / m).is_dir())
    if missing:
        print(f"[warning] Eval movies not found under {ann_root}: {missing}", flush=True)

    records, counts = collect_records(ann_root, target_field=target_field)
    eval_records = [r for r in records if r["movie"] in held_out]
    train_records = [r for r in records if r["movie"] not in held_out]

    train_path = output_dir / "train_manifest.jsonl"
    eval_path = output_dir / "eval_manifest.jsonl"
    write_jsonl(train_path, train_records)
    write_jsonl(eval_path, eval_records)

    summary = {
        **counts,
        "valid": len(records),
        "train": len(train_records),
        "eval": len(eval_records),
        "train_movies": len({r["movie"] for r in train_records}),
        "eval_movies": len({r["movie"] for r in eval_records}),
        "train_path": str(train_path),
        "eval_path": str(eval_path),
    }
    return summary


def manifest_len(path: Optional[Path]) -> int:
    if path is None or not path.exists():
        return 0
    n = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n
