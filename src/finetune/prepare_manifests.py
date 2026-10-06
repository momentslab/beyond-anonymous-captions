"""Build train/eval manifests from BAC shot folders.

Default: put every valid training shot in train and leave eval empty
(one-epoch fine-tune over the full set). Pass ``eval_ratio > 0`` to hold
out a random split for hyperparameter search.

A shot is kept if it has ``caption_pre`` and at least one extracted frame.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple


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
    eval_ratio: float = 0.0,
    seed: int = 42,
    target_field: str = "caption_pre",
) -> Dict[str, object]:
    """Write train_manifest.jsonl and eval_manifest.jsonl.

    ``eval_ratio=0`` (default) puts every sample in train and writes an
    empty eval file. ``eval_ratio>0`` holds out that fraction for eval.
    """
    if not 0.0 <= eval_ratio < 1.0:
        raise ValueError(f"eval_ratio must be in [0, 1), got {eval_ratio}")

    records, counts = collect_records(ann_root, target_field=target_field)
    random.seed(seed)
    random.shuffle(records)

    if eval_ratio > 0 and records:
        n_eval = max(1, int(len(records) * eval_ratio))
        eval_records = records[:n_eval]
        train_records = records[n_eval:]
    else:
        eval_records = []
        train_records = records

    train_path = output_dir / "train_manifest.jsonl"
    eval_path = output_dir / "eval_manifest.jsonl"
    write_jsonl(train_path, train_records)
    write_jsonl(eval_path, eval_records)

    summary = {
        **counts,
        "valid": len(records),
        "train": len(train_records),
        "eval": len(eval_records),
        "train_path": str(train_path),
        "eval_path": str(eval_path),
        "eval_ratio": eval_ratio,
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
