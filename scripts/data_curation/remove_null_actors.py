#!/usr/bin/env python3
"""
Copy shots that have unrecognized people (actor is null) into null_remove,
drop those person boxes, and rebuild frames + sequence grids.

The original split under data/shot_captions/{split} is left unchanged.

Usage:
    python scripts/data_curation/remove_null_actors.py
    python scripts/data_curation/remove_null_actors.py --dry_run
    python scripts/data_curation/remove_null_actors.py --split test --limit 3
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from extract_shot_frames import (  # noqa: E402
    extract_shot,
    find_video,
    read_frames_by_index,
)

import cv2


def is_null_actor(person: dict) -> bool:
    actor = person.get("actor")
    return actor is None or not str(actor).strip()


def shot_has_null_actor(data: dict) -> bool:
    return any(
        is_null_actor(person)
        for frame in data.get("frames") or []
        for person in frame.get("persons") or []
    )


def strip_null_actors(data: dict) -> tuple[dict, dict]:
    cleaned = deepcopy(data)
    removed_boxes = 0
    kept_boxes = 0
    removed_characters: dict[str, int] = defaultdict(int)

    for frame in cleaned.get("frames") or []:
        kept = []
        for person in frame.get("persons") or []:
            if is_null_actor(person):
                removed_boxes += 1
                char = person.get("character") or "(no character)"
                removed_characters[str(char)] += 1
            else:
                kept.append(person)
                kept_boxes += 1
        frame["persons"] = kept
        frame["has_grounding"] = bool(kept)

    stats = {
        "removed_boxes": removed_boxes,
        "kept_boxes": kept_boxes,
        "removed_characters": dict(removed_characters),
        "empty_after": not any(
            frame.get("persons") for frame in cleaned.get("frames") or []
        ),
    }
    return cleaned, stats


def collect_affected_shots(src_root: Path) -> list[Path]:
    shots = []
    for path in sorted(src_root.glob("*/*/shot*/input_sequence.json")):
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if shot_has_null_actor(data):
            shots.append(path)
    return shots


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def extract_clip_frames(clip_jobs: list[tuple[Path, dict]]) -> int:
    first_json, first_data = clip_jobs[0]
    clip = first_data.get("source_clip") or first_json.parent.parent.name
    hint = first_data.get("video")
    video_path = find_video(clip, hint)
    if video_path is None:
        print(f"  x video not found for {clip}")
        return 0

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  x cannot open {video_path}")
        return 0

    needed: set[int] = set()
    for _json_path, data in clip_jobs:
        for frame in data.get("frames") or []:
            needed.add(int(frame["frame_index"]))
    frames_by_index = read_frames_by_index(cap, needed)
    cap.release()

    saved = 0
    for json_path, _data in clip_jobs:
        images_dir = json_path.parent / "frames"
        if extract_shot(json_path, images_dir, frames_by_index):
            saved += 1
    return saved


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Drop actor:null person boxes from affected shots and write a "
            "verification copy under null_remove."
        )
    )
    parser.add_argument("--split", default="test", choices=["test", "train"])
    parser.add_argument(
        "--src",
        type=Path,
        default=None,
        help="Shot-caption split root (default: data/shot_captions/{split})",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output root (default: null_remove)",
    )
    parser.add_argument("--limit", type=int, default=None, help="Max shots to write")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    src_root = args.src or (PROJECT_ROOT / "data" / "shot_captions" / args.split)
    out_root = args.out or (PROJECT_ROOT / "null_remove")
    if not src_root.exists():
        print(f"Error: source not found: {src_root}")
        return 1

    print("=" * 60)
    print("Remove unrecognized (actor: null) people")
    print("=" * 60)
    print(f"Source: {src_root}")
    print(f"Output: {out_root}")
    print("Original split is not modified.\n")

    shot_paths = collect_affected_shots(src_root)
    if args.limit is not None:
        shot_paths = shot_paths[: args.limit]
    print(f"Affected shots: {len(shot_paths)}")

    if not shot_paths:
        print("Nothing to do.")
        return 0

    jobs_by_clip: dict[tuple[str, str], list[tuple[Path, dict]]] = defaultdict(list)
    summary_shots = []
    total_removed = 0
    total_kept = 0
    empty_after = 0

    for src_path in shot_paths:
        data = json.loads(src_path.read_text())
        cleaned, stats = strip_null_actors(data)
        movie = src_path.parts[-4]
        clip = src_path.parts[-3]
        shot = src_path.parts[-2]
        dest = out_root / movie / clip / shot / "input_sequence.json"
        rel = f"{movie}/{clip}/{shot}"

        total_removed += stats["removed_boxes"]
        total_kept += stats["kept_boxes"]
        if stats["empty_after"]:
            empty_after += 1

        summary_shots.append({
            "shot": rel,
            "removed_boxes": stats["removed_boxes"],
            "kept_boxes": stats["kept_boxes"],
            "removed_characters": stats["removed_characters"],
            "empty_after": stats["empty_after"],
        })
        print(
            f"  {rel}: removed {stats['removed_boxes']} null box(es), "
            f"kept {stats['kept_boxes']}"
            + (" [no named actors left]" if stats["empty_after"] else "")
        )

        if args.dry_run:
            continue
        write_json(dest, cleaned)
        jobs_by_clip[(movie, clip)].append((dest, cleaned))

    print(
        f"\n{len(shot_paths)} shots, {total_removed} null boxes removed, "
        f"{total_kept} named boxes kept, {empty_after} shots empty after filter"
    )

    if args.dry_run:
        print("\nDry run: no files written.")
        return 0

    print("\nExtracting frames and sequence grids...")
    saved_shots = 0
    for (movie, clip), clip_jobs in jobs_by_clip.items():
        print(f"  {movie}/{clip} ({len(clip_jobs)} shot(s))")
        saved_shots += extract_clip_frames(clip_jobs)

    manifest = {
        "source": str(src_root),
        "output": str(out_root),
        "split": args.split,
        "affected_shots": len(shot_paths),
        "frames_written": saved_shots,
        "null_boxes_removed": total_removed,
        "named_boxes_kept": total_kept,
        "shots_with_no_named_actors": empty_after,
        "shots": summary_shots,
    }
    write_json(out_root / "manifest.json", manifest)
    print(f"\nWrote {saved_shots} shot folders → {out_root}")
    print(f"Manifest: {out_root / 'manifest.json'}")
    print("Original test split was not changed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
