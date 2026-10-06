#!/usr/bin/env python3
"""
Extract the 8 sampled frames per shot from shot JSONs with actor-name bboxes.

Reads:
    data/shot_captions/{split}/{movie}/{clip}/shot{N}/input_sequence.json
    {videos_root}/{lsmdc,mpiimd}/{movie}/{clip}.avi   (--videos-root or $LSMDC_VIDEOS_ROOT)

Writes inside each shot folder:

    {clip}/shot{N}/frames/frame{X}.jpg
    {clip}/shot{N}/frames/frame{X}_bbox.jpg
    {clip}/shot{N}/frames/sequence.jpg

The shot JSON is already named input_sequence.json. This script does not rename it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
# Root of the LSMDC / MPII-MD videos: {VIDEOS_ROOT}/{lsmdc,mpiimd}/{movie}/{clip}.avi
VIDEOS_ROOT = Path(os.environ.get("LSMDC_VIDEOS_ROOT", ROOT / "data" / "videos"))

COLORS = [
    (0, 255, 0),
    (0, 165, 255),
    (255, 0, 0),
    (255, 0, 255),
    (0, 255, 255),
    (255, 255, 0),
    (128, 255, 128),
    (255, 128, 0),
]


def color_for(name: str, table: dict) -> tuple:
    if name not in table:
        table[name] = COLORS[len(table) % len(COLORS)]
    return table[name]


def find_video(clip: str, hint: str | None = None) -> Path | None:
    if hint:
        rel = hint[len("data/videos/"):] if hint.startswith("data/videos/") else hint
        candidate = VIDEOS_ROOT / rel
        if candidate.exists():
            return candidate
        rooted = VIDEOS_ROOT.parent.parent / hint
        if rooted.exists():
            return rooted
    direct = list(VIDEOS_ROOT.glob(f"*/*/{clip}.avi"))
    if direct:
        return direct[0]
    matches = list(VIDEOS_ROOT.rglob(f"{clip}.avi"))
    return matches[0] if matches else None


def draw_person(frame, bbox, label, color):
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    ty1 = max(0, y1 - th - baseline - 6)
    cv2.rectangle(frame, (x1, ty1), (min(w - 1, x1 + tw + 6), y1), color, -1)
    cv2.putText(
        frame, label, (x1 + 3, y1 - baseline - 3),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA,
    )


def build_sequence_grid(images, frames, cols=4, pad=14, label_h=28, tile_w=360):
    """2x4 (or similar) grid of the shot frames, labeled with order, index, time."""
    n = len(images)
    if n == 0:
        return None

    rows = (n + cols - 1) // cols
    h, w = images[0].shape[:2]
    tile_h = max(1, int(tile_w * h / w))
    cell_h = tile_h + label_h
    grid_w = cols * tile_w + (cols + 1) * pad
    grid_h = rows * cell_h + (rows + 1) * pad
    grid = np.full((grid_h, grid_w, 3), 240, dtype=np.uint8)

    font = cv2.FONT_HERSHEY_SIMPLEX
    for i, img in enumerate(images):
        r, c = divmod(i, cols)
        x0 = pad + c * (tile_w + pad)
        y0 = pad + r * (cell_h + pad)
        resized = cv2.resize(img, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
        grid[y0:y0 + tile_h, x0:x0 + tile_w] = resized
        item = frames[i] if i < len(frames) else {}
        idx = item.get("frame_index", "")
        ts = item.get("timestamp")
        label = f"{i + 1}/{n}"
        if idx != "":
            label += f" idx={int(idx)}"
        if ts is not None:
            label += f" t={float(ts):.3f}s"
        cv2.rectangle(grid, (x0, y0 + tile_h), (x0 + tile_w, y0 + cell_h), (40, 40, 40), -1)
        cv2.putText(
            grid, label, (x0 + 4, y0 + tile_h + label_h - 8),
            font, 0.45, (255, 255, 255), 1, cv2.LINE_AA,
        )
    return grid


def read_frames_by_index(cap: cv2.VideoCapture, needed: set[int]) -> dict[int, np.ndarray]:
    """Decode in order. cap.set(POS_FRAMES) is off by one on these AVIs."""
    found: dict[int, np.ndarray] = {}
    if not needed:
        return found
    last = max(needed)
    idx = 0
    while idx <= last:
        ok, frame = cap.read()
        if not ok:
            break
        if idx in needed:
            found[idx] = frame
        idx += 1
    return found


def extract_shot(
    shot_json_path: Path,
    images_dir: Path,
    frames_by_index: dict[int, np.ndarray],
) -> bool:
    data = json.loads(shot_json_path.read_text())
    shot_idx = int(data.get("shot_index", 0))
    clip = data.get("source_clip") or images_dir.parent.parent.name
    shot_id = f"{clip}_shot{shot_idx}"

    images_dir.mkdir(parents=True, exist_ok=True)

    color_table: dict[str, tuple] = {}
    sequence: list[dict] = []
    bbox_frames = []
    total_faces = 0

    for item in data.get("frames", []):
        frame_idx = int(item["frame_index"])
        frame = frames_by_index.get(frame_idx)
        if frame is None:
            print(f"    x cannot read frame {frame_idx}")
            continue

        raw_name = f"frame{frame_idx}.jpg"
        bbox_name = f"frame{frame_idx}_bbox.jpg"

        cv2.imwrite(str(images_dir / raw_name), frame)

        persons = item.get("persons", [])
        frame_bbox = frame.copy()
        for person in persons:
            actor = person.get("actor") or person.get("character") or "Unknown"
            score = person.get("ident_score", person.get("score"))
            label = f"{actor} {score:.2f}" if isinstance(score, (int, float)) else actor
            draw_person(frame_bbox, person["bbox"], label, color_for(actor, color_table))
            total_faces += 1

        cv2.imwrite(str(images_dir / bbox_name), frame_bbox)
        bbox_frames.append(frame_bbox)

        sequence.append({
            "frame_index": frame_idx,
            "timestamp": item.get("timestamp"),
        })

    if not sequence:
        return False

    grid = build_sequence_grid(bbox_frames, sequence)
    if grid is not None:
        cv2.imwrite(str(images_dir / "sequence.jpg"), grid)

    print(f"    + {shot_id}: {len(sequence)} frames, {total_faces} faces -> {images_dir}")
    return True


def _shot_done(images_dir: Path) -> bool:
    return (images_dir / "sequence.jpg").is_file()


def process_clip(clip_dir: Path, output_dir: Path) -> int:
    clip = clip_dir.name
    shot_jsons = sorted(output_dir.glob("shot*/input_sequence.json"))
    if not shot_jsons:
        return 0

    pending = []
    done = 0
    for shot_path in shot_jsons:
        images_dir = shot_path.parent / "frames"
        if _shot_done(images_dir):
            done += 1
        else:
            pending.append((shot_path, images_dir))
    if not pending:
        return done

    hint = None
    try:
        hint = json.loads(shot_jsons[0].read_text()).get("video")
    except (OSError, json.JSONDecodeError):
        hint = None

    video_path = find_video(clip, hint)
    if video_path is None:
        print(f"  x video not found for {clip}")
        return 0

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  x cannot open {video_path}")
        return 0

    needed: set[int] = set()
    for shot_path, _ in pending:
        try:
            data = json.loads(shot_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        for item in data.get("frames", []):
            needed.add(int(item["frame_index"]))
    frames_by_index = read_frames_by_index(cap, needed)

    saved = done
    for shot_path, images_dir in pending:
        if extract_shot(shot_path, images_dir, frames_by_index):
            saved += 1

    cap.release()
    return saved


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract 8 frames per shot with actor-name bboxes.",
    )
    parser.add_argument(
        "--split", default="test", choices=["test", "train"],
        help="Which split to process.",
    )
    parser.add_argument(
        "--clips", nargs="*", default=None,
        help="Clip folder names to process (default: all in split).",
    )
    parser.add_argument(
        "--annotations-dir", type=Path, default=None,
        help="Split root with movie/clip folders (default: data/shot_captions/{split}).",
    )
    parser.add_argument(
        "--videos-root", type=Path, default=None,
        help="LSMDC/MPII-MD videos root (default: $LSMDC_VIDEOS_ROOT or data/videos).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Flat output root. Default writes inside each clip folder.",
    )
    args = parser.parse_args()
    global VIDEOS_ROOT
    if args.videos_root:
        VIDEOS_ROOT = args.videos_root

    ann_dir = args.annotations_dir or (DATA_DIR / "shot_captions" / args.split)
    if not ann_dir.exists():
        print(f"Annotation dir not found: {ann_dir}")
        return 1

    wanted = set(args.clips) if args.clips else None
    clip_dirs = []
    for movie_dir in sorted(p for p in ann_dir.iterdir() if p.is_dir()):
        for clip_dir in sorted(p for p in movie_dir.iterdir() if p.is_dir()):
            if wanted is None or clip_dir.name in wanted:
                clip_dirs.append(clip_dir)

    print(f"Split: {args.split}")
    print(f"Annotations: {ann_dir}")
    print(f"Videos: {VIDEOS_ROOT}")
    print(f"Clips: {len(clip_dirs)}")
    print(f"Output: {'inside each clip folder' if args.output_dir is None else args.output_dir}\n")

    total_shots = 0
    for clip_dir in clip_dirs:
        out = args.output_dir or clip_dir
        total_shots += process_clip(clip_dir, out)

    print(f"\nDone. Extracted {total_shots} shots")
    return 0


if __name__ == "__main__":
    sys.exit(main())
