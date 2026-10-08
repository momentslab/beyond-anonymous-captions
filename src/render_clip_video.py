#!/usr/bin/env python3
"""
Render annotated videos from clip_faces JSONs.

Reads:
    data/clip_faces/{dataset}/{movie}/{clip}.json
    {videos_root}/{dataset}/{movie}/{clip}.avi   (--videos-root or $LSMDC_VIDEOS_ROOT)

Writes:
    {output_dir}/{clip}_bbox.mp4

Each frame gets face bboxes drawn with actor name + score from the JSON's
``sampled_frames[].faces[]`` entries.

Usage:
    python3 src/render_clip_video.py --dataset lsmdc --movie 30_MINUTES_OR_LESS
    python3 src/render_clip_video.py --dataset lsmdc --movie 30_MINUTES_OR_LESS --limit 3
    python3 src/render_clip_video.py --clip 3002_30_MINUTES_OR_LESS_00.03.40.059-00.03.45.728
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
# Root of the LSMDC / MPII-MD videos: {VIDEOS_ROOT}/{lsmdc,mpiimd}/{movie}/{clip}.avi
VIDEOS_ROOT = Path(os.environ.get("LSMDC_VIDEOS_ROOT", ROOT / "data" / "videos"))
CLIP_FACES_DIR = DATA_DIR / "clip_faces"

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


def find_video(dataset: str, movie: str, clip: str, hint: str | None = None) -> Path | None:
    """Look for the video: JSON ``video`` path first, then the expected location."""
    if hint:
        hint = hint.removeprefix("./")
        rel = hint[len("data/videos/"):] if hint.startswith("data/videos/") else hint
        candidate = VIDEOS_ROOT / rel
        if candidate.exists():
            return candidate
    direct = VIDEOS_ROOT / dataset / movie / f"{clip}.avi"
    if direct.exists():
        return direct
    matches = list(VIDEOS_ROOT.rglob(f"{clip}.avi"))
    return matches[0] if matches else None


def _title_case_name(name: str) -> str:
    """Convert 'nick davis' → 'Nick Davis'."""
    return " ".join(w.capitalize() for w in name.strip().split())


def draw_face(frame, bbox, label, color):
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


def render_clip(clip_json: Path, dataset: str, movie: str, output_dir: Path) -> Path | None:
    clip = clip_json.stem
    data = json.loads(clip_json.read_text())

    faces_by_frame: dict[int, list] = {}
    for item in data.get("sampled_frames", []):
        fi = int(item["frame_index"])
        faces = item.get("faces", [])
        if faces:
            faces_by_frame[fi] = faces

    if not faces_by_frame:
        print(f"  - {clip}: no faces at all, skipping")
        return None

    video_path = find_video(dataset, movie, clip, data.get("video"))
    if video_path is None:
        print(f"  x video not found for {clip}")
        return None

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  x cannot open {video_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or data.get("fps") or 24.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{clip}_bbox.mp4"
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height),
    )
    if not writer.isOpened():
        cap.release()
        print(f"  x cannot create {out_path}")
        return None

    color_table: dict[str, tuple] = {}
    idx = 0
    boxed = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        faces = faces_by_frame.get(idx, [])
        if faces:
            boxed += 1
            for face in faces:
                name = _title_case_name(face.get("name", "Unknown"))
                score = face.get("score")
                label = f"{name} {score:.2f}" if isinstance(score, (int, float)) else name
                draw_face(frame, face["bbox"], label, color_for(name, color_table))
        writer.write(frame)
        idx += 1

    cap.release()
    writer.release()
    print(f"  + {out_path.name}  frames={idx} with_bbox={boxed}")
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render annotated videos from clip_faces JSONs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python3 src/render_clip_video.py --dataset lsmdc --movie 30_MINUTES_OR_LESS
  python3 src/render_clip_video.py --dataset lsmdc --movie 30_MINUTES_OR_LESS --limit 3
  python3 src/render_clip_video.py --clip 3002_30_MINUTES_OR_LESS_00.03.40.059-00.03.45.728
""",
    )
    parser.add_argument("--dataset", default=None, choices=["lsmdc", "mpiimd"],
                        help="Dataset folder inside clip_faces/")
    parser.add_argument("--movie", default=None,
                        help="Movie folder name (e.g. 30_MINUTES_OR_LESS)")
    parser.add_argument("--clip", default=None,
                        help="Render a single clip by name (searches all datasets)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only first N clips")
    parser.add_argument("--videos-root", type=Path, default=None,
                        help="LSMDC/MPII-MD videos root (default: $LSMDC_VIDEOS_ROOT or data/videos)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output dir (default: output/videos/clip_faces)")
    args = parser.parse_args()
    global VIDEOS_ROOT
    if args.videos_root:
        VIDEOS_ROOT = args.videos_root

    output_dir = args.output_dir or (ROOT / "output" / "videos" / "clip_faces")

    if args.clip:
        matches = list(CLIP_FACES_DIR.rglob(f"{args.clip}.json"))
        if not matches:
            print(f"Clip JSON not found: {args.clip}")
            return 1
        clip_json = matches[0]
        dataset = clip_json.parent.parent.name
        movie = clip_json.parent.name
        print(f"Rendering single clip: {args.clip} ({dataset}/{movie})")
        result = render_clip(clip_json, dataset, movie, output_dir / movie)
        return 0 if result else 1

    if not args.dataset:
        datasets = [d.name for d in sorted(CLIP_FACES_DIR.iterdir()) if d.is_dir()]
    else:
        datasets = [args.dataset]

    total = 0
    saved = 0
    for ds in datasets:
        ds_dir = CLIP_FACES_DIR / ds
        if not ds_dir.is_dir():
            continue
        if args.movie:
            movie_dirs = [ds_dir / args.movie]
        else:
            movie_dirs = sorted(d for d in ds_dir.iterdir() if d.is_dir())

        for movie_dir in movie_dirs:
            if not movie_dir.is_dir():
                continue
            movie = movie_dir.name
            clips = sorted(movie_dir.glob("*.json"))
            if args.limit:
                clips = clips[:args.limit]
            if not clips:
                continue
            print(f"\n{ds}/{movie}: {len(clips)} clips")
            for clip_json in clips:
                total += 1
                result = render_clip(clip_json, ds, movie, output_dir / movie)
                if result:
                    saved += 1

    print(f"\nDone. Saved {saved}/{total} videos to {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
