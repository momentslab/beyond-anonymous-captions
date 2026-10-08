#!/usr/bin/env python3
"""
Drop test-removed null identities from matching clip_faces JSONs.

Source of truth: null_remove/manifest.json (the characters dropped
from test as actor=null). For each affected test clip, those gallery
names are removed from every sampled frame of the clip_faces JSON, not
only the 8 test-selected frames.

Usage:
    python3 scripts/data_curation/strip_test_nulls_from_clip_faces.py
    python3 scripts/data_curation/strip_test_nulls_from_clip_faces.py --dry_run
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLIP_FACES = PROJECT_ROOT / "data" / "clip_faces"
MANIFEST = PROJECT_ROOT / "null_remove" / "manifest.json"


def clip_id_from_shot(shot: str) -> str:
    # movie/clip/shotN
    return shot.split("/")[1]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Remove test-null characters from entire clip_faces JSONs."
    )
    parser.add_argument("--root", type=Path, default=CLIP_FACES)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    if not args.manifest.exists():
        print(f"Error: manifest not found: {args.manifest}")
        return 1

    manifest = json.loads(args.manifest.read_text())
    drop_by_clip: dict[str, set[str]] = defaultdict(set)
    for entry in manifest.get("shots") or []:
        clip = clip_id_from_shot(entry["shot"])
        for name in (entry.get("removed_characters") or {}):
            drop_by_clip[clip].add(name.strip().lower())

    json_by_stem = {p.stem: p for p in args.root.glob("*/*/*.json")}

    print("=" * 60)
    print("Strip test-null identities from clip_faces (whole JSON)")
    print("=" * 60)
    print(f"Affected test clips: {len(drop_by_clip)}")
    print(f"clip_faces root:     {args.root}\n")

    missing = 0
    files_changed = 0
    faces_removed = 0
    files_empty = 0
    per_name: Counter[str] = Counter()

    for clip, names in sorted(drop_by_clip.items()):
        path = json_by_stem.get(clip)
        if path is None:
            missing += 1
            print(f"  MISSING  {clip}")
            continue
        data = json.loads(path.read_text())
        removed = 0
        kept = 0
        for frame in data.get("sampled_frames") or []:
            kept_faces = []
            for face in frame.get("faces") or []:
                name = str(face.get("name") or "").strip().lower()
                if name in names:
                    removed += 1
                    per_name[name] += 1
                else:
                    kept_faces.append(face)
                    kept += 1
            frame["faces"] = kept_faces
        data["faces_detected"] = kept
        faces_removed += removed
        if removed:
            files_changed += 1
        if kept == 0:
            files_empty += 1
        if removed and not args.dry_run:
            path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")

    print(f"clip_faces files updated: {files_changed}")
    print(f"clip_faces missing:       {missing}")
    print(f"Faces removed:            {faces_removed}")
    print(f"Files with no faces left: {files_empty}")
    print("Removed names:")
    for name, n in per_name.most_common():
        print(f"  {n:6d}  {name}")
    if args.dry_run:
        print("\nDry run: no files written.")
    else:
        print(f"\nUpdated in place under {args.root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
