#!/usr/bin/env python3
"""
In-place clip_faces cleanup, matching train/test prepare filters:

  1. mirror-track dedup (same actor, multiple tracks in a frame)
  2. drop faces with actor=null / empty
  3. delete clips that have no named actor left

Dedup was never previously run on clip_faces (only on shot_captions).

Usage:
    python3 scripts/data_curation/clean_clip_faces.py
    python3 scripts/data_curation/clean_clip_faces.py --dry_run --limit 20
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLIP_FACES = PROJECT_ROOT / "data" / "clip_faces"


def keep_named_actor(face: dict) -> bool:
    actor = face.get("actor")
    return bool(actor) and bool(str(actor).strip())


def count_faces(data: dict) -> int:
    return sum(len(frame.get("faces") or []) for frame in data.get("sampled_frames") or [])


def has_named_actor(data: dict) -> bool:
    return any(
        keep_named_actor(face)
        for frame in data.get("sampled_frames") or []
        for face in frame.get("faces") or []
    )


def deduplicate_mirror_faces(data: dict) -> int:
    """Keep the highest-average-score track when the same actor appears twice in a frame."""
    frames = data.get("sampled_frames") or []
    actors_with_duplicates: set[str] = set()
    for frame in frames:
        actor_counts: dict[str, int] = defaultdict(int)
        for face in frame.get("faces") or []:
            actor = face.get("actor")
            if actor:
                actor_counts[str(actor)] += 1
        for actor, count in actor_counts.items():
            if count > 1:
                actors_with_duplicates.add(actor)
    if not actors_with_duplicates:
        return 0

    actor_track_scores: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for frame in frames:
        for face in frame.get("faces") or []:
            actor = face.get("actor")
            if actor not in actors_with_duplicates:
                continue
            track_id = face.get("track_id")
            if track_id is None:
                continue
            actor_track_scores[str(actor)][track_id].append(float(face.get("score") or 0.0))

    best_track_per_actor: dict[str, int] = {}
    for actor, tracks in actor_track_scores.items():
        best_track = max(tracks.keys(), key=lambda tid: sum(tracks[tid]) / len(tracks[tid]))
        best_track_per_actor[actor] = best_track

    removed = 0
    for frame in frames:
        kept = []
        for face in frame.get("faces") or []:
            actor = face.get("actor")
            track_id = face.get("track_id")
            if actor in best_track_per_actor and track_id != best_track_per_actor[actor]:
                removed += 1
                continue
            kept.append(face)
        frame["faces"] = kept
    return removed


def strip_null_actors(data: dict) -> tuple[int, int, dict[str, int]]:
    removed = 0
    kept = 0
    names: dict[str, int] = defaultdict(int)
    for frame in data.get("sampled_frames") or []:
        named = []
        for face in frame.get("faces") or []:
            if keep_named_actor(face):
                named.append(face)
                kept += 1
            else:
                removed += 1
                names[str(face.get("name") or "(no name)")] += 1
        frame["faces"] = named
    return removed, kept, dict(names)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Dedup clip_faces and drop remaining actor=null faces in place."
    )
    parser.add_argument("--root", type=Path, default=CLIP_FACES)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--progress", type=int, default=1000)
    args = parser.parse_args()

    if not args.root.exists():
        print(f"Error: clip_faces not found: {args.root}")
        return 1

    files = sorted(args.root.glob("*/*/*.json"))
    if args.limit is not None:
        files = files[: args.limit]

    print("=" * 60)
    print("Clean clip_faces (dedup + drop actor=null)")
    print("=" * 60)
    print(f"Root: {args.root}")
    print(f"JSON files: {len(files)}")
    print("Dedup was not previously applied to clip_faces.\n")

    total_null = total_dedup = total_kept = 0
    files_with_null = files_with_dedup = files_deleted = 0
    per_name: dict[str, int] = defaultdict(int)
    deleted_movies: Counter[str] = Counter()

    for i, path in enumerate(files, 1):
        data = json.loads(path.read_text())
        n_before = count_faces(data)
        dedup_removed = deduplicate_mirror_faces(data)
        null_removed, kept, names = strip_null_actors(data)
        data["faces_detected"] = kept

        total_dedup += dedup_removed
        total_null += null_removed
        total_kept += kept
        if dedup_removed:
            files_with_dedup += 1
        if null_removed:
            files_with_null += 1
        for name, n in names.items():
            per_name[name] += n

        empty = not has_named_actor(data)
        if empty:
            files_deleted += 1
            deleted_movies[f"{path.parts[-3]}/{path.parts[-2]}"] += 1
            if not args.dry_run:
                path.unlink()
        elif (dedup_removed or null_removed) and not args.dry_run:
            path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")

        if args.progress and i % args.progress == 0:
            print(
                f"  {i}/{len(files)}  null=-{total_null}  "
                f"dedup=-{total_dedup}  empty_files={files_deleted}",
                flush=True,
            )

    print("\n" + "=" * 60)
    print(f"Files scanned:          {len(files)}")
    print(f"Files with null dropped:{files_with_null}")
    print(f"Files with mirror dedup:{files_with_dedup}")
    print(f"Null faces removed:     {total_null}")
    print(f"Dedup faces removed:    {total_dedup}")
    print(f"Named faces kept:       {total_kept}")
    print(f"Empty clips deleted:    {files_deleted}")
    if per_name:
        print("Dropped null names:")
        for name, n in sorted(per_name.items(), key=lambda x: -x[1]):
            print(f"  {n:6d}  {name}")
    if deleted_movies:
        print("Deleted empty clips by movie:")
        for key, n in deleted_movies.most_common():
            print(f"  {n:5d}  {key}")
    if args.dry_run:
        print("\nDry run: no files written or deleted.")
    else:
        print(f"\nUpdated in place under {args.root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
