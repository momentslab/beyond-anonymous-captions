#!/usr/bin/env python3
"""
Rewrite every shot JSON with the same prepare-time filters as run.py:

  1. mirror-bbox dedup  (utils.dedup.deduplicate_mirror_bboxes)
  2. drop persons with actor=null / empty  (grounding.prepare._build_prompt_persons)

The original split is not modified.

Usage:
    python scripts/data_curation/update_test_prepare_jsons.py
    python scripts/data_curation/update_test_prepare_jsons.py \\
        --src data/shot_captions/train \\
        --out data/shot_captions/train_update
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils.dedup import deduplicate_mirror_bboxes  # noqa: E402

DEFAULT_SRC = PROJECT_ROOT / "data" / "shot_captions" / "test"
DEFAULT_OUT = PROJECT_ROOT / "data" / "shot_captions" / "test_update"


def count_persons(data: dict) -> int:
    return sum(len(frame.get("persons") or []) for frame in data.get("frames") or [])


def keep_named_actor(person: dict) -> bool:
    """Same truthiness check as grounding.prepare._build_prompt_persons."""
    actor = person.get("actor")
    return bool(actor) and bool(str(actor).strip())


def strip_null_actors(data: dict) -> tuple[int, int, dict[str, int]]:
    removed = 0
    kept = 0
    characters: dict[str, int] = defaultdict(int)
    for frame in data.get("frames") or []:
        named = []
        for person in frame.get("persons") or []:
            if keep_named_actor(person):
                named.append(person)
                kept += 1
            else:
                removed += 1
                char = person.get("character") or "(no character)"
                characters[str(char)] += 1
        frame["persons"] = named
        frame["has_grounding"] = bool(named)
    return removed, kept, dict(characters)


def has_named_actor(data: dict) -> bool:
    return any(
        keep_named_actor(person)
        for frame in data.get("frames") or []
        for person in frame.get("persons") or []
    )


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Apply run.py prepare filters (dedup + drop null actors) and write "
            "cleaned shot JSONs to a new split folder."
        )
    )
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print every shot. Default prints progress every --progress shots.",
    )
    parser.add_argument("--progress", type=int, default=500)
    args = parser.parse_args()

    src_root = args.src
    out_root = args.out
    if not src_root.exists():
        print(f"Error: source not found: {src_root}")
        return 1
    if src_root.resolve() == out_root.resolve():
        print("Error: --out must differ from --src so the original split is kept")
        return 1

    paths = sorted(src_root.glob("*/*/shot*/input_sequence.json"))
    if args.limit is not None:
        paths = paths[: args.limit]

    print("=" * 60)
    print("Prepare-filter shot JSONs (dedup + drop actor=null)")
    print("=" * 60)
    print(f"Source: {src_root}")
    print(f"Output: {out_root}")
    print("Original split is not modified.")
    print(f"Shots:  {len(paths)}\n")

    total_null = 0
    total_dedup = 0
    total_kept = 0
    changed = 0
    empty_after = 0
    shots_with_null = 0
    shots_with_dedup = 0
    per_character: dict[str, int] = defaultdict(int)

    for i, src_path in enumerate(paths, 1):
        rel = src_path.relative_to(src_root)
        data = json.loads(src_path.read_text())
        n_before = count_persons(data)
        data = deduplicate_mirror_bboxes(data)
        n_after_dedup = count_persons(data)
        dedup_removed = n_before - n_after_dedup
        null_removed, kept, characters = strip_null_actors(data)

        total_dedup += dedup_removed
        total_null += null_removed
        total_kept += kept
        if dedup_removed:
            shots_with_dedup += 1
        if null_removed:
            shots_with_null += 1
        if dedup_removed or null_removed:
            changed += 1
        if not has_named_actor(data):
            empty_after += 1
        for char, n in characters.items():
            per_character[char] += n

        if args.verbose:
            note = []
            if dedup_removed:
                note.append(f"dedup -{dedup_removed}")
            if null_removed:
                note.append(f"null -{null_removed}")
            if not has_named_actor(data):
                note.append("no named actors left")
            suffix = f"  ({', '.join(note)})" if note else ""
            print(f"  {rel.parent}: kept {kept}{suffix}", flush=True)
        elif args.progress and i % args.progress == 0:
            print(
                f"  {i}/{len(paths)}  changed={changed}  "
                f"null_boxes=-{total_null}  dedup_boxes=-{total_dedup}",
                flush=True,
            )

        if not args.dry_run:
            write_json(out_root / rel, data)

    print("\n" + "=" * 60)
    print(f"Shots written:          {len(paths)}")
    print(f"Shots changed:          {changed}")
    print(f"  with null dropped:    {shots_with_null}")
    print(f"  with mirror dedup:    {shots_with_dedup}")
    print(f"Null boxes removed:     {total_null}")
    print(f"Dedup boxes removed:    {total_dedup}")
    print(f"Named boxes kept:       {total_kept}")
    print(f"Shots with no named actor left: {empty_after}")
    if per_character:
        print("Dropped null characters:")
        for char, n in sorted(per_character.items(), key=lambda x: -x[1]):
            print(f"  {n:4d}  {char}")

    if args.dry_run:
        print("\nDry run: no files written.")
        return 0

    manifest = {
        "source": str(src_root),
        "output": str(out_root),
        "shots": len(paths),
        "shots_changed": changed,
        "shots_with_null_dropped": shots_with_null,
        "shots_with_mirror_dedup": shots_with_dedup,
        "null_boxes_removed": total_null,
        "dedup_boxes_removed": total_dedup,
        "named_boxes_kept": total_kept,
        "shots_with_no_named_actors": empty_after,
        "dropped_null_characters": dict(sorted(per_character.items(), key=lambda x: -x[1])),
    }
    write_json(out_root / "manifest.json", manifest)
    print(f"\nWrote JSONs → {out_root}")
    print(f"Manifest: {out_root / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
