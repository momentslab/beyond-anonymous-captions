#!/usr/bin/env python3
"""
Add an ``actor`` field to every face in data/clip_faces.

Character lookup is looser than exact string match: slash aliases
(``Malcolm Turner / Big Momma``), punctuation (``Dr. Perkins``),
rank abbreviations (``ssgt``), and unique first/last names in that
movie's cast. Faces are never dropped. A remaining miss is ``null``.

Usage:
    python3 scripts/data_curation/add_actor_to_clip_faces.py
    python3 scripts/data_curation/add_actor_to_clip_faces.py --dry_run --limit 20
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLIP_FACES = PROJECT_ROOT / "data" / "clip_faces"
# LSMDC release folder holding movie_actor_character_dictionary_{lsmdc,mpiimd}.json
LSMDC_DATA = Path(os.environ.get("LSMDC_DATA_ROOT", PROJECT_ROOT / "data"))
DICT_LSMDC = LSMDC_DATA / "movie_actor_character_dictionary_lsmdc.json"
DICT_MPIIMD = LSMDC_DATA / "movie_actor_character_dictionary_mpiimd.json"

CLIP_MOVIE_RE = re.compile(r"^(\d+_.+?)_\d{2}\.\d{2}\.\d{2}")


def _normalize(name: str) -> str:
    return name.upper().replace("-", "_")


def strip_movie_prefix(name: str) -> str:
    if "_" not in name:
        return name
    prefix, rest = name.split("_", 1)
    return rest if prefix.isdigit() else name


def load_actor_dictionaries() -> dict[str, list]:
    combined: dict[str, list] = {}
    for dict_path in (DICT_LSMDC, DICT_MPIIMD):
        if not dict_path.exists():
            print(f"Warning: dictionary not found: {dict_path}")
            continue
        data = json.loads(dict_path.read_text())
        for movie_key, actor_list in data.items():
            if not isinstance(actor_list, list):
                continue
            combined[_normalize(movie_key)] = actor_list
            combined[_normalize(strip_movie_prefix(movie_key))] = actor_list
    return combined


def get_actor_list(movie_folder: str, clip_stem: str, actor_dicts: dict) -> list:
    candidates = [movie_folder, strip_movie_prefix(movie_folder)]
    match = CLIP_MOVIE_RE.match(clip_stem)
    if match:
        prefixed = match.group(1)
        candidates.extend([prefixed, strip_movie_prefix(prefixed)])
    for candidate in candidates:
        key = _normalize(candidate)
        if key in actor_dicts:
            return actor_dicts[key]
    return []


def alphanorm(value: str) -> str:
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


RANK_EXPANSIONS = {
    "ssgt": "staff sergeant",
    "sgt": "sergeant",
    "cpl": "corporal",
    "pfc": "private first class",
    "tsgt": "technical sergeant",
    "lcpl": "lance corporal",
    "lt": "lieutenant",
    "dr": "doctor",
    "mr": "mister",
    "mrs": "missus",
    "ms": "miss",
}


def expand_tokens(value: str) -> str:
    parts = []
    for token in alphanorm(value).split():
        parts.extend(RANK_EXPANSIONS.get(token, token).split())
    return " ".join(parts)


def character_aliases(character: str) -> list[str]:
    """Slash aliases, punctuation stripped, ranks expanded."""
    aliases = []
    seen = set()
    chunks = [character] + [p.strip() for p in character.split("/") if p.strip()]
    for chunk in chunks:
        chunk = re.sub(r"[\"'`]", "", chunk)
        chunk = re.sub(r"[()]", " ", chunk)
        alias = expand_tokens(chunk)
        if alias and alias not in seen:
            seen.add(alias)
            aliases.append(alias)
    return aliases


def build_resolver(actor_list: list):
    """Map gallery character names to a unique actor for one movie."""
    exact: dict[str, str] = {}
    collisions: set[str] = set()
    first_tokens: dict[str, set[str]] = defaultdict(set)
    last_tokens: dict[str, set[str]] = defaultdict(set)

    for entry in actor_list:
        actor = str(entry.get("actor") or "").strip()
        character = str(entry.get("character") or "")
        if not actor or not character:
            continue
        for alias in character_aliases(character):
            if alias in exact and exact[alias] != actor:
                collisions.add(alias)
            else:
                exact[alias] = actor
            tokens = alias.split()
            first_tokens[tokens[0]].add(actor)
            last_tokens[tokens[-1]].add(actor)
    for alias in collisions:
        exact.pop(alias, None)

    def resolve(char_name: str) -> str | None:
        gallery = expand_tokens(char_name)
        if not gallery:
            return None
        if gallery in exact:
            return exact[gallery]

        gtoks = gallery.split()
        if len(gtoks) >= 2:
            for alias, actor in exact.items():
                atoks = alias.split()
                if all(token in atoks for token in gtoks):
                    return actor
                if len(atoks) >= 2 and all(token in gtoks for token in atoks):
                    return actor

        first = gtoks[0]
        if len(first_tokens[first]) == 1:
            return next(iter(first_tokens[first]))
        if len(gtoks) == 1 and len(last_tokens[first]) == 1:
            return next(iter(last_tokens[first]))
        return None

    return resolve


def add_actor_fields(data: dict, resolve) -> tuple[int, int, int]:
    resolved = missed = empty_name = 0
    for frame in data.get("sampled_frames") or []:
        for face in frame.get("faces") or []:
            name = face.get("name")
            if name is None or not str(name).strip():
                face["actor"] = None
                empty_name += 1
                continue
            actor = resolve(str(name)) if resolve else None
            face["actor"] = actor
            if actor:
                resolved += 1
            else:
                missed += 1
    return resolved, missed, empty_name


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Add actor names to clip_faces JSONs from the character dictionaries."
    )
    parser.add_argument("--root", type=Path, default=CLIP_FACES)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--progress", type=int, default=1000)
    args = parser.parse_args()

    if not args.root.exists():
        print(f"Error: clip_faces not found: {args.root}")
        return 1

    actor_dicts = load_actor_dictionaries()
    files = sorted(args.root.glob("*/*/*.json"))
    if args.limit is not None:
        files = files[: args.limit]

    print("=" * 60)
    print("Add actor field to clip_faces (no faces dropped)")
    print("=" * 60)
    print(f"Root: {args.root}")
    print(f"Dictionary movies: {len({k for k in actor_dicts if k[:1].isdigit()})}")
    print(f"JSON files: {len(files)}\n")

    total_resolved = total_missed = total_empty = 0
    files_no_dict = 0
    files_changed = 0
    missing_movies: Counter[str] = Counter()

    for i, path in enumerate(files, 1):
        dataset, movie = path.parts[-3], path.parts[-2]
        actor_list = get_actor_list(movie, path.stem, actor_dicts)
        if not actor_list:
            files_no_dict += 1
            missing_movies[f"{dataset}/{movie}"] += 1
            resolve = None
        else:
            resolve = build_resolver(actor_list)

        data = json.loads(path.read_text())
        resolved, missed, empty_name = add_actor_fields(data, resolve)
        total_resolved += resolved
        total_missed += missed
        total_empty += empty_name
        files_changed += 1

        if not args.dry_run:
            path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")

        if args.progress and i % args.progress == 0:
            print(
                f"  {i}/{len(files)}  actor={total_resolved}  "
                f"null={total_missed}  empty_name={total_empty}",
                flush=True,
            )

    print("\n" + "=" * 60)
    print(f"Files:              {len(files)}")
    print(f"Faces with actor:   {total_resolved}")
    print(f"Faces actor=null:   {total_missed}")
    print(f"Faces empty name:   {total_empty}")
    print(f"Files with no dict: {files_no_dict}")
    if missing_movies:
        print("Movies with no dictionary match:")
        for key, n in missing_movies.most_common():
            print(f"  {n:5d}  {key}")
    if args.dry_run:
        print("\nDry run: no files written.")
    else:
        print(f"\nWrote actor fields in place under {args.root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
