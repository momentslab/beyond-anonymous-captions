#!/usr/bin/env python3
"""
Check Qwen3-8B coord_face QA JSONs one by one: null / unrecognized actors
must not appear.

A person is a "null actor" in the source shot captions when ``actor`` is
missing. This script:

  1. Walks every result JSON under the QA output root.
  2. Fails if any person box still has actor=null / empty.
  3. Fails if a source-shot character that had actor=null leaked into
     ``actors``, ``p_to_full``, or ``frames_metadata`` person names.

Usage:
    python3 scripts/data_curation/test_null_actors_absent.py
    python3 scripts/data_curation/test_null_actors_absent.py --qa_root /path/to/coord_face
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PROJECT_ROOT.parent

SHOT_RE = re.compile(r"^(?P<clip>.+)_shot(?P<shot>\d+)$")

DEFAULT_QA_ROOT = (
    WORKSPACE_ROOT / "outputs" / "output_section1" / "Qwen3_8b_qa" / "coord_face"
)
DEFAULT_SRC_ROOT = PROJECT_ROOT / "data" / "shot_captions" / "test"


def is_null_actor(person: dict) -> bool:
    actor = person.get("actor")
    return actor is None or not str(actor).strip()


def norm(value) -> str:
    return str(value or "").strip().lower()


def source_null_characters(src_json: Path) -> set[str]:
    data = json.loads(src_json.read_text())
    names = set()
    for frame in data.get("frames") or []:
        for person in frame.get("persons") or []:
            if is_null_actor(person):
                char = norm(person.get("character"))
                if char:
                    names.add(char)
    return names


def load_null_map(src_root: Path) -> dict[tuple[str, str, str], set[str]]:
    mapping: dict[tuple[str, str, str], set[str]] = {}
    for path in sorted(src_root.glob("*/*/shot*/input_sequence.json")):
        movie, clip, shot = path.parts[-4], path.parts[-3], path.parts[-2]
        names = source_null_characters(path)
        if names:
            mapping[(movie, clip, shot)] = names
    return mapping


def result_identities(data: dict) -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    for actor in data.get("actors") or []:
        found.append(("actors", str(actor)))
    for key, val in (data.get("p_to_full") or {}).items():
        found.append((f"p_to_full.{key}", str(val)))
    for key, val in (data.get("name_mapping") or {}).items():
        found.append((f"name_mapping.{key}", str(val)))
    for frame_i, frame in enumerate(data.get("frames_metadata") or []):
        for person_i, person in enumerate(frame.get("persons") or []):
            found.append((
                f"frames_metadata[{frame_i}].persons[{person_i}].actor",
                str(person.get("actor")),
            ))
    return found


def null_actor_fields(data: dict) -> list[str]:
    hits = []
    for frame_i, frame in enumerate(data.get("frames_metadata") or []):
        for person_i, person in enumerate(frame.get("persons") or []):
            if is_null_actor(person):
                hits.append(
                    f"frames_metadata[{frame_i}].persons[{person_i}].actor"
                    f"={person.get('actor')!r}"
                )
    for actor in data.get("actors") or []:
        if actor is None or not str(actor).strip():
            hits.append(f"actors contains {actor!r}")
    return hits


def parse_result_clip(folder_name: str) -> tuple[str, str] | None:
    match = SHOT_RE.match(folder_name)
    if not match:
        return None
    return match.group("clip"), f"shot{int(match.group('shot'))}"


def check_one(
    path: Path,
    qa_root: Path,
    null_map: dict[tuple[str, str, str], set[str]],
) -> list[str]:
    failures: list[str] = []
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return [f"cannot read JSON: {exc}"]

    failures.extend(null_actor_fields(data))

    rel = path.relative_to(qa_root)
    movie = rel.parts[0]
    parsed = parse_result_clip(rel.parts[1])
    if parsed is None:
        failures.append(f"cannot parse clip/shot from folder {rel.parts[1]}")
        return failures

    clip, shot = parsed
    forbidden = null_map.get((movie, clip, shot), set())
    if not forbidden:
        return failures

    for where, value in result_identities(data):
        if norm(value) in forbidden:
            failures.append(
                f"{where}={value!r} matches source null-actor character"
            )
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(
        description="One-by-one check that QA JSONs contain no null actors."
    )
    parser.add_argument("--qa_root", type=Path, default=DEFAULT_QA_ROOT)
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC_ROOT)
    parser.add_argument(
        "--fail_fast",
        action="store_true",
        help="Stop at the first failing JSON",
    )
    args = parser.parse_args()

    qa_root = args.qa_root
    src_root = args.src
    if not qa_root.exists():
        print(f"Error: QA root not found: {qa_root}")
        return 1
    if not src_root.exists():
        print(f"Error: source captions not found: {src_root}")
        return 1

    null_map = load_null_map(src_root)
    files = sorted(qa_root.glob("*/*/result_identity_qa_coord_face.json"))
    print("=" * 60)
    print("Null-actor absence check (one file at a time)")
    print("=" * 60)
    print(f"QA root: {qa_root}")
    print(f"Source:  {src_root}")
    print(f"Source shots with actor=null: {len(null_map)}")
    print(f"QA JSON files: {len(files)}\n")

    if not files:
        print("Error: no result_identity_qa_coord_face.json files found")
        return 1

    failed: list[tuple[str, list[str]]] = []
    affected_checked = 0
    for i, path in enumerate(files, 1):
        rel = str(path.relative_to(qa_root))
        parsed = parse_result_clip(path.parent.name)
        key = None
        if parsed is not None:
            key = (path.relative_to(qa_root).parts[0], parsed[0], parsed[1])
            if key in null_map:
                affected_checked += 1
        problems = check_one(path, qa_root, null_map)
        tag = "affected" if key in null_map else "other"
        if problems:
            print(f"[{i}/{len(files)}] FAIL {tag}  {rel}")
            for problem in problems:
                print(f"         - {problem}")
            failed.append((rel, problems))
            if args.fail_fast:
                break
        else:
            print(f"[{i}/{len(files)}] OK   {tag}  {rel}")

    print("\n" + "=" * 60)
    print(
        f"Checked {len(files)} JSON files "
        f"({affected_checked} matched a source shot with actor=null)."
    )
    if failed:
        print(f"FAILED: {len(failed)} file(s)")
        for rel, problems in failed:
            print(f"  {rel}")
            for problem in problems:
                print(f"    - {problem}")
        return 1

    print("PASSED: no null actors and no leaked unrecognized characters.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
