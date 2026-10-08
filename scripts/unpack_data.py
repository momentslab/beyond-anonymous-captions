#!/usr/bin/env python3
"""Unpack the BAC annotations downloaded from Hugging Face into per-file JSONs.

The Hub stores one compressed JSONL per subset (one line per original JSON
file, with its relative path in ``path``). This rebuilds the ``data/`` layout
the code expects:

    data/shot_captions/test.jsonl.gz   -> data/shot_captions/test/{movie}/{clip}/shot{N}/input_sequence.json
    data/shot_captions/train.jsonl.gz  -> data/shot_captions/train/...
    data/clip_faces/lsmdc.jsonl.gz     -> data/clip_faces/lsmdc/{MOVIE}/{clip}.json
    data/clip_faces/mpiimd.jsonl.gz    -> data/clip_faces/mpiimd/...

    python scripts/unpack_data.py
    python scripts/unpack_data.py --remove_archives
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ARCHIVES = [
    "shot_captions/test.jsonl.gz",
    "shot_captions/train.jsonl.gz",
    "clip_faces/lsmdc.jsonl.gz",
    "clip_faces/mpiimd.jsonl.gz",
]


def dump(record: dict) -> str:
    """Same formatting as the original files."""
    return json.dumps(record, indent=2, ensure_ascii=False) + "\n"


def unpack(archive: Path, data_dir: Path) -> int:
    n = 0
    with gzip.open(archive, "rt", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            rel = record.pop("path")
            out = data_dir / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(dump(record), encoding="utf-8")
            n += 1
    return n


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", type=Path, default=PROJECT_ROOT / "data")
    p.add_argument("--remove_archives", action="store_true",
                   help="Delete each .jsonl.gz after unpacking it")
    args = p.parse_args()

    found = 0
    for rel in ARCHIVES:
        archive = args.data_dir / rel
        if not archive.exists():
            print(f"  - {rel}: not downloaded, skipping")
            continue
        found += 1
        n = unpack(archive, args.data_dir)
        print(f"  + {rel}: {n} files")
        if args.remove_archives:
            archive.unlink()

    if not found:
        print(f"No archives found under {args.data_dir}. Download the dataset first.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
