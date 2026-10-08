#!/usr/bin/env python3
"""Pack the per-file BAC annotations into one compressed JSONL per subset.

Inverse of ``scripts/unpack_data.py``: each line is one original JSON file,
with its path relative to ``data/`` in ``path``. These are the files hosted
on Hugging Face.

    python3 scripts/data_curation/pack_data.py --out_dir /tmp/bac_packed
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SUBSETS = {
    "shot_captions/test.jsonl.gz": "shot_captions/test/*/*/shot*/input_sequence.json",
    "shot_captions/train.jsonl.gz": "shot_captions/train/*/*/shot*/input_sequence.json",
    "clip_faces/lsmdc.jsonl.gz": "clip_faces/lsmdc/*/*.json",
    "clip_faces/mpiimd.jsonl.gz": "clip_faces/mpiimd/*/*.json",
}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", type=Path, default=PROJECT_ROOT / "data")
    p.add_argument("--out_dir", type=Path, required=True)
    args = p.parse_args()

    for archive, pattern in SUBSETS.items():
        files = sorted(args.data_dir.glob(pattern))
        out = args.out_dir / archive
        out.parent.mkdir(parents=True, exist_ok=True)
        # mtime=0 keeps the archive bytes reproducible across runs
        with open(out, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz:
            for f in files:
                record = {"path": f.relative_to(args.data_dir).as_posix(), **json.loads(f.read_text(encoding="utf-8"))}
                gz.write((json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8"))
        print(f"  + {archive}: {len(files)} files, {out.stat().st_size / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
