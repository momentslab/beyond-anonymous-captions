#!/usr/bin/env python3
"""Build train/eval JSONL manifests from BAC shot folders.

Default: all valid train shots go to train_manifest.jsonl, eval is empty.

    python3 scripts/prepare_manifests.py
    python3 scripts/prepare_manifests.py --eval_ratio 0.1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from finetune.prepare_manifests import build_manifests  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ann_root", type=Path,
                   default=PROJECT_ROOT / "data" / "shot_captions" / "train")
    p.add_argument("--output_dir", type=Path,
                   default=PROJECT_ROOT / "data" / "finetune")
    p.add_argument("--eval_ratio", type=float, default=0.0,
                   help="Fraction of shots to hold out as eval (default: 0 = empty eval)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--target_field", type=str, default="caption_pre")
    args = p.parse_args()

    summary = build_manifests(
        ann_root=args.ann_root,
        output_dir=args.output_dir,
        eval_ratio=args.eval_ratio,
        seed=args.seed,
        target_field=args.target_field,
    )
    print(json.dumps(summary, indent=2))
    print(f"\nTrain: {summary['train']} → {summary['train_path']}")
    print(f"Eval:  {summary['eval']} → {summary['eval_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
