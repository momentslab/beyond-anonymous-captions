#!/usr/bin/env python3
"""
BAC LoRA fine-tuning — Qwen3-VL 2b/4b/8b on identity captions.

Default: train on the full train split for 1 epoch, no eval.
Hyperparams (combin_2): lr=5e-5, r=16, α=32, dropout=0.05.

Usage:
    python scripts/train.py --size 2b
    python scripts/train.py --size 8b
    python scripts/train.py --size 4b --eval --eval_ratio 0.1
    python scripts/train.py --size 2b --lr 3e-5 --lora_r 8 --lora_alpha 16 --eval
    python scripts/train.py --size 2b --max_steps 20
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict

import torch
from transformers import (
    AutoProcessor,
    Qwen3VLForConditionalGeneration,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import (  # noqa: E402
    FINETUNE_LORA,
    FINETUNE_TRAINING,
    MAX_IMAGE_PIXELS,
    MIN_PIXELS,
    MODEL_PRESETS,
)
from finetune.collator import Qwen3VLDataCollator  # noqa: E402
from finetune.dataset import DatasetConfig, GroundedCaptionDataset  # noqa: E402
from finetune.lora import LoraSettings, build_peft_model, count_trainable_parameters  # noqa: E402
from finetune.prepare_manifests import build_manifests, manifest_len  # noqa: E402


ANN_TRAIN = PROJECT_ROOT / "data" / "shot_captions" / "train"
MANIFEST_DIR = PROJECT_ROOT / "data" / "finetune"
MANIFEST_TRAIN = MANIFEST_DIR / "train_manifest.jsonl"
MANIFEST_EVAL = MANIFEST_DIR / "eval_manifest.jsonl"


def _torch_dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
        "float16": torch.float16, "fp16": torch.float16,
        "float32": torch.float32, "fp32": torch.float32,
    }.get(name.lower(), torch.bfloat16)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="BAC LoRA fine-tuning — Qwen3-VL on identity captions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python scripts/train.py --size 2b
  python scripts/train.py --size 8b --eval --eval_ratio 0.1
  python scripts/train.py --size 4b --lr 3e-5 --lora_r 8 --lora_alpha 16 --eval
  python scripts/train.py --size 2b --max_steps 20
""",
    )
    p.add_argument("--size", choices=["2b", "4b", "8b"], default="2b",
                   help="Base model size (default: 2b)")
    p.add_argument("--model", type=str, default=None,
                   help="Override HF model id")
    p.add_argument("--output_dir", type=Path, default=None,
                   help="Override output directory")
    p.add_argument("--ann_root", type=Path, default=ANN_TRAIN,
                   help="Shot annotation root")
    p.add_argument("--manifest_train", type=Path, default=MANIFEST_TRAIN)
    p.add_argument("--manifest_eval", type=Path, default=MANIFEST_EVAL)

    p.add_argument("--eval", action="store_true",
                   help="Run eval during training (needs a non-empty eval split)")
    p.add_argument("--eval_ratio", type=float, default=0.0,
                   help="Hold out this fraction of train as eval and enable --eval")
    p.add_argument("--prepare", action="store_true",
                   help="Rebuild manifests before training")

    p.add_argument("--lr", "--learning_rate", dest="learning_rate",
                   type=float, default=FINETUNE_TRAINING["learning_rate"])
    p.add_argument("--lora_r", type=int, default=FINETUNE_LORA["r"])
    p.add_argument("--lora_alpha", type=int, default=FINETUNE_LORA["alpha"])
    p.add_argument("--lora_dropout", type=float, default=FINETUNE_LORA["dropout"])
    p.add_argument("--epochs", dest="num_train_epochs",
                   type=float, default=FINETUNE_TRAINING["num_train_epochs"])
    p.add_argument("--batch_size", type=int,
                   default=FINETUNE_TRAINING["per_device_train_batch_size"])
    p.add_argument("--eval_batch_size", type=int,
                   default=FINETUNE_TRAINING["per_device_eval_batch_size"])
    p.add_argument("--grad_accum", type=int,
                   default=FINETUNE_TRAINING["gradient_accumulation_steps"])
    p.add_argument("--max_steps", type=int, default=None,
                   help="Stop after N steps (smoke / debug)")
    p.add_argument("--seed", type=int, default=FINETUNE_TRAINING["seed"])
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--save_total_limit", type=int, default=3)
    p.add_argument("--eval_steps", type=int, default=200)
    p.add_argument("--resume", type=str, default=None,
                   help="Resume from a checkpoint directory")
    p.add_argument("--dtype", type=str, default="bfloat16")
    p.add_argument("--report_to", type=str, nargs="+", default=["tensorboard"])
    return p.parse_args()


class VariableEvalCallback(TrainerCallback):
    def __init__(self, dense_steps=20, dense_until=100, sparse_steps=200):
        self.dense_steps = dense_steps
        self.dense_until = dense_until
        self.sparse_steps = sparse_steps

    def on_step_end(self, args, state, control, **kwargs):
        step = state.global_step
        if step <= self.dense_until:
            if step % self.dense_steps == 0:
                control.should_evaluate = True
        else:
            if step % self.sparse_steps == 0:
                control.should_evaluate = True
        return control


class SaveBestModelCallback(TrainerCallback):
    def __init__(self, processor):
        self.best_eval_loss = float("inf")
        self.processor = processor

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return
        eval_loss = metrics.get("eval_loss")
        if eval_loss is None:
            return
        is_main = int(os.environ.get("LOCAL_RANK", "0")) == 0
        if not is_main:
            return
        if eval_loss < self.best_eval_loss:
            self.best_eval_loss = eval_loss
            best_dir = Path(args.output_dir) / "best"
            if best_dir.exists():
                shutil.rmtree(best_dir)
            best_dir.mkdir(parents=True, exist_ok=True)
            model = kwargs.get("model")
            if model is not None:
                model.save_pretrained(str(best_dir))
            self.processor.save_pretrained(str(best_dir))
            meta = {"eval_loss": eval_loss, "step": state.global_step, "epoch": state.epoch}
            with (best_dir / "best_meta.json").open("w") as f:
                json.dump(meta, f, indent=2)
            print(f"[best] New best eval_loss={eval_loss:.6f} at step {state.global_step} "
                  f"→ saved to {best_dir}", flush=True)


def _plot_loss_curves(log_history, output_dir: Path, title: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        train_data = [(e["step"], e["loss"]) for e in log_history if "loss" in e]
        eval_data = [(e["step"], e["eval_loss"]) for e in log_history if "eval_loss" in e]
        lr_data = [(e["step"], e["learning_rate"]) for e in log_history if "learning_rate" in e]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        ax = axes[0]
        if train_data:
            steps, losses = zip(*train_data)
            ax.plot(steps, losses, label="Train loss", alpha=0.7, linewidth=0.8)
        if eval_data:
            steps, losses = zip(*eval_data)
            ax.plot(steps, losses, label="Eval loss", marker="o", markersize=4,
                    linewidth=1.5, color="tab:orange")
            best_idx = list(losses).index(min(losses))
            ax.axvline(steps[best_idx], color="tab:orange", linestyle="--", alpha=0.4,
                       label=f"Best eval: {min(losses):.4f} @ step {steps[best_idx]}")
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title("Training & Validation Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        if lr_data:
            steps, vals = zip(*lr_data)
            ax.plot(steps, vals, color="tab:green", linewidth=0.8)
        ax.set_xlabel("Step")
        ax.set_ylabel("Learning Rate")
        ax.set_title("Learning Rate Schedule")
        ax.grid(True, alpha=0.3)

        fig.suptitle(title, fontsize=11)
        fig.tight_layout()
        fig.savefig(str(output_dir / "loss_curves.png"), dpi=150)
        plt.close(fig)
        print(f"Saved loss curves plot to {output_dir / 'loss_curves.png'}", flush=True)
    except Exception as exc:
        print(f"[warning] Could not generate loss plot: {exc}", flush=True)


def _ensure_manifests(args: argparse.Namespace) -> None:
    need_rebuild = (
        args.prepare
        or args.eval_ratio > 0
        or not args.manifest_train.exists()
    )
    if not need_rebuild:
        return
    print("Building manifests...", flush=True)
    summary = build_manifests(
        ann_root=args.ann_root,
        output_dir=args.manifest_train.parent,
        eval_ratio=args.eval_ratio,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2), flush=True)


def _maybe_eval_dataset(args, processor, use_eval: bool):
    n_eval = manifest_len(args.manifest_eval)
    if not use_eval:
        if n_eval:
            print(f"Eval split has {n_eval} samples but --eval was not set; "
                  "training without evaluation.", flush=True)
        return None
    if n_eval == 0:
        print("Eval requested but eval_manifest is empty. Training without eval.\n"
              "  To hold out a split:  python scripts/train.py --size "
              f"{args.size} --eval --eval_ratio 0.1", flush=True)
        return None
    return GroundedCaptionDataset(
        DatasetConfig(
            manifest_path=args.manifest_eval,
            input_root=args.ann_root,
            min_pixels=MIN_PIXELS,
            max_pixels=MAX_IMAGE_PIXELS,
            target_field="caption_pre",
            box_source="face",
            use_p_indices=True,
        ),
        processor=processor,
    )


def main() -> int:
    args = parse_args()
    use_eval = bool(args.eval or args.eval_ratio > 0)
    set_seed(args.seed)

    _ensure_manifests(args)

    preset = MODEL_PRESETS[args.size]
    model_path = args.model or preset["model"]
    output_dir = args.output_dir or (PROJECT_ROOT / "models" / "finetune" / args.size)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg_snapshot: Dict[str, Any] = {
        "size": args.size,
        "model": model_path,
        "output_dir": str(output_dir),
        "eval": use_eval,
        "eval_ratio": args.eval_ratio,
        "lora": {
            "r": args.lora_r,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
        },
        "training": {
            "learning_rate": args.learning_rate,
            "num_train_epochs": args.num_train_epochs,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "max_steps": args.max_steps,
            "seed": args.seed,
        },
        "data": {
            "ann_root": str(args.ann_root),
            "manifest_train": str(args.manifest_train),
            "manifest_eval": str(args.manifest_eval),
            "n_train": manifest_len(args.manifest_train),
            "n_eval": manifest_len(args.manifest_eval),
        },
    }
    print("=" * 60, flush=True)
    print("Effective config:", flush=True)
    print(json.dumps(cfg_snapshot, indent=2), flush=True)
    print("=" * 60, flush=True)

    dtype = _torch_dtype(args.dtype)
    print(f"Loading processor: {model_path}", flush=True)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    print(f"Loading model: {model_path}", flush=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=dtype, attn_implementation="sdpa",
        trust_remote_code=True)

    if FINETUNE_TRAINING.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    lora_cfg = LoraSettings(
        r=args.lora_r, alpha=args.lora_alpha,
        dropout=args.lora_dropout, bias=FINETUNE_LORA.get("bias", "none"),
        target_modules=list(FINETUNE_LORA["target_modules"]))
    model = build_peft_model(model, lora_cfg)
    trainable, total = count_trainable_parameters(model)
    print(f"Trainable params: {trainable:,} / {total:,} "
          f"({100.0 * trainable / total:.4f}%)", flush=True)
    model.print_trainable_parameters()

    train_ds = GroundedCaptionDataset(
        DatasetConfig(
            manifest_path=args.manifest_train,
            input_root=args.ann_root,
            min_pixels=MIN_PIXELS,
            max_pixels=MAX_IMAGE_PIXELS,
            target_field="caption_pre",
            box_source="face",
            use_p_indices=True,
        ),
        processor=processor,
    )
    eval_ds = _maybe_eval_dataset(args, processor, use_eval)
    print(f"Train samples: {len(train_ds)}"
          + (f" | Eval samples: {len(eval_ds)}" if eval_ds else " | Eval: off"),
          flush=True)

    collator = Qwen3VLDataCollator(processor=processor)

    warmup_ratio = FINETUNE_TRAINING.get("warmup_ratio", 0.03)
    n_train = len(train_ds)
    effective_batch = (
        args.batch_size
        * args.grad_accum
        * max(1, torch.cuda.device_count())
    )
    if args.max_steps:
        total_steps = args.max_steps
    else:
        total_steps = math.ceil(n_train / effective_batch) * int(args.num_train_epochs)
    warmup_steps = int(round(warmup_ratio * total_steps))
    print(f"Computed warmup: {warmup_steps} steps "
          f"({warmup_ratio:.0%} of {total_steps} total)", flush=True)

    ta_params = set(inspect.signature(TrainingArguments.__init__).parameters.keys())
    logging_dir = str(output_dir / "logs")

    ta_kwargs = dict(
        output_dir=str(output_dir),
        logging_dir=logging_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        weight_decay=FINETUNE_TRAINING.get("weight_decay", 0.0),
        warmup_steps=warmup_steps,
        lr_scheduler_type=FINETUNE_TRAINING.get("lr_scheduler_type", "cosine"),
        bf16=FINETUNE_TRAINING.get("bf16", True),
        fp16=FINETUNE_TRAINING.get("fp16", False),
        gradient_checkpointing=FINETUNE_TRAINING.get("gradient_checkpointing", True),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim=FINETUNE_TRAINING.get("optim", "adamw_torch"),
        max_grad_norm=FINETUNE_TRAINING.get("max_grad_norm", 1.0),
        seed=args.seed,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        eval_strategy="no",
        eval_steps=args.eval_steps,
        report_to=list(args.report_to),
        run_name=f"qwen3vl-{args.size}-iclr-lora",
        dataloader_num_workers=2,
        remove_unused_columns=False,
        label_names=["labels"],
    )
    if args.max_steps:
        ta_kwargs["max_steps"] = args.max_steps
    unsupported = [k for k in ta_kwargs if k not in ta_params and "kwargs" not in ta_params]
    for k in unsupported:
        print(f"  [warning] TrainingArguments does not accept '{k}', skipping", flush=True)
        del ta_kwargs[k]

    training_args = TrainingArguments(**ta_kwargs)

    callbacks = []
    if eval_ds is not None:
        callbacks.append(VariableEvalCallback(20, 100, args.eval_steps))
        callbacks.append(SaveBestModelCallback(processor=processor))
        print(f"Eval schedule: every 20 steps until step 100, "
              f"then every {args.eval_steps} steps", flush=True)

    trainer_params = inspect.signature(Trainer.__init__).parameters
    trainer_kwargs = dict(
        model=model, args=training_args, train_dataset=train_ds,
        eval_dataset=eval_ds, data_collator=collator, callbacks=callbacks)
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = processor
    elif "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = processor.tokenizer
    trainer = Trainer(**trainer_kwargs)

    trainer.train(resume_from_checkpoint=args.resume)

    is_main = int(os.environ.get("LOCAL_RANK", "0")) == 0
    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_dir))
    processor.save_pretrained(str(final_dir))
    print(f"Saved final adapter + processor to {final_dir}", flush=True)

    log_history = trainer.state.log_history
    if is_main:
        log_path = output_dir / "train_log_history.json"
        with log_path.open("w", encoding="utf-8") as f:
            json.dump(log_history, f, indent=2)
        print(f"Saved training log history ({len(log_history)} entries) to {log_path}", flush=True)
        _plot_loss_curves(
            log_history, output_dir,
            title=(f"LoRA r={args.lora_r}  α={args.lora_alpha}  "
                   f"lr={args.learning_rate}  epochs={args.num_train_epochs}  "
                   f"dropout={args.lora_dropout}"),
        )
        with (output_dir / "train_config.json").open("w", encoding="utf-8") as f:
            json.dump(cfg_snapshot, f, indent=2)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
