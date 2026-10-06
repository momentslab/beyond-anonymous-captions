"""Qwen3-VL captioner wrapper."""

from __future__ import annotations

import torch
from transformers import AutoProcessor

from config import DEFAULT_MODEL, MAX_IMAGE_PIXELS, MIN_PIXELS


def load_qwen_class():
    from transformers import Qwen3VLForConditionalGeneration
    return Qwen3VLForConditionalGeneration


def model_primary_device(model) -> torch.device:
    if hasattr(model, "device"):
        try:
            return model.device
        except Exception:
            pass
    return next(model.parameters()).device


class QwenCaptioner:
    """Load and hold a Qwen3-VL model + processor."""

    def __init__(self, model_name: str = DEFAULT_MODEL):
        Qwen3VLForConditionalGeneration = load_qwen_class()
        print(f"Loading Qwen model: {model_name}", flush=True)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            dtype="auto",
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(model_name)
        print("Qwen model loaded.", flush=True)


def build_captioner(model_name: str = None) -> QwenCaptioner:
    return QwenCaptioner(model_name or DEFAULT_MODEL)


def load_batched_hf_captioner(
    model_name: str,
    use_compile: bool = True,
    lora_path: str | None = None,
) -> QwenCaptioner:
    """Same load path as TAM's mvad_caption_from_video_grounding_vllm.py."""
    Qwen3VLForConditionalGeneration = load_qwen_class()
    print(f"Loading Qwen model: {model_name}", flush=True)
    try:
        processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
            min_pixels=MIN_PIXELS,
            max_pixels=MAX_IMAGE_PIXELS,
        )
    except TypeError:
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    processor.tokenizer.padding_side = "left"
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    import os
    # If the container's cuDNN is broken (undefined-symbol abort), the launcher
    # sets DISABLE_CUDNN=1 after a failed preflight. Turning cuDNN off makes
    # PyTorch use native CUDA convolutions for the vision encoder instead of
    # crashing (slower, but it actually runs).
    if os.environ.get("DISABLE_CUDNN") == "1":
        torch.backends.cudnn.enabled = False
        print("cuDNN DISABLED (native conv fallback) per DISABLE_CUDNN=1", flush=True)
    # Always use SDPA — it dispatches to Flash Attention (memory-efficient,
    # no cuDNN).  "eager" materializes O(n²) attention matrices and causes OOM
    # on 24 GB GPUs with the 8B model.  The env var TORCH_CUDNN_SDPA_ENABLED=0
    # (set by train_entry.py) blocks the cuDNN-specific SDPA backend, so SDPA
    # only uses FlashAttention / math backends which don't touch cuDNN.
    attn_impl = "sdpa"
    print(f"Attention implementation: {attn_impl}", flush=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation=attn_impl,
    )
    if lora_path:
        from peft import PeftModel

        print(f"Loading LoRA adapter: {lora_path}", flush=True)
        model = PeftModel.from_pretrained(model, lora_path)
        model.eval()
        print("LoRA adapter loaded.", flush=True)
    model.eval()
    if use_compile:
        print("Applying torch.compile()...", flush=True)
        model = torch.compile(model, mode="reduce-overhead")

    captioner = QwenCaptioner.__new__(QwenCaptioner)
    captioner.model = model
    captioner.processor = processor
    print("Qwen model loaded.", flush=True)
    return captioner
