"""LoRA wiring + parameter freezing for Qwen3-VL.

Public API:

- ``freeze_vision_and_projector(model)``:
      freezes the vision encoder and multimodal projector (including the
      DeepStack mergers) so only the language model remains trainable.

- ``build_peft_model(model, lora_cfg)``:
      wraps the model with a PEFT LoRA adapter targeting the requested modules
      (``q_proj``, ``k_proj``, ``v_proj``, ``o_proj``, ``gate_proj``,
      ``up_proj``, ``down_proj``). Because the Qwen3-VL vision tower uses
      ``qkv``/``proj`` naming, this target list only matches the language
      model's linear layers — no adapter is added on the vision side.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List

from peft import LoraConfig, TaskType, get_peft_model


@dataclass
class LoraSettings:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    bias: str = "none"
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])


def _iter_submodules(model, prefixes: Iterable[str]):
    """Yield (name, module) whose name starts with any of ``prefixes``."""
    prefixes = tuple(prefixes)
    for name, module in model.named_modules():
        if name.startswith(prefixes):
            yield name, module


def freeze_vision_and_projector(model) -> None:
    """Freeze the vision encoder + multimodal projector (incl. DeepStack mergers).

    For ``Qwen3VLForConditionalGeneration``:

        model                                (top-level LM class)
          └── model                          (Qwen3VLModel)
                ├── visual                   (Qwen3VLVisionModel + merger + deepstack_merger_list)
                └── language_model           (Qwen3VLTextModel)  <-- trainable
          └── lm_head                        (kept trainable to match language_model)
    """
    for p in model.parameters():
        p.requires_grad = False

    unfrozen = False
    for path in ("model.model.language_model", "model.language_model"):
        obj = model
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
        except AttributeError:
            continue
        for p in obj.parameters():
            p.requires_grad = True
        unfrozen = True
        break

    if not unfrozen:
        vision_prefixes = ("model.visual", "visual")
        for name, param in model.named_parameters():
            if not any(name.startswith(p) for p in vision_prefixes):
                param.requires_grad = True


def build_peft_model(model, lora_cfg: LoraSettings):
    """Freeze vision + projector, then attach a PEFT LoRA adapter.

    Returns the wrapped ``PeftModel``.
    """
    freeze_vision_and_projector(model)

    peft_config = LoraConfig(
        r=lora_cfg.r,
        lora_alpha=lora_cfg.alpha,
        lora_dropout=lora_cfg.dropout,
        bias=lora_cfg.bias,
        target_modules=list(lora_cfg.target_modules),
        task_type=TaskType.CAUSAL_LM,
    )
    peft_model = get_peft_model(model, peft_config)

    for name, param in peft_model.named_parameters():
        if "visual" in name.split("."):
            param.requires_grad = False

    return peft_model


def count_trainable_parameters(model) -> tuple:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
