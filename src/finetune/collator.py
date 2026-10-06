"""Batching collator for Qwen3-VL multimodal fine-tuning.

Each dataset item comes already tokenized (variable length) and carries its own
``pixel_values`` / ``image_grid_thw`` tensors. This collator:

- right-pads ``input_ids`` / ``attention_mask`` / ``labels`` to the batch max
  length, using the processor's pad token (or eos as a fallback) for
  ``input_ids`` and ``-100`` for ``labels``;
- concatenates ``pixel_values`` and ``image_grid_thw`` along dim 0 (Qwen3-VL
  processors natively expect a flat list of image patches with per-image
  ``(t, h, w)`` grid entries);
- forwards optional video fields as well.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch


IGNORE_INDEX = -100


@dataclass
class Qwen3VLDataCollator:
    processor: object
    padding_side: str = "right"

    def __post_init__(self):
        tokenizer = self.processor.tokenizer
        self.pad_id = tokenizer.pad_token_id
        if self.pad_id is None:
            self.pad_id = tokenizer.eos_token_id

    def _pad_1d(self, tensors: List[torch.Tensor], pad_value: int) -> torch.Tensor:
        max_len = max(t.size(0) for t in tensors)
        out = torch.full((len(tensors), max_len), pad_value, dtype=tensors[0].dtype)
        for i, t in enumerate(tensors):
            if self.padding_side == "right":
                out[i, : t.size(0)] = t
            else:
                out[i, -t.size(0):] = t
        return out

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids = [f["input_ids"] for f in features]
        attention_mask = [f["attention_mask"] for f in features]
        labels = [f["labels"] for f in features]

        batch: Dict[str, torch.Tensor] = {
            "input_ids": self._pad_1d(input_ids, self.pad_id),
            "attention_mask": self._pad_1d(attention_mask, 0),
            "labels": self._pad_1d(labels, IGNORE_INDEX),
        }

        if "mm_token_type_ids" in features[0]:
            mm_ids = [f["mm_token_type_ids"] for f in features]
            batch["mm_token_type_ids"] = self._pad_1d(mm_ids, 0)

        for key in ("pixel_values", "pixel_values_videos"):
            parts = [f[key] for f in features if key in f]
            if parts:
                batch[key] = torch.cat(parts, dim=0)

        for key in ("image_grid_thw", "video_grid_thw"):
            parts = [f[key] for f in features if key in f]
            if parts:
                batch[key] = torch.cat(parts, dim=0)

        secs = [f["second_per_grid_ts"] for f in features if "second_per_grid_ts" in f]
        if secs:
            batch["second_per_grid_ts"] = torch.cat(secs, dim=0)

        return batch
