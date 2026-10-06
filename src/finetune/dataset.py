"""PyTorch Dataset for Qwen3-VL LoRA fine-tuning on BAC shot folders.

Each shot is ``{movie}/{clip}/shotN/`` with ``input_sequence.json`` and
``frames/``. Target captions are ``caption_pre`` (P-indices) from the JSON
or from the manifest ``target_caption`` field.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import torch
from PIL import Image
from torch.utils.data import Dataset

from grounding.one_prompt import prepare_one_prompt_inputs
from prompts import IDENTITY_CAPTION_PROMPT


SYSTEM_PROMPT = "You are a helpful assistant."
IGNORE_INDEX = -100


def _read_manifest(path: Path) -> List[Dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def shot_folder_from_record(input_root: Path, record: Dict) -> Path:
    movie = record["movie"]
    clip = record["clip"]
    shot = record.get("shot")
    if shot:
        return input_root / movie / clip / shot
    return input_root / movie / clip


@dataclass
class DatasetConfig:
    manifest_path: Path
    input_root: Path
    min_pixels: int
    max_pixels: int
    target_field: str = "caption_pre"
    box_source: str = "face"
    use_p_indices: bool = True
    draw_boxes: bool = True


class GroundedCaptionDataset(Dataset):
    """Fine-tuning dataset over BAC shot folders."""

    def __init__(self, cfg: DatasetConfig, processor):
        self.cfg = cfg
        self.processor = processor
        self.records = _read_manifest(cfg.manifest_path)
        if not self.records:
            raise ValueError(f"Manifest {cfg.manifest_path} is empty")

    def __len__(self) -> int:
        return len(self.records)

    def _load_target_caption(self, record: Dict, shot_folder: Path) -> Optional[str]:
        if record.get("target_caption"):
            return record["target_caption"].strip()

        json_path = shot_folder / "input_sequence.json"
        if not json_path.exists():
            return None
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        caption = data.get(self.cfg.target_field, "") or ""
        return caption.strip() or None

    def _build_sample(self, idx: int):
        record = self.records[idx]
        clip_folder = shot_folder_from_record(self.cfg.input_root, record)
        if not (clip_folder / "input_sequence.json").exists():
            return None, None

        target = self._load_target_caption(record, clip_folder)
        if not target:
            return None, None

        try:
            prepared = prepare_one_prompt_inputs(
                clip_folder,
                prompt_template=IDENTITY_CAPTION_PROMPT,
                use_p_indices=self.cfg.use_p_indices,
                box_source=self.cfg.box_source,
                draw_boxes=self.cfg.draw_boxes,
            )
        except Exception:
            return None, None

        return prepared, target

    def _build_messages(self, prepared: Dict, target: str):
        annotated_images = prepared["annotated_images"]
        prompt = prepared["prompt"]

        pil_images = [
            Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            for img in annotated_images
        ]

        user_content = []
        for img in pil_images:
            user_content.append({
                "type": "image",
                "image": img,
                "min_pixels": self.cfg.min_pixels,
                "max_pixels": self.cfg.max_pixels,
            })
        user_content.append({"type": "text", "text": prompt})

        prompt_messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": user_content},
        ]
        full_messages = prompt_messages + [
            {"role": "assistant",
             "content": [{"type": "text", "text": target}]},
        ]
        return prompt_messages, full_messages

    def _encode(self, prepared: Dict, target: str) -> Dict[str, torch.Tensor]:
        prompt_messages, full_messages = self._build_messages(prepared, target)

        from qwen_vl_utils import process_vision_info
        image_inputs, video_inputs = process_vision_info(full_messages, image_patch_size=16)

        prompt_text = self.processor.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True,
        )
        full_text = self.processor.apply_chat_template(
            full_messages, tokenize=False, add_generation_prompt=False,
        )

        full_inputs = self.processor(
            text=[full_text],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            padding=False,
        )

        prompt_inputs = self.processor(
            text=[prompt_text],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            padding=False,
        )
        prompt_len = prompt_inputs["input_ids"].shape[1]

        input_ids = full_inputs["input_ids"][0]
        attention_mask = full_inputs["attention_mask"][0]

        labels = input_ids.clone()
        labels[:prompt_len] = IGNORE_INDEX

        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels[input_ids == pad_id] = IGNORE_INDEX

        out: Dict[str, torch.Tensor] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        for key in ("pixel_values", "image_grid_thw",
                     "pixel_values_videos", "video_grid_thw",
                     "second_per_grid_ts", "mm_token_type_ids"):
            if key in full_inputs:
                val = full_inputs[key]
                if key == "mm_token_type_ids" and val.dim() == 2:
                    val = val[0]
                out[key] = val
        return out

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        n = len(self.records)
        for offset in range(n):
            j = (idx + offset) % n
            prepared, target = self._build_sample(j)
            if prepared is None:
                continue
            try:
                return self._encode(prepared, target)
            except Exception as exc:
                print(
                    f"[dataset] skipping {self.records[j]} due to error: {exc}",
                    file=sys.stderr, flush=True,
                )
                continue
        raise RuntimeError("No valid sample found in the entire manifest.")
