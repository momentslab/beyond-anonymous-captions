"""Multi-frame generation helper for Qwen3-VL."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from config import (
    MAX_IMAGE_PIXELS,
    MAX_NEW_TOKENS,
    MIN_PIXELS,
    REPETITION_PENALTY,
    TEMPERATURE,
    TOP_K,
    TOP_P,
)
from utils.bbox import bbox_pixels_to_1000
from utils.captioner import model_primary_device


def _to_pil(img) -> Image.Image:
    if isinstance(img, np.ndarray):
        return Image.fromarray(img[:, :, ::-1])
    if isinstance(img, Image.Image):
        return img
    return img


def generate_multiframe(
    captioner,
    images,
    formatted_prompt,
    max_new_tokens=MAX_NEW_TOKENS,
    min_pixels=MIN_PIXELS,
    max_pixels=MAX_IMAGE_PIXELS,
    temperature=TEMPERATURE,
    top_p=TOP_P,
    top_k=TOP_K,
    repetition_penalty=REPETITION_PENALTY,
):
    content = []
    for img in images:
        image_content = {"type": "image", "image": _to_pil(img)}
        if min_pixels is not None:
            image_content["min_pixels"] = min_pixels
        if max_pixels is not None:
            image_content["max_pixels"] = max_pixels
        content.append(image_content)

    content.append({"type": "text", "text": formatted_prompt})
    messages = [{"role": "user", "content": content}]

    from qwen_vl_utils import process_vision_info

    image_inputs, video_inputs = process_vision_info(messages, image_patch_size=16)
    text = captioner.processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = captioner.processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
    ).to(model_primary_device(captioner.model))

    with torch.inference_mode():
        outputs = captioner.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            do_sample=temperature > 0,
        )

    generated_ids = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, outputs)
    ]
    return captioner.processor.batch_decode(
        generated_ids, skip_special_tokens=True,
    )[0].strip()


def generate_multiframe_batch(
    captioner,
    batch_images,
    batch_prompts,
    max_new_tokens=MAX_NEW_TOKENS,
    min_pixels=MIN_PIXELS,
    max_pixels=MAX_IMAGE_PIXELS,
    temperature=TEMPERATURE,
    top_p=TOP_P,
    top_k=TOP_K,
    repetition_penalty=REPETITION_PENALTY,
):
    """Batched HF generate, matching TAM's mvad_caption_from_video_grounding_vllm.py."""
    from qwen_vl_utils import process_vision_info

    batch_texts = []
    all_images = []
    all_videos = []
    for images, prompt in zip(batch_images, batch_prompts):
        content = []
        for img in images:
            image_content = {"type": "image", "image": _to_pil(img)}
            if min_pixels is not None:
                image_content["min_pixels"] = min_pixels
            if max_pixels is not None:
                image_content["max_pixels"] = max_pixels
            content.append(image_content)
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        batch_texts.append(
            captioner.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        )
        image_inputs, video_inputs = process_vision_info(messages, image_patch_size=16)
        if image_inputs:
            all_images.extend(image_inputs)
        if video_inputs:
            all_videos.extend(video_inputs)

    inputs = captioner.processor(
        text=batch_texts,
        images=all_images or None,
        videos=all_videos or None,
        return_tensors="pt",
        padding=True,
    ).to(model_primary_device(captioner.model))

    with torch.inference_mode():
        outputs = captioner.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            temperature=temperature if temperature > 0 else None,
            top_p=top_p if temperature > 0 else None,
            top_k=top_k if temperature > 0 else None,
            repetition_penalty=repetition_penalty,
            do_sample=temperature > 0,
        )

    generated_ids = [
        out_ids[inputs.input_ids[i].shape[0]:]
        for i, out_ids in enumerate(outputs)
    ]
    decoded = captioner.processor.batch_decode(
        generated_ids, skip_special_tokens=True,
    )
    del inputs, outputs
    torch.cuda.empty_cache()
    return [text.strip() for text in decoded]


def build_multiframe_boxes_string(frames_data, image_w, image_h):
    sections = []
    has_any = False
    for frame in frames_data:
        order = frame["frame_order"]
        persons_for_prompt = frame.get("_persons_for_prompt", [])
        if not persons_for_prompt:
            continue
        has_any = True
        box_lines = []
        for person in persons_for_prompt:
            ymin, xmin, ymax, xmax = bbox_pixels_to_1000(person["bbox"], image_w, image_h)
            box_lines.append(
                f"<ref>{person['name']}</ref><box>[{ymin},{xmin},{ymax},{xmax}]</box>"
            )
        sections.append(f"Frame {order + 1}:\n" + "\n".join(box_lines))

    if not has_any:
        return "(No named-person annotations available.)"
    return "Named people visible in this clip:\n" + "\n\n".join(sections)
