"""One-prompt identity grounding for captioning and QA — BAC version."""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2

from grounding.annotate import build_frame_grid
from grounding.generate import build_multiframe_boxes_string, generate_multiframe
from grounding.prepare import load_clip
from config import MAX_NEW_TOKENS
from prompts import IDENTITY_CAPTION_PROMPT, IDENTITY_QA_PROMPT
from utils.names import replace_name_variants, replace_short_names_with_full
from utils.text_clean import clean_awkward_phrasings


def prepare_one_prompt_inputs(
    clip_folder: Path,
    prompt_template: str = IDENTITY_CAPTION_PROMPT,
    use_p_indices: bool = True,
    box_source: str = "face",
    draw_boxes: bool = True,
    large_p_labels: bool = False,
    no_text_coords: bool = False,
):
    """Load a clip and build the one-prompt identity grounding request."""
    prepared = load_clip(
        clip_folder,
        use_p_indices=use_p_indices,
        box_source=box_source,
        draw_boxes=draw_boxes,
        large_p_labels=large_p_labels,
    )
    input_data = prepared["input_data"]
    frames = input_data["frames"]
    image_w = input_data["frame_width"]
    image_h = input_data["frame_height"]
    short_to_full = prepared["short_to_full"]
    if no_text_coords:
        input_boxes_string = ""
    else:
        input_boxes_string = build_multiframe_boxes_string(frames, image_w, image_h)
    prepared["frames"] = frames
    prepared["image_w"] = image_w
    prepared["image_h"] = image_h
    prepared["actors"] = list(short_to_full.values())
    prepared["prompt"] = prompt_template.format(input_boxes_string=input_boxes_string)
    return prepared


def encode_questions_with_p_indices(questions, variant_to_p: dict) -> list:
    """Copy questions and replace names with P1, P2, ... for the model."""
    encoded = []
    for question in questions or []:
        item = dict(question)
        if item.get("question"):
            item["question"] = replace_name_variants(str(item["question"]), variant_to_p)
        encoded.append(item)
    return encoded


def format_questions_string(questions) -> str:
    """Format identity QA questions."""
    lines = []
    for question in questions or []:
        qid = str(question.get("id") or "").strip()
        text = str(question.get("question") or "").strip()
        if qid and text:
            lines.append(f"{qid}: {text}")
    return "\n".join(lines)


def prepare_one_prompt_qa_inputs(
    clip_folder: Path,
    qa_entry: dict,
    prompt_template: str = IDENTITY_QA_PROMPT,
    use_p_indices: bool = True,
    box_source: str = "face",
    draw_boxes: bool = True,
    large_p_labels: bool = False,
):
    """Load a clip with one-prompt path, then attach QA text.

    Placeholders are replaced (not str.format) because the QA template
    contains JSON braces.
    """
    prepared = prepare_one_prompt_inputs(
        clip_folder,
        prompt_template="{input_boxes_string}",
        use_p_indices=use_p_indices,
        box_source=box_source,
        draw_boxes=draw_boxes,
        large_p_labels=large_p_labels,
    )
    prepared["input_boxes_string"] = prepared["prompt"]
    return apply_qa_prompt(
        prepared,
        qa_entry,
        prompt_template=prompt_template,
        use_p_indices=use_p_indices,
    )


def apply_qa_prompt(
    prepared: dict,
    qa_entry: dict,
    prompt_template: str = IDENTITY_QA_PROMPT,
    use_p_indices: bool = True,
):
    """Reuse an already-loaded clip and only swap the question text."""
    boxes = prepared.get("input_boxes_string") or prepared["prompt"]
    questions = qa_entry.get("questions") or []
    encoded_questions = questions
    if use_p_indices:
        encoded_questions = encode_questions_with_p_indices(
            questions, prepared.get("variant_to_p") or {},
        )
    questions_string = format_questions_string(encoded_questions)
    updated = dict(prepared)
    updated["prompt"] = prompt_template.replace(
        "{input_boxes_string}", boxes
    ).replace(
        "{questions_string}", questions_string
    )
    updated["qa_entry"] = qa_entry
    updated["qa_entry_encoded"] = {**qa_entry, "questions": encoded_questions}
    return updated
