"""Shared clip preparation: load frames, estimate body bboxes, annotate.

Adapted for the BAC layout where each shot folder contains:
    input_sequence.json   (metadata + persons with bbox)
    frames/frame{idx}.jpg (raw extracted frames)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

from grounding.annotate import build_name_color_map, create_annotated_image
from utils.bbox import estimate_body_bbox_from_face
from utils.dedup import deduplicate_mirror_bboxes
from utils.names import assign_p_indices, name_variants, resolve_short_names


def _build_prompt_persons(
    persons: List[Dict],
    image_w: int,
    image_h: int,
    box_source: str = "face",
) -> List[Dict]:
    """Build prompt/draw persons from face bboxes.

    ``box_source='face'`` keeps the input face box.
    ``box_source='body'`` expands it to an estimated full-body box.
    """
    result = []
    for person in persons:
        actor = person.get("actor")
        if not actor:
            continue
        face_bbox = person["bbox"]
        if box_source == "face":
            bbox = face_bbox
        else:
            bbox = estimate_body_bbox_from_face(
                face_bbox, k_h=6.5, k_w=2.5, k_offset=0.15,
                image_width=image_w, image_height=image_h,
            )
        result.append({"name": actor, "bbox": bbox, "face_bbox": face_bbox})
    return result


def load_clip(
    clip_folder: Path,
    use_p_indices: bool = True,
    box_source: str = "face",
    draw_boxes: bool = True,
    large_p_labels: bool = False,
    color_palette=None,
) -> Dict:
    """Load a shot folder and prepare annotated frames.

    Expects ``clip_folder`` to contain ``input_sequence.json`` and a
    ``frames/`` sub-directory with ``frame{idx}.jpg`` images.
    """
    input_json_path = clip_folder / "input_sequence.json"
    if not input_json_path.exists():
        raise FileNotFoundError(f"Missing input_sequence.json in {clip_folder}")

    with open(input_json_path, "r", encoding="utf-8") as f:
        input_data = json.load(f)

    input_data = deduplicate_mirror_bboxes(input_data)

    frames_data = input_data["frames"]
    if not frames_data:
        raise ValueError(f"No frames in {clip_folder}")

    image_w = input_data["frame_width"]
    image_h = input_data["frame_height"]

    # Ensure every frame has an ``image`` path relative to the shot folder.
    for frame in frames_data:
        if "image" not in frame:
            idx = frame["frame_index"]
            frame["image"] = f"frames/frame{idx}.jpg"

    seen_actors: List[str] = []
    for frame in frames_data:
        for person in frame.get("persons", []):
            actor = person.get("actor")
            if actor and actor not in seen_actors:
                seen_actors.append(actor)

    if not seen_actors:
        raise ValueError(f"No grounded persons in {clip_folder}")

    full_to_short = resolve_short_names(seen_actors)
    short_to_full = {short: full for full, short in full_to_short.items()}
    full_to_p = assign_p_indices(seen_actors) if use_p_indices else {}
    p_to_full = {label: full for full, label in full_to_p.items()}
    p_to_short = {
        full_to_p[full]: short
        for full, short in full_to_short.items()
        if full in full_to_p
    }
    variant_to_p = {}
    if use_p_indices:
        for full, label in full_to_p.items():
            for variant in name_variants(full, full_to_short[full]):
                variant_to_p[variant] = label
        display_names = [full_to_p[name] for name in seen_actors]
    else:
        display_names = list(full_to_short.values())
    actor_color_map = build_name_color_map(display_names, color_palette=color_palette)

    raw_images: List[np.ndarray] = []
    annotated_images: List[np.ndarray] = []

    for frame in frames_data:
        frame_path = clip_folder / frame["image"]
        if not frame_path.exists():
            raise FileNotFoundError(f"Missing frame: {frame_path}")

        raw_frame = cv2.imread(str(frame_path))
        if raw_frame is None:
            raise ValueError(f"Could not read frame: {frame_path}")

        raw_images.append(raw_frame)
        persons = frame.get("persons", [])

        if persons:
            gt_persons = _build_prompt_persons(
                persons, image_w, image_h, box_source=box_source,
            )
            for p in gt_persons:
                if use_p_indices:
                    p["name"] = full_to_p.get(p["name"], p["name"])
                else:
                    p["name"] = full_to_short.get(p["name"], p["name"])
            frame["_persons_for_prompt"] = gt_persons
            if draw_boxes:
                annotated = create_annotated_image(
                    raw_frame,
                    gt_persons,
                    name_colors=actor_color_map,
                    large_p_labels=large_p_labels,
                    box_source=box_source,
                    color_palette=color_palette,
                )
            else:
                annotated = raw_frame
        else:
            frame["_persons_for_prompt"] = []
            annotated = raw_frame

        annotated_images.append(annotated)

    return {
        "input_data": input_data,
        "short_to_full": short_to_full,
        "raw_images": raw_images,
        "annotated_images": annotated_images,
        "use_p_indices": use_p_indices,
        "box_source": box_source,
        "draw_boxes": draw_boxes,
        "large_p_labels": large_p_labels,
        "full_to_p": full_to_p,
        "p_to_full": p_to_full,
        "p_to_short": p_to_short,
        "variant_to_p": variant_to_p,
    }
