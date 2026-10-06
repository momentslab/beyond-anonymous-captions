"""Draw body bboxes + name labels on frames."""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import cv2
import numpy as np

BBOX_COLORS = [
    (0, 255, 0),
    (255, 0, 0),
    (0, 128, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (128, 0, 255),
    (255, 128, 0),
]

# Visualization-only palette: red → magenta!80!black; yellow → orange
# (green / magenta / blue already used for P1–P3).
VIS_BBOX_COLORS = [
    (0, 255, 0),
    (204, 0, 204),
    (0, 128, 255),
    (255, 140, 0),
    (255, 0, 255),
    (0, 255, 255),
    (128, 0, 255),
    (255, 128, 0),
]

# Bbox-proportional P-label sizing: scale = clamp(bbox_h / K, 0.4, 3.0)
# K calibrated so that typical face/body bboxes get readable labels.
# Override at runtime via env vars FACE_P_K / BODY_P_K.
FACE_P_K = int(os.environ.get("FACE_P_K", 190))   # face bbox h~80 -> scale~0.42, h~300 -> scale~1.58
BODY_P_K = int(os.environ.get("BODY_P_K", 650))    # body bbox h~460 -> scale~0.71, h~900 -> scale~1.38


def _bbox_proportional_p_scale(bbox, box_source: str):
    """Return (font_scale, thickness) proportional to bbox height."""
    k = FACE_P_K if box_source == "face" else BODY_P_K
    bbox_h = bbox[3] - bbox[1]
    scale = max(0.4, min(3.0, bbox_h / k))
    thick = max(1, round(scale))
    return scale, thick


def _is_p_index_label(name: str) -> bool:
    return bool(name) and name[0] == "P" and name[1:].isdigit()


def _labels_overlap(rect_a, rect_b):
    ax1, ay1, ax2, ay2 = rect_a
    bx1, by1, bx2, by2 = rect_b
    return ax1 < bx2 and ax2 > bx1 and ay1 < by2 and ay2 > by1


def _compute_label_rect(text_x, text_y, text_w, text_h):
    return (text_x - 2, text_y - text_h - 2, text_x + text_w + 2, text_y + 2)


def _find_non_overlapping_position(bbox, text_w, text_h, placed_rects, img_w, img_h):
    x1, y1, x2, y2 = bbox
    pad = 5
    candidates = [
        (x1 + pad, y1 + text_h + pad),
        (x2 - text_w - pad, y1 + text_h + pad),
        (x1 + pad, y2 - pad),
        (x2 - text_w - pad, y2 - pad),
        (x1 + pad, y1 - pad),
        (x2 - text_w - pad, y1 - pad),
    ]
    best_pos = candidates[0]
    best_overlap_count = float("inf")
    for text_x, text_y in candidates:
        text_x = max(2, min(text_x, img_w - text_w - 2))
        text_y = max(text_h + 2, min(text_y, img_h - 2))
        rect = _compute_label_rect(text_x, text_y, text_w, text_h)
        overlap_count = sum(1 for pr in placed_rects if _labels_overlap(rect, pr))
        if overlap_count == 0:
            return text_x, text_y
        if overlap_count < best_overlap_count:
            best_overlap_count = overlap_count
            best_pos = (text_x, text_y)
    return best_pos


def build_name_color_map(names: List[str], color_palette=None) -> Dict[str, tuple]:
    palette = color_palette or BBOX_COLORS
    color_map = {}
    seen = []
    for name in names:
        if name not in seen:
            seen.append(name)
    for i, name in enumerate(seen):
        color_rgb = palette[i % len(palette)]
        color_map[name] = (color_rgb[2], color_rgb[1], color_rgb[0])
    return color_map


def create_annotated_image(
    image: np.ndarray,
    persons: List[Dict],
    name_colors: Optional[Dict[str, tuple]] = None,
    large_p_labels: bool = False,
    box_source: str = "body",
    p_font_scale_override: Optional[float] = None,
    p_text_thickness_override: Optional[int] = None,
    color_palette=None,
) -> np.ndarray:
    annotated = image.copy()
    h, w = annotated.shape[:2]
    name_font_scale = max(0.6, min(2.0, w / 1000.0))
    box_thickness = max(2, int(w / 500.0))
    font = cv2.FONT_HERSHEY_SIMPLEX
    placed_rects = []
    palette = color_palette or BBOX_COLORS

    for i, person in enumerate(persons):
        name = person.get("name", "Unknown")
        bbox = [int(round(v)) for v in person["bbox"]]
        x1, y1, x2, y2 = bbox
        if name_colors and name in name_colors:
            color_bgr = name_colors[name]
        else:
            color_rgb = palette[i % len(palette)]
            color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])

        compact = _is_p_index_label(name) and not large_p_labels
        if _is_p_index_label(name) and p_font_scale_override is not None:
            font_scale = p_font_scale_override
            text_thickness = (
                p_text_thickness_override
                if p_text_thickness_override is not None
                else (1 if compact else box_thickness)
            )
        elif compact:
            font_scale, text_thickness = _bbox_proportional_p_scale(bbox, box_source)
        else:
            font_scale = name_font_scale
            text_thickness = box_thickness

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color_bgr, box_thickness)
        text_size, _ = cv2.getTextSize(name, font, font_scale, text_thickness)
        text_w, text_h = text_size
        text_x, text_y = _find_non_overlapping_position(
            bbox, text_w, text_h, placed_rects, w, h
        )
        rect = _compute_label_rect(text_x, text_y, text_w, text_h)
        placed_rects.append(rect)
        bg_x1, bg_y1, bg_x2, bg_y2 = rect
        cv2.rectangle(annotated, (bg_x1, bg_y1), (bg_x2, bg_y2), color_bgr, -1)
        cv2.putText(
            annotated, name, (text_x, text_y), font, font_scale,
            (255, 255, 255), text_thickness, cv2.LINE_AA,
        )
    return annotated


def _frame_grid_label(i: int, n: int, frame=None) -> str:
    """`1/8 idx=3 t=0.136s` when frame metadata is available, else `1/8`."""
    label = f"{i + 1}/{n}"
    if not frame:
        return label
    idx = frame.get("source_frame_index", frame.get("frame_index"))
    ts = frame.get("timestamp")
    if idx is not None:
        label += f" idx={int(idx)}"
    if ts is not None:
        label += f" t={float(ts):.3f}s"
    return label


def build_frame_grid(
    images: List[np.ndarray],
    cols=4,
    pad=14,
    label_h=26,
    tile_w=320,
    frames=None,
):
    n = len(images)
    if n == 0:
        return None

    sample_labels = [
        _frame_grid_label(i, n, frames[i] if frames and i < len(frames) else None)
        for i in range(n)
    ]
    if any(len(s) > 8 for s in sample_labels):
        label_h = max(label_h, 28)
        tile_w = max(tile_w, 360)

    rows = (n + cols - 1) // cols
    h, w = images[0].shape[:2]
    tile_h = max(1, int(tile_w * h / w))
    cell_h = tile_h + label_h
    grid_w = cols * tile_w + (cols + 1) * pad
    grid_h = rows * cell_h + (rows + 1) * pad
    grid = np.full((grid_h, grid_w, 3), 240, dtype=np.uint8)

    font = cv2.FONT_HERSHEY_SIMPLEX
    for i, img in enumerate(images):
        r, c = divmod(i, cols)
        x0 = pad + c * (tile_w + pad)
        y0 = pad + r * (cell_h + pad)
        resized = cv2.resize(img, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
        grid[y0:y0 + tile_h, x0:x0 + tile_w] = resized
        label = sample_labels[i]
        cv2.rectangle(grid, (x0, y0 + tile_h), (x0 + tile_w, y0 + cell_h), (40, 40, 40), -1)
        cv2.putText(
            grid, label, (x0 + 4, y0 + tile_h + label_h - 8),
            font, 0.40, (255, 255, 255), 1, cv2.LINE_AA,
        )
    return grid
