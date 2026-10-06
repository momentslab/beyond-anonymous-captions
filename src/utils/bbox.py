"""Body-bbox estimation and coordinate conversion."""

from __future__ import annotations

from typing import List, Optional


def estimate_body_bbox_from_face(
    face_bbox: List[int],
    k_h: float = 6.5,
    k_w: float = 2.5,
    k_offset: float = 0.15,
    image_width: Optional[int] = None,
    image_height: Optional[int] = None,
) -> List[int]:
    """Estimate a full-body [x1, y1, x2, y2] from a face bbox."""
    x1_f, y1_f, x2_f, y2_f = face_bbox
    w_f = x2_f - x1_f
    h_f = y2_f - y1_f

    h_b = int(k_h * h_f)
    w_b = int(k_w * w_f)

    x1_b = int((x1_f + w_f / 2) - (w_b / 2))
    y1_b = int(y1_f - (k_offset * h_f))
    x2_b = x1_b + w_b
    y2_b = y1_b + h_b

    x1_b = max(0, x1_b)
    y1_b = max(0, y1_b)
    if image_width is not None:
        x2_b = min(image_width, x2_b)
    if image_height is not None:
        y2_b = min(image_height, y2_b)

    return [x1_b, y1_b, x2_b, y2_b]


def bbox_pixels_to_1000(bbox_pixels: List[int], image_w: int, image_h: int) -> List[int]:
    """Convert pixel [x1,y1,x2,y2] to Qwen-style [ymin,xmin,ymax,xmax] on 0-1000."""
    x1, y1, x2, y2 = bbox_pixels
    xmin = max(0, min(1000, int(round((x1 / image_w) * 1000))))
    ymin = max(0, min(1000, int(round((y1 / image_h) * 1000))))
    xmax = max(0, min(1000, int(round((x2 / image_w) * 1000))))
    ymax = max(0, min(1000, int(round((y2 / image_h) * 1000))))
    return [ymin, xmin, ymax, xmax]
