"""Short-name grounding helpers (first name + collision fallback)."""

from __future__ import annotations

import re
from collections import Counter
from typing import Dict, List


def extract_last_name(full_name: str) -> str:
    parts = full_name.strip().split(maxsplit=1)
    if len(parts) < 2:
        return parts[0] if parts else "Unknown"
    return parts[1]


def extract_first_name(full_name: str) -> str:
    parts = full_name.strip().split()
    if not parts:
        return "Unknown"
    return parts[0]


def resolve_short_names(full_names: List[str]) -> Dict[str, str]:
    """
    Build full_name -> short_name mapping for grounding labels.

    Collision strategy:
      - unique first name -> use first name
      - shared first name -> fall back to last name
      - shared last name too -> impossible in practice because
        deduplicate_mirror_bboxes already removes same-actor duplicates
    """
    first_names = [extract_first_name(fn) for fn in full_names]
    first_name_counts = Counter(first_names)

    full_to_short: Dict[str, str] = {}
    for full_name, first_name in zip(full_names, first_names):
        if first_name_counts[first_name] == 1:
            full_to_short[full_name] = first_name
        else:
            full_to_short[full_name] = extract_last_name(full_name)

    return full_to_short


def assign_p_indices(full_names: List[str]) -> Dict[str, str]:
    """Assign P1, P2, P3, ... to unique people in first-seen order."""
    full_to_p: Dict[str, str] = {}
    idx = 1
    for full_name in full_names:
        if full_name in full_to_p:
            continue
        full_to_p[full_name] = f"P{idx}"
        idx += 1
    return full_to_p


def name_variants(full_name: str, short_name: str) -> List[str]:
    """Name forms that may appear in QA text for one actor."""
    variants: List[str] = []
    for value in (full_name, short_name, extract_first_name(full_name)):
        value = (value or "").strip()
        if value and value not in variants:
            variants.append(value)
    parts = full_name.strip().split()
    if len(parts) >= 2 and parts[0].endswith("."):
        two = " ".join(parts[:2])
        if two not in variants:
            variants.append(two)
    return variants


def _replace_token(text: str, source: str, dest: str) -> str:
    escaped = re.escape(source)
    if source and not source[-1].isalnum() and source[-1] != "_":
        pattern = r"\b" + escaped + r"(?=\W|$)"
    else:
        pattern = r"\b" + escaped + r"\b"
    return re.sub(pattern, dest, text)


def replace_name_variants(text: str, variant_to_label: Dict[str, str]) -> str:
    """Replace longer name variants first so 'J. K.' wins over 'J.'."""
    for variant, label in sorted(
        variant_to_label.items(), key=lambda item: len(item[0]), reverse=True
    ):
        if variant:
            text = _replace_token(text, variant, label)
    return text


def replace_p_indices(text: str, p_to_name: Dict[str, str]) -> str:
    """Replace P12 before P1."""
    for label, name in sorted(
        p_to_name.items(),
        key=lambda item: int(item[0][1:]) if item[0][1:].isdigit() else 0,
        reverse=True,
    ):
        text = _replace_token(text, label, name)
    return text


def replace_short_names_with_full(text: str, short_to_full_mapping: Dict[str, str]) -> str:
    """
    Replace grounding short names with full names (longer names first).
    
    Handles names ending with non-word characters (e.g. "J.") by using a
    negative lookahead instead of \\b at the end.
    """
    for short_name, full_name in sorted(
        short_to_full_mapping.items(), key=lambda x: len(x[0]), reverse=True
    ):
        text = _replace_token(text, short_name, full_name)
    return text
