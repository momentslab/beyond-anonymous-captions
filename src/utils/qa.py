"""QA parsing and accuracy scoring — copied from root project."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional

from utils.names import replace_p_indices


RESULT_FILENAME = "result_identity_qa.json"
PINDICE_RESULT_FILENAME = "result_identity_qa_coord_face.json"


def _parse_json_dicts(text: str) -> List[Dict]:
    decoder = json.JSONDecoder()
    objs: List[Dict] = []
    idx = 0
    n = len(text)
    while idx < n:
        while idx < n and text[idx].isspace():
            idx += 1
        if idx >= n:
            break
        try:
            obj, end = decoder.raw_decode(text, idx)
        except json.JSONDecodeError:
            break
        if isinstance(obj, dict):
            objs.append(obj)
        idx = end
    return objs


def load_qa_result_json(path: Path) -> Dict:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return {}
    if not text.strip():
        return {}
    try:
        loaded = json.loads(text)
        return loaded if isinstance(loaded, dict) else {}
    except json.JSONDecodeError:
        pass
    objs = _parse_json_dicts(text)
    if not objs:
        return {}
    merged = dict(max(objs, key=lambda o: len(o.get("questions") or [])))
    by_id: Dict[str, Dict] = {}
    raw_by_id: Dict[str, str] = dict(merged.get("model_output_raw_by_id") or {})
    for obj in objs:
        for item in extract_answers(obj.get("model_output_parsed")):
            qid = str(item.get("id") or "").strip()
            if qid:
                by_id[qid] = item
        raw_by_id.update(obj.get("model_output_raw_by_id") or {})
    merged["model_output_parsed"] = {"answers": list(by_id.values())}
    if raw_by_id:
        merged["model_output_raw_by_id"] = raw_by_id
    return merged


def parse_qa_output(output_text: str):
    try:
        return json.loads(output_text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", output_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            return None
    return None


def parse_qa_output_plain(output_text: str, question_id: str = "Q1"):
    text = output_text.strip()
    m = re.search(r"\b(P\d{1,2})\b", text)
    if m:
        answer = m.group(1)
    elif re.search(r"\bUNIDENTIFIED\b", text, re.IGNORECASE):
        answer = "UNIDENTIFIED"
    else:
        answer = text.split("\n")[0].strip().rstrip(".")
        if len(answer) > 30:
            answer = "UNIDENTIFIED"
    return {"answers": [{"id": question_id, "answer": answer}]}


def parse_qa_output_or_plain(output_text: str, question_id: str = "Q1"):
    text = re.sub(r"<think>.*?</think>", "", output_text or "", flags=re.DOTALL).strip()
    parsed = parse_qa_output(text)
    if parsed is not None and extract_answers(parsed):
        return parsed
    match = re.search(r'"answer"\s*:\s*"([^"]+)"', text)
    if match:
        return {"answers": [{"id": question_id, "answer": match.group(1).strip()}]}
    if re.search(r"\bP\d{1,2}\b", text) or re.search(r"\bUNIDENTIFIED\b", text, re.IGNORECASE):
        return parse_qa_output_plain(text, question_id)
    return None


def compare_names_from_qa(qa_entry: dict, p_to_short: Dict[str, str], variant_to_p: Dict[str, str]) -> Dict[str, str]:
    p_to_compare = dict(p_to_short)
    for question in qa_entry.get("questions") or []:
        answer = str(question.get("answer") or "").strip()
        if not answer or answer.upper() == "UNIDENTIFIED":
            continue
        label = variant_to_p.get(answer)
        if label:
            p_to_compare[label] = answer
    return p_to_compare


def decode_p_answers(parsed, p_to_name: Dict[str, str]):
    if not isinstance(parsed, dict):
        return parsed
    decoded = dict(parsed)
    answers = []
    for item in extract_answers(parsed):
        copied = dict(item)
        answer = str(copied.get("answer") or "")
        if answer and answer.upper() != "UNIDENTIFIED":
            copied["answer_p"] = answer
            copied["answer"] = replace_p_indices(answer, p_to_name)
        answers.append(copied)
    decoded["answers"] = answers
    return decoded


def extract_answers(parsed) -> List[Dict]:
    if isinstance(parsed, dict):
        items = []
        for item in (parsed.get("answers") or []):
            if isinstance(item, dict):
                items.append(item)
            elif isinstance(item, str) and item.strip():
                items.append({"answer": item.strip()})
        return items
    return []


def parsed_answer_ids(path: Path) -> set:
    if not path.exists() or path.stat().st_size < 20:
        return set()
    data = load_qa_result_json(path)
    if not data:
        return set()
    ids = set()
    for item in extract_answers(data.get("model_output_parsed") if isinstance(data, dict) else None):
        qid = str(item.get("id") or "").strip()
        answer = str(item.get("answer") or "").strip()
        if qid and answer:
            ids.add(qid)
    return ids


def merge_separate_qa_result(
    result_path: Path,
    *,
    movie_name: str,
    clip_id: str,
    mode: str,
    full_questions: List[Dict],
    single_questions: List[Dict],
    parsed_answers,
    output_text: str,
    actors: List[str],
    name_mapping: Dict[str, str],
    num_frames: int,
    per_clip_time: float,
    model_name: str,
    extra: Optional[Dict] = None,
) -> Dict:
    existing: Dict = {}
    if result_path.exists():
        loaded = load_qa_result_json(result_path)
        if isinstance(loaded, dict):
            existing = loaded

    def _answer_item(item: Dict) -> Dict:
        out = {"id": str(item.get("id") or "").strip(), "answer": item.get("answer")}
        if item.get("answer_p"):
            out["answer_p"] = item.get("answer_p")
        return out

    by_id: Dict[str, Dict] = {}
    for item in extract_answers(existing.get("model_output_parsed")):
        qid = str(item.get("id") or "").strip()
        if qid:
            by_id[qid] = _answer_item(item)

    new_items = extract_answers(parsed_answers)
    for item in new_items:
        qid = str(item.get("id") or "").strip()
        if qid:
            by_id[qid] = _answer_item(item)
    if len(single_questions) == 1 and new_items:
        qid = str(single_questions[0].get("id") or "").strip()
        if qid and (qid not in by_id or not str(by_id[qid].get("answer") or "").strip()):
            by_id[qid] = _answer_item({"id": qid, **new_items[0]})

    raw_by_id = dict(existing.get("model_output_raw_by_id") or {})
    for question in single_questions:
        qid = str(question.get("id") or "").strip()
        if qid:
            raw_by_id[qid] = output_text

    ordered = []
    for question in full_questions:
        qid = str(question.get("id") or "").strip()
        if qid in by_id:
            ordered.append(by_id[qid])

    pred_by_id = {str(a.get("id") or "").strip(): a for a in ordered}
    enriched_questions = []
    for q in full_questions:
        qid = str(q.get("id") or "").strip()
        eq = dict(q)
        pred_item = pred_by_id.get(qid)
        if pred_item:
            eq["predicted"] = pred_item.get("answer") or ""
            if pred_item.get("answer_p"):
                eq["predicted_p"] = pred_item["answer_p"]
        else:
            eq["predicted"] = ""
        enriched_questions.append(eq)

    result = {
        "mode": mode,
        "separate_questions": True,
        "engine": "batched_hf_compile",
        "model": model_name,
        "movie": movie_name,
        "clip": clip_id,
        "num_frames": num_frames,
        "num_unique_actors": len(actors),
        "actors": list(actors),
        "name_mapping": name_mapping or {},
        "questions": enriched_questions,
        "model_output_raw": output_text,
        "model_output_raw_by_id": raw_by_id,
        "model_output_parsed": {"answers": ordered},
        "inference_time_seconds": float(existing.get("inference_time_seconds") or 0) + per_clip_time,
    }
    if extra:
        result.update(extra)
    return result


def _norm_answer(value) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _person_key_map(data: Dict) -> Dict[str, str]:
    mapping = {"unidentified": "UNIDENTIFIED"}
    p_to_full = data.get("p_to_full") or {}
    p_to_short = data.get("p_to_short") or data.get("name_mapping") or {}
    short_to_full = data.get("name_mapping") or {}
    if p_to_full:
        for label, full in p_to_full.items():
            short = p_to_short.get(label, "")
            key = full or short or label
            for value in (full, short, label):
                n = _norm_answer(value)
                if n:
                    mapping[n] = key
            if full:
                parts = str(full).split()
                if parts:
                    mapping[_norm_answer(parts[0])] = key
                if len(parts) >= 2:
                    mapping[_norm_answer(" ".join(parts[:2]))] = key
    else:
        for short, full in short_to_full.items():
            key = full or short
            for value in (full, short):
                n = _norm_answer(value)
                if n:
                    mapping[n] = key
    return mapping


def _canonical(answer, mapping: Dict[str, str]) -> str:
    n = _norm_answer(answer)
    return mapping.get(n, n)


def score_identity_qa_dir(output_root: Path, result_filename: str = PINDICE_RESULT_FILENAME) -> Dict:
    paths = sorted(output_root.glob(f"**/{result_filename}"))
    by_type: Dict[str, Dict[str, int]] = {}
    named_total = named_correct = 0
    unid_total = unid_correct = 0
    exact = 0
    questions_total = 0
    missing = 0
    clips_all_correct = 0

    for path in paths:
        data = load_qa_result_json(path)
        if not data:
            continue
        questions = {
            str(q.get("id") or "").strip(): q
            for q in (data.get("questions") or [])
            if str(q.get("id") or "").strip()
        }
        preds = {
            str(a.get("id") or "").strip(): a
            for a in extract_answers(data.get("model_output_parsed"))
        }
        mapping = _person_key_map(data)
        clip_ok = bool(questions)
        for qid, question in questions.items():
            questions_total += 1
            qtype = str(question.get("type") or "unknown")
            stats = by_type.setdefault(qtype, {"total": 0, "exact": 0, "correct": 0, "missing": 0})
            stats["total"] += 1
            gt = question.get("answer")
            pred_item = preds.get(qid)
            pred = (pred_item or {}).get("answer")
            if pred_item is None or not str(pred or "").strip():
                missing += 1
                stats["missing"] += 1
                clip_ok = False
                continue
            is_exact = _norm_answer(pred) == _norm_answer(gt)
            is_correct = _canonical(pred, mapping) == _canonical(gt, mapping)
            if is_exact:
                exact += 1
                stats["exact"] += 1
            if is_correct:
                stats["correct"] += 1
            else:
                clip_ok = False
            if _norm_answer(gt) == "unidentified":
                unid_total += 1
                unid_correct += int(is_correct)
            else:
                named_total += 1
                named_correct += int(is_correct)
        if clip_ok:
            clips_all_correct += 1

    def _pct(num, den):
        return round(100.0 * num / den, 2) if den else 0.0

    type_rows = {}
    for qtype, stats in sorted(by_type.items()):
        type_rows[qtype] = {
            **stats,
            "accuracy_percent": _pct(stats["correct"], stats["total"]),
            "exact_percent": _pct(stats["exact"], stats["total"]),
        }

    return {
        "clips": len(paths),
        "questions": questions_total,
        "predicted": questions_total - missing,
        "missing": missing,
        "exact_match": exact,
        "exact_accuracy_percent": _pct(exact, questions_total),
        "canonical_match": sum(s["correct"] for s in by_type.values()),
        "accuracy_percent": _pct(sum(s["correct"] for s in by_type.values()), questions_total),
        "named": {
            "total": named_total,
            "correct": named_correct,
            "accuracy_percent": _pct(named_correct, named_total),
        },
        "unidentified": {
            "total": unid_total,
            "correct": unid_correct,
            "accuracy_percent": _pct(unid_correct, unid_total),
        },
        "clips_all_correct": clips_all_correct,
        "clips_all_correct_percent": _pct(clips_all_correct, len(paths)),
        "by_type": type_rows,
        "result_filename": result_filename,
    }


def write_identity_qa_accuracies(
    output_root: Path,
    mode: str,
    result_filename: str,
) -> Path:
    accuracies = score_identity_qa_dir(output_root, result_filename=result_filename)
    accuracies["mode"] = mode
    path = output_root / f"{mode}_accuracies.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(accuracies, f, indent=2, ensure_ascii=False)
    return path
