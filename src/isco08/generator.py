from __future__ import annotations

import json
import re
from typing import Iterable, Sequence

import pandas as pd


def format_candidate_block(
    candidates: pd.DataFrame,
    indices: Sequence[int],
    max_examples: int = 10,
    description_chars: int = 900,
) -> str:
    blocks: list[str] = []
    for raw_index in indices:
        index = int(raw_index)
        row = candidates.iloc[index]
        description = str(row.get("description", "")).strip()
        if len(description) > description_chars:
            description = description[:description_chars].rstrip()
        examples = row.get("examples", [])
        if isinstance(examples, str):
            try:
                examples = json.loads(examples)
            except json.JSONDecodeError:
                examples = []
        if not isinstance(examples, list):
            examples = []
        examples_text = "; ".join(str(item) for item in examples[:max_examples])
        blocks.append(
            f"Code: {row['code']}\n"
            f"Label: {row['label']}\n"
            f"Description: {description}\n"
            f"Example jobs: {examples_text}"
        )
    return "\n\n".join(blocks)


def build_classification_messages(
    title: str,
    candidate_block: str,
) -> list[dict]:
    content = f"""You are an expert assigning CIUO-08.CL labels (the Chilean adaptation of ISCO-08) to job titles.
The job title may be in Spanish or Portuguese. Select the single best 4-digit CIUO-08.CL code.

JOB TITLE:
{title}

CANDIDATES:
{candidate_block}

Rules:
- Use the candidate descriptions and example jobs to choose the best match.
- The job title may be short, generic or contain spelling errors.
- Pick one candidate even if you are not fully sure.
- Return ONLY the 4-digit code. Do not write explanations, labels or punctuation.
"""
    return [
        {
            "role": "system",
            "content": (
                "You classify job titles into CIUO-08.CL occupation codes. "
                "Think briefly: at most 2-3 short sentences, then return only "
                "the 4-digit code."
            ),
        },
        {"role": "user", "content": content},
    ]


def parse_code(text: object, valid_codes: Iterable[str] | None = None) -> str | None:
    if text is None:
        return None
    raw = str(text).strip()
    if not raw:
        return None
    valid = {str(code).zfill(4) for code in (valid_codes or [])}
    matches = re.findall(r"(?<!\d)(\d{4})(?!\d)", raw)
    if matches:
        if valid:
            for match in matches:
                if match in valid:
                    return match
        return matches[0]
    fallback = re.search(r"\d{4}", raw)
    if fallback:
        return fallback.group(0)
    return None


def correct_code(
    code: object,
    valid_codes: Iterable[str],
) -> tuple[str | None, bool]:
    """Valida un codigo y, si no existe, lo repara por prefijo jerarquico."""
    digits = re.sub(r"\D", "", str(code or ""))
    if not digits:
        return None, False
    if len(digits) > 4:
        digits = digits[:4]
    valid = sorted({str(item).zfill(4) for item in valid_codes})
    valid_set = set(valid)
    if digits in valid_set:
        return digits, False
    for length in (3, 2, 1):
        if len(digits) < length:
            continue
        prefix = digits[:length]
        matches = [item for item in valid if item.startswith(prefix)]
        if matches:
            return min(matches), True
    return None, True


def top_candidates_payload(
    candidates: pd.DataFrame,
    indices: Sequence[int],
    distances: Sequence[float],
    description_chars: int = 300,
) -> list[dict]:
    payload: list[dict] = []
    for raw_index, distance in zip(indices, distances):
        row = candidates.iloc[int(raw_index)]
        description = str(row.get("description", "")).strip()
        if len(description) > description_chars:
            description = description[:description_chars].rstrip()
        examples = row.get("examples", [])
        if isinstance(examples, str):
            try:
                examples = json.loads(examples)
            except json.JSONDecodeError:
                examples = []
        numeric_distance = float(distance)
        payload.append(
            {
                "code": str(row["code"]).zfill(4),
                "label": str(row["label"]),
                "distance": round(numeric_distance, 6),
                "similarity": round(1.0 - numeric_distance, 6),
                "description": description,
                "examples": examples[:10] if isinstance(examples, list) else [],
            }
        )
    return payload
