from __future__ import annotations

import pandas as pd

from isco08.generator import (
    build_classification_messages,
    correct_code,
    format_candidate_block,
    parse_code,
    top_candidates_payload,
)


def _candidates() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "code": ["4226", "5132"],
            "label": ["Recepcionistas", "Baristas"],
            "description": ["Atienden recepcion", "Preparan cafe"],
            "examples": [["Recepcionista de hotel"], ["Barista"]],
        }
    )


def test_candidate_block_has_no_body():
    block = format_candidate_block(_candidates(), [0, 1], max_examples=1)
    assert "Code: 4226" in block
    assert "Label: Recepcionistas" in block
    assert "Example jobs: Recepcionista de hotel" in block
    assert "body" not in block.lower()


def test_classification_prompt_is_title_only():
    block = format_candidate_block(_candidates(), [0, 1], max_examples=1)
    messages = build_classification_messages("Recepcionista", block)
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    content = messages[1]["content"]
    assert "JOB TITLE:\nRecepcionista" in content
    assert "CANDIDATES:" in content
    assert "body" not in content.lower()
    assert "Return ONLY the 4-digit code" in content


def test_parse_code_and_repair_by_prefix():
    valid = ["4226", "4227", "5132"]
    assert parse_code("4226") == "4226"
    assert parse_code("El codigo es 4226.") == "4226"
    assert parse_code("no hay codigo") is None
    assert correct_code("4228", valid) == ("4226", True)
    assert correct_code("4226", valid) == ("4226", False)


def test_top_candidates_payload_contains_distance_and_similarity():
    payload = top_candidates_payload(
        _candidates(),
        [0, 1],
        [0.25, 0.75],
        description_chars=100,
    )
    assert payload[0]["code"] == "4226"
    assert payload[0]["distance"] == 0.25
    assert payload[0]["similarity"] == 0.75
    assert payload[1]["distance"] == 0.75
