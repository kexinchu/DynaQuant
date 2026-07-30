from __future__ import annotations

import pytest

from scripts.build_independent_calibration import build_records


def test_calibration_builder_uses_stable_training_row_ids():
    dataset = [
        {"text": "a" * 80},
        {"text": ""},
        {"text": "b" * 80},
        {"text": "c" * 80},
        {"text": "d" * 80},
    ] * 80
    records, consumed = build_records(
        dataset,
        count=128,
        min_chars=70,
        max_chars=100,
    )
    assert len(records) == 128
    assert consumed > 128
    assert records[0]["split"] == "train"
    assert records[0]["id"] == "rows-0-0"
    assert len({record["id"] for record in records}) == 128


def test_calibration_builder_requires_formal_sample_count():
    with pytest.raises(ValueError, match="at least 128"):
        build_records(
            [{"text": "content"}],
            count=127,
            min_chars=1,
            max_chars=10,
        )
