from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "export_qwen30b_replay_evaluation.py"
SPEC = importlib.util.spec_from_file_location("export_qwen30b_replay_evaluation", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _source(tmp_path: Path, *, replay: bool = True) -> Path:
    path = tmp_path / "summary.json"
    path.write_text(
        json.dumps(
            {
                "artifact_type": "a6000_wikitext_deadline_replay",
                "not_end_to_end": replay,
                "lengths": [
                    {
                        "input_tokens": 2048,
                        "coupling_grid": [
                            {
                                "split": "test",
                                "phi": 0.85,
                                "fid_bytes": 10,
                                "res_bytes": 70,
                                "look_bytes": 5,
                                "budget_bytes": 85,
                                "quality_delta": 0.2,
                                "perplexity": 7.4,
                                "exposed_wait_mean_ms": 1.5,
                                "exposed_wait_p95_ms": 2.0,
                                "n_high": 1,
                                "n_low": 2,
                                "staging_slots": 1,
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_export_preserves_replay_label_and_byte_identity(tmp_path: Path) -> None:
    source = _source(tmp_path)
    output = tmp_path / "coupling.jsonl"
    assert MODULE.export(source, output) == 1
    row = json.loads(output.read_text(encoding="utf-8"))
    assert row["evidence_class"] == "trace_replay"
    assert row["not_end_to_end"] is True
    assert row["fid_bytes"] + row["res_bytes"] + row["look_bytes"] == row["budget_bytes"]


def test_export_rejects_end_to_end_relabeling(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not_end_to_end"):
        MODULE.export(_source(tmp_path, replay=False), tmp_path / "out.jsonl")
