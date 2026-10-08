import json
import csv

import pytest

from scripts import summarize_qwen30b_p1b_switch as summary


def _artifact(policy: str, p95_tpot: float) -> dict:
    return {
        "benchmark": {
            "metrics": {
                "model_tpot_ms": {"p95": p95_tpot, "mean": p95_tpot - 10.0},
                "model_ttft_ms": {"p95": 2000.0, "mean": 1500.0},
            }
        },
        "transition_stats": {
            "accepted_bytes": 1_000_000_000,
            "copied_bytes": 2_000_000_000,
            "failed_transitions": 0,
            "active_transitions": 0,
            "arena": {
                "fid_bytes": 3_000_000_000,
                "res_bytes": 4_000_000_000,
                "look_bytes": 5_000_000_000,
            },
        },
        "wrapper_stats": {"scheduler_p99_ms": 3.5},
        "allocation_protocol": {"allocation_policy": policy},
    }


def test_p1b_switch_summary_scores_strongest_baseline(tmp_path, monkeypatch, capsys):
    scores = {
        "dynabyte": 290.0,
        "static_joint": 330.0,
        "seq_fr": 310.0,
        "seq_rf": 305.0,
    }
    for label, name in summary.ARTIFACTS.items():
        (tmp_path / name).write_text(
            json.dumps(_artifact(label, scores[label])),
            encoding="utf-8",
        )

    monkeypatch.setattr(summary, "RESULTS", tmp_path)
    monkeypatch.setattr(summary, "active_outputs", lambda: {})
    monkeypatch.setattr(summary, "validate", lambda path: (True, None))

    summary.main([])

    output = json.loads(capsys.readouterr().out)
    assert output["decision"]["ready_to_rescore"] is True
    assert output["decision"]["winner_by_p95_tpot"] == "dynabyte"
    assert output["decision"]["strongest_deployable_baseline"] == "seq_rf"
    assert output["decision"]["dynabyte_beats_strongest_baseline"] is True
    assert output["decision"]["dynabyte_minus_strongest_baseline_ms"] == -15.0


def test_p1b_switch_summary_writes_stable_csv(tmp_path, monkeypatch, capsys):
    scores = {
        "dynabyte": 290.0,
        "static_joint": 330.0,
        "seq_fr": 310.0,
        "seq_rf": 305.0,
    }
    for label, name in summary.ARTIFACTS.items():
        (tmp_path / name).write_text(
            json.dumps(_artifact(label, scores[label])),
            encoding="utf-8",
        )

    monkeypatch.setattr(summary, "RESULTS", tmp_path)
    monkeypatch.setattr(summary, "active_outputs", lambda: {})
    monkeypatch.setattr(summary, "validate", lambda path: (True, None))

    output_csv = tmp_path / "summary.csv"
    summary.main(["--output-csv", str(output_csv)])
    capsys.readouterr()

    rows = list(csv.DictReader(output_csv.open(encoding="utf-8")))
    assert [row["policy"] for row in rows] == [
        "dynabyte",
        "static_joint",
        "seq_fr",
        "seq_rf",
    ]
    assert rows[0]["artifact"] == "rq2_switch_dynabyte_seed42.json"
    assert rows[0]["status"] == "valid"
    assert rows[0]["p95_tpot_ms"] == "290.0"
    assert rows[0]["accepted_gb"] == "1.0"
    assert rows[0]["copied_gb"] == "2.0"


def test_p1b_switch_summary_writes_json_payload(tmp_path, monkeypatch, capsys):
    for label, name in summary.ARTIFACTS.items():
        (tmp_path / name).write_text(
            json.dumps(_artifact(label, 300.0)),
            encoding="utf-8",
        )

    monkeypatch.setattr(summary, "RESULTS", tmp_path)
    monkeypatch.setattr(summary, "active_outputs", lambda: {})
    monkeypatch.setattr(summary, "validate", lambda path: (True, None))

    output_json = tmp_path / "summary.json"
    summary.main(["--output-json", str(output_json), "--require-ready"])
    printed = json.loads(capsys.readouterr().out)
    written = json.loads(output_json.read_text(encoding="utf-8"))

    assert written == printed
    assert written["decision"]["ready_to_rescore"] is True
    assert set(written["artifacts"]) == {
        "dynabyte",
        "static_joint",
        "seq_fr",
        "seq_rf",
    }


def test_p1b_switch_summary_writes_latex_table(tmp_path, monkeypatch, capsys):
    for label, name in summary.ARTIFACTS.items():
        (tmp_path / name).write_text(
            json.dumps(_artifact(label, 300.0)),
            encoding="utf-8",
        )

    monkeypatch.setattr(summary, "RESULTS", tmp_path)
    monkeypatch.setattr(summary, "active_outputs", lambda: {})
    monkeypatch.setattr(summary, "validate", lambda path: (True, None))

    output_latex = tmp_path / "summary.tex"
    summary.main(["--output-latex", str(output_latex), "--require-ready"])
    capsys.readouterr()

    table = output_latex.read_text(encoding="utf-8")
    assert r"\label{tab:eval-rq2-p1b-switch}" in table
    assert r"\systemname" in table
    assert "Static-Joint" in table
    assert "Seq-FR" in table
    assert "Seq-RF" in table
    assert "300.0" in table


def test_p1b_switch_summary_require_ready_rejects_running(tmp_path, monkeypatch, capsys):
    for label, name in summary.ARTIFACTS.items():
        (tmp_path / name).write_text(
            json.dumps(_artifact(label, 300.0)),
            encoding="utf-8",
        )

    monkeypatch.setattr(summary, "RESULTS", tmp_path)
    monkeypatch.setattr(
        summary,
        "active_outputs",
        lambda: {"rq2_switch_seq_fr_seed42.json": [1234]},
    )
    monkeypatch.setattr(
        summary,
        "validate",
        lambda path: (
            False,
            "missing",
        )
        if path.name == "rq2_switch_seq_fr_seed42.json"
        else (True, None),
    )

    with pytest.raises(SystemExit, match="not all valid"):
        summary.main(["--require-ready"])
    output = json.loads(capsys.readouterr().out)
    assert output["artifacts"]["seq_fr"]["status"] == "running"
    assert output["decision"]["ready_to_rescore"] is False
