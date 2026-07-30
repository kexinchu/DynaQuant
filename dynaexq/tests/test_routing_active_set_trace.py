from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from scripts import collect_routing_active_set_trace as trace_module
from scripts.collect_routing_active_set_trace import (
    MAX_INPUT_TOKENS,
    TRIALS,
    build_token_windows,
    load_prompt_rows,
    measure_expert_bytes,
    measure_expert_storage,
)


class _Tokenizer:
    eos_token_id = 2

    def __call__(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return {"input_ids": [int(text)] * MAX_INPUT_TOKENS}


class _ExpertLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.experts = torch.nn.ModuleList(
            [
                torch.nn.Linear(2, 2, bias=False),
                torch.nn.Linear(2, 2, bias=False),
            ]
        )
        self.shared_experts = torch.nn.Linear(2, 2, bias=False)


class _TinyExpertModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [_ExpertLayer(), _ExpertLayer()]
        )


def test_prompt_stream_builds_exact_disjoint_nested_source_windows(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(trace_module, "ROOT", tmp_path)
    prompt_path = tmp_path / "prompts.jsonl"
    rows = [
        {"id": f"id-{index}", "prompt": str(index + 1)}
        for index in range(TRIALS)
    ]
    prompt_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    loaded, source = load_prompt_rows(prompt_path)
    windows, selection = build_token_windows(_Tokenizer(), loaded)

    assert source["path"] == "prompts.jsonl"
    assert len(windows) == TRIALS
    assert all(len(window) == MAX_INPUT_TOKENS for window in windows)
    assert selection["selected_token_count"] == TRIALS * MAX_INPUT_TOKENS
    assert len(selection["selected_ids_sha256"]) == 64
    assert windows[0][:3] == [1, 1, 1]
    assert windows[-1][-3:] == [12, 12, 12]

    rows[-1]["id"] = rows[0]["id"]
    prompt_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unique"):
        load_prompt_rows(prompt_path)


def test_expert_byte_measurement_excludes_shared_experts():
    # Each routed expert owns a 2x2 float32 matrix: 16 bytes.
    model = _TinyExpertModel()
    assert measure_expert_bytes(
        model,
        layer_ids=[0, 1],
        experts_per_layer=2,
    ) == [16, 16]
    per_expert, storage = measure_expert_storage(
        model,
        layer_ids=[0, 1],
        experts_per_layer=2,
    )
    assert per_expert == [16, 16]
    assert len(storage["0"]) == 2
    assert all("shared_experts" not in item["name"] for item in storage["0"])
    assert sum(item["size_bytes"] for item in storage["0"]) == 32


def test_routing_trace_script_is_directly_executable():
    root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [
            sys.executable,
            str(
                root
                / "scripts"
                / "collect_routing_active_set_trace.py"
            ),
            "--help",
        ],
        cwd="/tmp",
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
