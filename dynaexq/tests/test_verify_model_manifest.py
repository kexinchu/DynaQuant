from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import save_file

from scripts.build_model_manifest import build_manifest
from scripts.verify_model_manifest import verify_manifest


def _snapshot(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    config = {
        "model_type": "test_moe",
        "architectures": ["TestForCausalLM"],
        "num_hidden_layers": 1,
        "tie_word_embeddings": False,
    }
    (model / "config.json").write_text(json.dumps(config), encoding="utf-8")
    tensors = {
        "model.embed_tokens.weight": torch.ones(1),
        "model.layers.0.weight": torch.ones(1),
        "model.norm.weight": torch.ones(1),
        "lm_head.weight": torch.ones(1),
    }
    shard = model / "model.safetensors"
    save_file(tensors, shard)
    (model / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 16},
                "weight_map": {name: shard.name for name in tensors},
            }
        ),
        encoding="utf-8",
    )
    return model


def test_verify_model_manifest_hashes_every_file(tmp_path):
    model = _snapshot(tmp_path)
    manifest = build_manifest(
        model,
        provider="huggingface",
        repository="org/test",
        revision="a" * 40,
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = verify_manifest(manifest_path)

    assert result["file_count"] == 3
    assert result["relocated"] is False
    assert result["verified_safetensors_tensor_count"] == 4
    assert len(result["manifest_sha256"]) == 64


def test_verify_model_manifest_rejects_content_change(tmp_path):
    model = _snapshot(tmp_path)
    manifest = build_manifest(
        model,
        provider="huggingface",
        repository="org/test",
        revision="b" * 40,
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    (model / "config.json").write_text(
        (model / "config.json").read_text() + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="size mismatch"):
        verify_manifest(manifest_path)


def test_verify_model_manifest_accepts_explicit_relocation(tmp_path):
    model = _snapshot(tmp_path)
    manifest = build_manifest(
        model,
        provider="huggingface",
        repository="org/test",
        revision="c" * 40,
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    relocated = tmp_path / "copy"
    relocated.mkdir()
    for source in model.iterdir():
        (relocated / source.name).write_bytes(source.read_bytes())

    result = verify_manifest(manifest_path, model_dir=relocated)

    assert result["relocated"] is True
    assert result["verified_local_path"] == str(relocated)
