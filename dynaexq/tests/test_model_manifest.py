from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import save_file

from scripts.build_model_manifest import build_manifest


def _snapshot(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps(
            {
                "model_type": "test_moe",
                "architectures": ["TestMoeForCausalLM"],
                "num_hidden_layers": 1,
                "tie_word_embeddings": False,
            }
        ),
        encoding="utf-8",
    )
    tensors = {
        "model.embed_tokens.weight": torch.ones(1),
        "model.layers.0.weight": torch.ones(1),
        "model.norm.weight": torch.ones(1),
        "lm_head.weight": torch.ones(1),
    }
    shard = model / "model-00001-of-00001.safetensors"
    save_file(tensors, shard)
    (model / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 16},
                "weight_map": {
                    "model.embed_tokens.weight": shard.name,
                    "model.layers.0.weight": (
                        "model-00001-of-00001.safetensors"
                    ),
                    "model.norm.weight": shard.name,
                    "lm_head.weight": shard.name,
                },
            }
        ),
        encoding="utf-8",
    )
    return model


def test_build_model_manifest_hashes_complete_snapshot(tmp_path):
    model = _snapshot(tmp_path)
    manifest = build_manifest(
        model,
        provider="huggingface",
        repository="org/model",
        revision="a" * 40,
    )
    assert manifest["source"]["revision"] == "a" * 40
    assert manifest["model_type"] == "test_moe"
    assert manifest["tensor_count"] == 4
    assert manifest["verified_safetensors_tensor_count"] == 4
    assert manifest["verified_tensor_bytes"] == 16
    assert manifest["verified_hidden_layer_count"] == 1
    assert manifest["weight_shard_count"] == 1
    assert manifest["file_count"] == 3
    assert all(len(item["sha256"]) == 64 for item in manifest["files"])


def test_build_model_manifest_rejects_moving_revision_and_missing_shard(
    tmp_path,
):
    model = _snapshot(tmp_path)
    with pytest.raises(ValueError, match="hexadecimal"):
        build_manifest(
            model,
            provider="modelscope",
            repository="org/model",
            revision="master",
        )
    (model / "model-00001-of-00001.safetensors").unlink()
    with pytest.raises(ValueError, match="missing shards"):
        build_manifest(
            model,
            provider="huggingface",
            repository="org/model",
            revision="b" * 40,
        )


def test_build_model_manifest_rejects_incomplete_declared_layers(tmp_path):
    model = _snapshot(tmp_path)
    config_path = model / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["num_hidden_layers"] = 2
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="every declared hidden layer"):
        build_manifest(
            model,
            provider="huggingface",
            repository="org/model",
            revision="c" * 40,
        )


def test_build_model_manifest_rejects_index_header_mismatch(tmp_path):
    model = _snapshot(tmp_path)
    index_path = model / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["weight_map"]["model.layers.0.not_in_shard"] = (
        "model-00001-of-00001.safetensors"
    )
    index_path.write_text(json.dumps(index), encoding="utf-8")
    with pytest.raises(ValueError, match="index/header mismatch"):
        build_manifest(
            model,
            provider="huggingface",
            repository="org/model",
            revision="d" * 40,
        )
