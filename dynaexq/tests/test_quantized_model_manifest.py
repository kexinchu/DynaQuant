from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import save_file

from scripts.build_quantized_model_manifest import build_quantized_manifest


def test_quantized_manifest_hashes_checkpoint_and_provenance(tmp_path):
    model = tmp_path / "quantized"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps(
            {
                "model_type": "test_moe",
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
    (model / "quantization_provenance.json").write_text(
        json.dumps(
            {
                "artifact_type": "local_quantization_provenance",
                "output": {"path": str(model)},
                "source_manifest": {
                    "path": "/source.json",
                    "sha256": "a" * 64,
                },
            }
        ),
        encoding="utf-8",
    )

    manifest = build_quantized_manifest(model)

    assert manifest["artifact_type"] == "quantized_model_manifest"
    assert manifest["relocated_copy"] is False
    assert manifest["source_manifest"]["sha256"] == "a" * 64
    assert manifest["verified_hidden_layer_count"] == 1
    assert manifest["verified_safetensors_tensor_count"] == 4
    assert manifest["file_count"] == 4
    assert len(manifest["quantization_provenance"]["sha256"]) == 64


def test_quantized_manifest_requires_explicit_relocation(tmp_path):
    model = tmp_path / "quantized"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps(
            {
                "model_type": "test_moe",
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
    (model / "quantization_provenance.json").write_text(
        json.dumps(
            {
                "artifact_type": "local_quantization_provenance",
                "output": {"path": str(tmp_path / "original")},
                "source_manifest": {},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="output path does not match"):
        build_quantized_manifest(model)
    manifest = build_quantized_manifest(model, allow_relocated=True)
    assert manifest["relocated_copy"] is True
    assert manifest["export_path"] == str(tmp_path / "original")
