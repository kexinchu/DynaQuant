from __future__ import annotations

import json

import torch
from safetensors.torch import load_file, save_file

from scripts.derive_autoround_int2_from_int4 import (
    _pack_axis0,
    _pack_axis1,
    _unpack_axis0,
    _unpack_axis1,
    derive_checkpoint,
    requantize_autogptq_tensor,
)


def _make_w4_parent(weight: torch.Tensor, group_size: int = 128):
    out_features, in_features = weight.shape
    grouped = weight.t().reshape(
        in_features // group_size,
        group_size,
        out_features,
    )
    scales = grouped.abs().amax(dim=1).clamp(min=1e-10) / 7
    signed = torch.round(grouped / scales.unsqueeze(1)).clamp(-8, 7)
    unsigned = (signed.to(torch.int64) + 8).reshape(
        in_features,
        out_features,
    )
    qweight = _pack_axis0(unsigned, 4)
    stored_zeros = torch.full(
        (in_features // group_size, out_features),
        7,
        dtype=torch.int64,
    )
    qzeros = _pack_axis1(stored_zeros, 4)
    return qweight, qzeros, scales.to(torch.float16)


def test_autogptq_axis_pack_roundtrip():
    generator = torch.Generator().manual_seed(7)
    values = torch.randint(0, 4, (128, 32), generator=generator)
    packed = _pack_axis0(values, 2)
    assert torch.equal(_unpack_axis0(packed, 2), values)

    zeros = torch.randint(0, 4, (2, 32), generator=generator)
    packed_zeros = _pack_axis1(zeros, 2)
    assert torch.equal(
        _unpack_axis1(packed_zeros, 2, out_features=32),
        zeros,
    )


def test_w4_to_w2_requantization_shapes_values_and_zero_convention():
    generator = torch.Generator().manual_seed(11)
    weight = torch.randn(32, 256, generator=generator)
    qweight4, qzeros4, scales4 = _make_w4_parent(weight)

    qweight2, qzeros2, scales2 = requantize_autogptq_tensor(
        qweight4,
        qzeros4,
        scales4,
    )

    assert qweight2.shape == (16, 32)
    assert qzeros2.shape == (4, 2)
    assert scales2.shape == (4, 32)
    assert scales2.dtype == torch.float16
    unsigned2 = _unpack_axis0(qweight2, 2)
    assert set(unsigned2.unique().tolist()) <= {0, 1, 2, 3}
    stored_zp2 = _unpack_axis1(qzeros2, 2, out_features=32)
    assert torch.equal(stored_zp2, torch.ones_like(stored_zp2))


def test_w4_to_w2_is_deterministic():
    weight = torch.linspace(-3, 3, steps=32 * 256).reshape(32, 256)
    parent = _make_w4_parent(weight)
    first = requantize_autogptq_tensor(*parent)
    second = requantize_autogptq_tensor(*parent)
    for left, right in zip(first, second, strict=True):
        assert torch.equal(left, right)


def test_integer_domain_requantization_matches_explicit_w4_reconstruction():
    generator = torch.Generator().manual_seed(23)
    weight = torch.randn(32, 256, generator=generator)
    qweight4, qzeros4, scales4 = _make_w4_parent(weight)
    qweight2, qzeros2, scales2 = requantize_autogptq_tensor(
        qweight4,
        qzeros4,
        scales4,
    )

    unsigned4 = _unpack_axis0(qweight4, 4).reshape(2, 128, 32)
    zp4 = _unpack_axis1(qzeros4, 4, out_features=32) + 1
    reconstructed = (
        (unsigned4 - zp4.unsqueeze(1)) * scales4.float().unsqueeze(1)
    ).reshape(4, 64, 32)
    reference_scales = reconstructed.abs().amax(dim=1).clamp(min=1e-10)
    reference_signed = torch.round(
        reconstructed / reference_scales.unsqueeze(1)
    ).clamp(-2, 1)
    actual_signed = (
        _unpack_axis0(qweight2, 2).reshape(4, 64, 32) - 2
    )

    assert torch.equal(actual_signed, reference_signed)
    assert torch.equal(
        _unpack_axis1(qzeros2, 2, out_features=32),
        torch.ones(4, 32, dtype=torch.int64),
    )
    assert torch.equal(scales2, reference_scales.to(torch.float16))


def test_integer_domain_requantization_handles_negative_parent_scales():
    weight = torch.linspace(-2, 2, steps=32 * 256).reshape(32, 256)
    qweight4, qzeros4, scales4 = _make_w4_parent(weight)
    scales4[:, ::2] *= -1
    qweight2, _, scales2 = requantize_autogptq_tensor(
        qweight4,
        qzeros4,
        scales4,
    )

    unsigned4 = _unpack_axis0(qweight4, 4).reshape(2, 128, 32)
    zp4 = _unpack_axis1(qzeros4, 4, out_features=32) + 1
    reconstructed = (
        (unsigned4 - zp4.unsqueeze(1)) * scales4.float().unsqueeze(1)
    ).reshape(4, 64, 32)
    reference_scales = reconstructed.abs().amax(dim=1).clamp(min=1e-10)
    reference_signed = torch.round(
        reconstructed / reference_scales.unsqueeze(1)
    ).clamp(-2, 1)
    actual_signed = (
        _unpack_axis0(qweight2, 2).reshape(4, 64, 32) - 2
    )

    assert torch.equal(actual_signed, reference_signed)
    assert torch.equal(scales2, reference_scales.to(torch.float16))


def test_w4_to_w2_rejects_non_autogptq_integer_layout():
    weight = torch.ones(32, 256)
    qweight, qzeros, scales = _make_w4_parent(weight)
    try:
        requantize_autogptq_tensor(qweight.to(torch.int64), qzeros, scales)
    except ValueError as error:
        assert "must be int32" in str(error)
    else:
        raise AssertionError("expected invalid qweight dtype to fail")


def test_checkpoint_derivation_records_parent_and_preserves_dense_tensors(
    tmp_path,
):
    parent = tmp_path / "parent"
    parent.mkdir()
    output = tmp_path / "child"
    base = "model.layers.0.mlp.experts.0.gate_proj"
    weight = torch.linspace(-2, 2, steps=32 * 256).reshape(32, 256)
    qweight, qzeros, scales = _make_w4_parent(weight)
    tensors = {
        base + ".qweight": qweight,
        base + ".qzeros": qzeros,
        base + ".scales": scales,
        "model.embed_tokens.weight": torch.ones(32, 32),
        "model.layers.0.input_layernorm.weight": torch.ones(32),
        "model.norm.weight": torch.ones(32),
        "lm_head.weight": torch.ones(32, 32),
    }
    shard_name = "model-00001-of-00001.safetensors"
    save_file(tensors, parent / shard_name)
    total_size = sum(
        tensor.numel() * tensor.element_size() for tensor in tensors.values()
    )
    (parent / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": total_size},
                "weight_map": {name: shard_name for name in tensors},
            }
        ),
        encoding="utf-8",
    )
    (parent / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_next",
                "architectures": ["Qwen3NextForCausalLM"],
                "num_hidden_layers": 1,
                "tie_word_embeddings": False,
                "quantization_config": {
                    "quant_method": "auto-round",
                    "bits": 4,
                    "group_size": 128,
                    "sym": True,
                    "packing_format": "auto_round:auto_gptq",
                    "extra_config": {},
                },
            }
        ),
        encoding="utf-8",
    )
    (parent / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    manifest_path = tmp_path / "parent_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "artifact_type": "model_snapshot_manifest",
                "local_path": str(parent),
                "source": {
                    "provider": "modelscope",
                    "repository": "Intel/test",
                    "revision": "a" * 64,
                },
            }
        ),
        encoding="utf-8",
    )

    provenance = derive_checkpoint(
        parent,
        output,
        parent_manifest_path=manifest_path,
    )

    child_config = json.loads((output / "config.json").read_text())
    assert child_config["quantization_config"]["bits"] == 2
    assert child_config["quantization_config"]["group_size"] == 64
    assert provenance["output"]["converted_w4_modules"] == 1
    assert provenance["calibration"]["sample_count"] == 0
    assert (output / "tokenizer_config.json").is_file()
    child_tensors = load_file(str(output / shard_name))
    assert child_tensors[base + ".qweight"].shape == (16, 32)
    assert torch.equal(
        child_tensors["model.embed_tokens.weight"],
        tensors["model.embed_tokens.weight"],
    )
