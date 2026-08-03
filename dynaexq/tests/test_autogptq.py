from __future__ import annotations

import torch

from dynaexq.core.autogptq import (
    pack_axis0,
    pack_axis1,
    packed_from_autogptq,
    requantize_autogptq_tensor,
    unpack_axis0,
    unpack_axis1,
)
from dynaexq.core.config import Tier
from dynaexq.core.quant import (
    QuantFormat,
    _pack_int2,
    _pack_int4,
    dequant_to_fp16,
)
from dynaexq.core.registry import ExpertKey
from dynaexq.core.weight_store import ModelWeightStore


class _FakeQuantLinear(torch.nn.Module):
    def __init__(self, signed_codes: torch.Tensor, scales: torch.Tensor):
        super().__init__()
        in_features, out_features = signed_codes.shape
        self.bits = 4
        self.group_size = 128
        self.infeatures = in_features
        self.outfeatures = out_features
        self.register_buffer(
            "qweight",
            pack_axis0(signed_codes.to(torch.int64) + 8, 4),
        )
        self.register_buffer(
            "qzeros",
            pack_axis1(
                torch.full(
                    (in_features // self.group_size, out_features),
                    7,
                    dtype=torch.int64,
                ),
                4,
            ),
        )
        self.register_buffer("scales", scales.clone())
        self.register_buffer(
            "g_idx",
            torch.arange(in_features, dtype=torch.int32) // self.group_size,
        )
        self.bias = None


class _FakeAutoGPTQExpert(torch.nn.Module):
    def __init__(self, seed: int):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        for name in ("gate_proj", "up_proj", "down_proj"):
            signed = torch.randint(
                -8,
                8,
                (128, 16),
                generator=generator,
                dtype=torch.int64,
            )
            scales = torch.rand(
                (1, 16),
                generator=generator,
                dtype=torch.float16,
            )
            setattr(self, name, _FakeQuantLinear(signed, scales))


class _FakeAutoGPTQModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        layer = torch.nn.Module()
        layer.experts = torch.nn.ModuleList([_FakeAutoGPTQExpert(19)])
        self.layers = torch.nn.ModuleList([layer])


def test_autogptq_w4_layout_conversion_is_bit_exact():
    signed = torch.arange(128 * 16).reshape(128, 16) % 16 - 8
    scales = (
        torch.arange(16, dtype=torch.float16).reshape(1, 16) + 1
    ) / 32
    linear = _FakeQuantLinear(signed, scales)

    packed = packed_from_autogptq(
        linear.qweight,
        linear.qzeros,
        linear.scales,
        source_bits=linear.bits,
        source_group_size=linear.group_size,
        target_format=QuantFormat.INT4,
        g_idx=linear.g_idx,
    )
    assert torch.equal(packed.qweight, _pack_int4(signed.transpose(0, 1)))
    assert torch.equal(packed.scales, scales.transpose(0, 1))
    expected = (
        signed.transpose(0, 1).to(torch.float16)
        * scales.transpose(0, 1)
    )
    torch.testing.assert_close(
        dequant_to_fp16(packed),
        expected,
        rtol=0,
        atol=0,
    )


def test_autogptq_runtime_w2_matches_checkpoint_derivation():
    generator = torch.Generator().manual_seed(23)
    signed = torch.randint(
        -8,
        8,
        (128, 16),
        generator=generator,
        dtype=torch.int64,
    )
    scales = torch.rand(
        (1, 16),
        generator=generator,
        dtype=torch.float16,
    )
    linear = _FakeQuantLinear(signed, scales)

    derived_qweight, derived_qzeros, derived_scales = (
        requantize_autogptq_tensor(
            linear.qweight,
            linear.qzeros,
            linear.scales,
        )
    )
    derived_unsigned = unpack_axis0(derived_qweight, 2)
    derived_zero_points = unpack_axis1(
        derived_qzeros,
        2,
        out_features=16,
    ) + 1
    derived_signed = (
        derived_unsigned.reshape(2, 64, 16)
        - derived_zero_points.unsqueeze(1)
    ).reshape(128, 16)

    packed = packed_from_autogptq(
        linear.qweight,
        linear.qzeros,
        linear.scales,
        source_bits=4,
        source_group_size=128,
        target_format=QuantFormat.INT2,
        g_idx=linear.g_idx,
    )
    assert torch.equal(
        packed.qweight,
        _pack_int2(derived_signed.transpose(0, 1)),
    )
    assert torch.equal(packed.scales, derived_scales.transpose(0, 1))


def test_weight_store_loads_and_releases_autogptq_experts_exactly():
    model = _FakeAutoGPTQModel()
    expert = model.layers[0].experts[0]
    expected_released = sum(
        tensor.numel() * tensor.element_size()
        for projection in (
            expert.gate_proj,
            expert.up_proj,
            expert.down_proj,
        )
        for tensor in (
            projection.qweight,
            projection.qzeros,
            projection.scales,
            projection.g_idx,
        )
    )
    store = ModelWeightStore(
        model=model,
        hi_format="int4",
        lo_format="int2",
    )
    key = ExpertKey(0, 0)
    high = store.load_weights(key, Tier.HI)
    low = store.load_weights(key, Tier.LO)
    assert set(high) == {"gate_proj", "up_proj", "down_proj"}
    assert all(item.fmt == QuantFormat.INT4 for item in high.values())
    assert all(item.fmt == QuantFormat.INT2 for item in low.values())
    assert store.get_byte_size(key, Tier.HI) == sum(
        item.nbytes for item in high.values()
    )
    assert store.get_byte_size(key, Tier.LO) == sum(
        item.nbytes for item in low.values()
    )

    summary = store.preload_and_release_all(1, 1)
    assert summary["released_native_expert_bytes"] == expected_released
    for projection in (
        expert.gate_proj,
        expert.up_proj,
        expert.down_proj,
    ):
        assert projection.qweight.numel() == 0
        assert projection.qzeros.numel() == 0
        assert projection.scales.numel() == 0
        assert projection.g_idx.numel() == 0
