from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from dynaexq.core.config import (
    DynaExqConfig,
    MemoryConfig,
    ModelConfig,
    PrecisionConfig,
    SchedulerConfig,
    Tier,
)
from dynaexq.experiments.run_shift import (
    _validate_model_contract,
    initialize_dynaexq,
)


class _Expert(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.randn(8, 128, dtype=torch.float16)
        )


class _Layer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.experts = torch.nn.ModuleList([_Expert(), _Expert()])


class _Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([_Layer()])


def test_initialize_materializes_every_expert_and_detaches_native_sources():
    model = _Model()
    config = DynaExqConfig(
        model=ModelConfig(
            name="tiny",
            layers=1,
            experts_per_layer=2,
            topk=1,
        ),
        precision=PrecisionConfig(hi="fp16", lo="int4"),
        scheduler=SchedulerConfig(update_period_steps=2),
        memory=MemoryConfig(
            device_mem_bytes=100_000,
            max_inflight=2,
        ),
    )
    (
        _,
        _,
        _,
        registry,
        engine,
        _,
        metadata,
    ) = initialize_dynaexq(config, model, torch.device("cpu"))
    try:
        assert len(registry.tier_snapshot()) == 2
        assert metadata["host_cache"]["entries"] == 4
        assert metadata["released_native_expert_bytes"] == 2 * 8 * 128 * 2
        assert metadata["bootstrap"]["failed_transitions"] == 0
        assert all(expert.weight.numel() == 0 for expert in model.layers[0].experts)
        assert engine.get_stats()["enqueue_attempts"] == 0
    finally:
        engine.shutdown()


def test_checkpoint_architecture_must_match_runtime_yaml():
    model = _Model()
    model.config = SimpleNamespace(
        num_hidden_layers=2,
        num_experts=4,
        num_experts_per_tok=2,
    )
    config = DynaExqConfig(
        model=ModelConfig(
            name="wrong",
            layers=1,
            experts_per_layer=2,
            topk=1,
        ),
        precision=PrecisionConfig(hi="fp16", lo="int4"),
        memory=MemoryConfig(device_mem_bytes=100_000),
    )
    with pytest.raises(ValueError, match="runtime model contract mismatch") as error:
        _validate_model_contract(config, model)
    assert "layers: yaml=1, checkpoint=2" in str(error.value)
    assert "experts_per_layer: yaml=2, checkpoint=4" in str(error.value)
    assert "topk: yaml=1, checkpoint=2" in str(error.value)


def test_initialize_uses_calibrated_ranking_prefix_instead_of_expert_ids():
    model = _Model()
    config = DynaExqConfig(
        model=ModelConfig(
            name="tiny",
            layers=1,
            experts_per_layer=2,
            topk=1,
        ),
        precision=PrecisionConfig(hi="fp16", lo="int4"),
        scheduler=SchedulerConfig(update_period_steps=2),
        memory=MemoryConfig(
            device_mem_bytes=5_000,
            max_inflight=1,
        ),
    )
    (
        _,
        _,
        _,
        registry,
        engine,
        _,
        metadata,
    ) = initialize_dynaexq(
        config,
        model,
        torch.device("cpu"),
        initial_expert_ranking={0: [1, 0]},
    )
    try:
        assert metadata["n_hi"] == [1]
        assert metadata["bootstrap_policy"] == "calibrated_ranking_prefix"
        assert metadata["bootstrap_hi_experts"] == {"0": [1]}
        tiers = {
            key.expert: tier for key, tier in registry.tier_snapshot().items()
        }
        # The ranking, not numeric expert order, determines the HI slot.
        assert tiers == {0: Tier.LO, 1: Tier.HI}
    finally:
        engine.shutdown()
