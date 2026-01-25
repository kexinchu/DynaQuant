import torch

from dynaexq.runtime.memmgr import MemoryManager, PoolConfig
from dynaexq.runtime.swap_engine import SwapConfig, SwapEngine
from dynaexq.runtime.types import Bitwidth, ExpertID, ResidencyLocation
from dynaexq.runtime.weights import DualPrecisionWeights, InMemoryWeightStore


def test_swap_engine_upgrade_and_downgrade():
    pool = PoolConfig(hot_capacity_bytes=4096,
                      cold_capacity_bytes=4096, transient_capacity_bytes=0)
    memory = MemoryManager(pool)
    w4_state = {"layers.0.experts.0.weight": torch.ones(4)}
    w2_state = {"layers.0.experts.0.weight": torch.ones(4) * 2}
    repo = DualPrecisionWeights.from_state_dicts(w4_state, w2_state)
    store = InMemoryWeightStore(repo, Bitwidth.W4)
    engine = SwapEngine(memory, store, SwapConfig(max_workers=1))

    expert = ExpertID(layer=0, idx=0)
    engine.upgrade(expert)
    residency = engine.wait_ready(expert)
    assert residency.location is ResidencyLocation.HBM
    assert residency.tensor_bundle is not None
    assert memory.residency(expert).location is ResidencyLocation.HBM

    engine.downgrade(expert)
    residency = engine.wait_ready(expert)
    assert residency.location is ResidencyLocation.DRAM
    assert memory.residency(expert).location is ResidencyLocation.DRAM

    engine.close()
