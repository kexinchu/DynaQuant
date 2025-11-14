import torch

from dynaexq.runtime.controller import PrecisionController
from dynaexq.runtime.monitor import ExpertMonitor
from dynaexq.runtime.prefetch import PrefetchPlanner
from dynaexq.runtime.swap_engine import SwapConfig, SwapEngine
from dynaexq.runtime.types import Bitwidth, ExpertID
from dynaexq.runtime.weights import DualPrecisionWeights, InMemoryWeightStore
from dynaexq.runtime.memmgr import MemoryManager, PoolConfig


def test_prefetch_triggers_upgrades():
    monitor = ExpertMonitor(ewma_alpha=1.0, epoch_decay=0.5)
    controller = PrecisionController(tau_h=0.5, tau_c=0.3, max_w4_slots=2)
    pool = PoolConfig(hot_capacity_bytes=4096,
                      cold_capacity_bytes=4096, transient_capacity_bytes=0)
    memory = MemoryManager(pool)
    w4_state = {"layers.0.experts.0.weight": torch.ones(4)}
    w2_state = {"layers.0.experts.0.weight": torch.ones(4) * 2}
    repo = DualPrecisionWeights.from_state_dicts(w4_state, w2_state)
    store = InMemoryWeightStore(repo, Bitwidth.W4)
    swap = SwapEngine(memory, store, SwapConfig(max_workers=1))
    planner = PrefetchPlanner(swap, controller, monitor)

    expert = ExpertID(layer=0, idx=0)
    monitor.update_batch(0, [[0]], [[0.9]])

    planner.lookahead(0, [expert])
    residency = swap.wait_ready(expert)
    assert residency.bitwidth is Bitwidth.W4
    swap.close()
