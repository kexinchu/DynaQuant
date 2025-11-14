from dynaexq.runtime.controller import PrecisionController
from dynaexq.runtime.monitor import ExpertMonitor
from dynaexq.runtime.types import Bitwidth, ExpertID


def test_controller_promotes_and_demotes_with_hysteresis():
    monitor = ExpertMonitor(ewma_alpha=1.0, epoch_decay=0.5)
    controller = PrecisionController(tau_h=0.6, tau_c=0.3, max_w4_slots=1)

    expert_hot = ExpertID(layer=0, idx=0)
    expert_cold = ExpertID(layer=0, idx=1)

    monitor.update_batch(0, [[0], [0]], [[0.9], [0.9]])
    targets = controller.plan([expert_hot, expert_cold], monitor)
    assert targets[expert_hot] is Bitwidth.W4
    assert targets[expert_cold] is Bitwidth.W2

    monitor.update_batch(0, [[1], [1]], [[0.95], [0.95]])
    targets = controller.plan([expert_cold], monitor)
    assert targets[expert_cold] is Bitwidth.W4
    assert controller.plan([expert_hot], monitor)[expert_hot] is Bitwidth.W2


def test_controller_enforces_slot_cap():
    monitor = ExpertMonitor(ewma_alpha=1.0, epoch_decay=0.5)
    controller = PrecisionController(tau_h=0.5, tau_c=0.3, max_w4_slots=2)

    experts = [ExpertID(layer=0, idx=i) for i in range(4)]
    for idx, expert in enumerate(experts):
        value = 0.9 - idx * 0.1
        monitor.update_batch(0, [[expert.idx]], [[value]])

    targets = controller.plan(experts, monitor)
    w4 = [e for e, b in targets.items() if b is Bitwidth.W4]
    assert len(w4) == 2

