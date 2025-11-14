from dynaexq.runtime.memmgr import MemoryManager, PoolConfig
from dynaexq.runtime.types import Bitwidth, ExpertID, Residency, ResidencyLocation


def make_residency(location: ResidencyLocation, bytes_: int) -> Residency:
    return Residency(bitwidth=Bitwidth.W4, location=location, bytes=bytes_)


def test_reserve_hot_evicts_lru():
    pool = PoolConfig(hot_capacity_bytes=10, cold_capacity_bytes=20, transient_capacity_bytes=0)
    manager = MemoryManager(pool)

    e0 = ExpertID(layer=0, idx=0)
    e1 = ExpertID(layer=0, idx=1)

    assert manager.reserve_hot(e0, 6)
    manager.place(e0, make_residency(ResidencyLocation.HBM, 6))

    assert manager.reserve_hot(e1, 6)
    evicted = manager.evict_hot()
    assert evicted == [e0]

    manager.place(e1, make_residency(ResidencyLocation.HBM, 6))
    assert manager.hot_occupancy() > 0.0


def test_place_updates_usage():
    pool = PoolConfig(hot_capacity_bytes=100, cold_capacity_bytes=100, transient_capacity_bytes=0)
    manager = MemoryManager(pool)
    expert = ExpertID(layer=1, idx=2)

    manager.place(expert, make_residency(ResidencyLocation.DRAM, 20))
    assert manager.cold_occupancy() == 0.2

    manager.place(expert, make_residency(ResidencyLocation.HBM, 10))
    assert manager.hot_occupancy() == 0.1

