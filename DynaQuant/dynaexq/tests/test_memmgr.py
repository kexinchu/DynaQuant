"""
Unit tests for MemoryManager
"""

import unittest

from dynaexq.runtime.memmgr import MemoryManager, MemoryPool
from dynaexq.runtime.types import ExpertID, Residency


class TestMemoryPool(unittest.TestCase):
    """Test MemoryPool functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.pool = MemoryPool(
            name="TestPool",
            capacity_bytes=1000,
            slot_size_bytes=100
        )

    def test_initialization(self):
        """Test pool initialization"""
        self.assertEqual(self.pool.max_slots, 10)
        self.assertEqual(self.pool.free_slots, 10)

    def test_allocate(self):
        """Test slot allocation"""
        expert = ExpertID(layer=0, idx=0)
        residency = Residency(bitwidth="W4", location="HBM", bytes=100)

        result = self.pool.allocate(expert, residency)

        self.assertTrue(result)
        self.assertEqual(self.pool.free_slots, 9)
        self.assertTrue(self.pool.contains(expert))

    def test_allocate_full(self):
        """Test allocation when pool is full"""
        # Fill the pool
        for i in range(10):
            expert = ExpertID(layer=0, idx=i)
            residency = Residency(bitwidth="W4", location="HBM", bytes=100)
            self.pool.allocate(expert, residency)

        # Try to allocate one more
        expert = ExpertID(layer=0, idx=10)
        residency = Residency(bitwidth="W4", location="HBM", bytes=100)
        result = self.pool.allocate(expert, residency)

        self.assertFalse(result)

    def test_evict_lru(self):
        """Test LRU eviction"""
        # Allocate 3 experts
        for i in range(3):
            expert = ExpertID(layer=0, idx=i)
            residency = Residency(bitwidth="W4", location="HBM", bytes=100)
            self.pool.allocate(expert, residency)

        # Touch expert 1 (move to end)
        self.pool.touch(ExpertID(layer=0, idx=1))

        # Evict LRU (should be expert 0)
        evicted = self.pool.evict_lru()

        self.assertEqual(evicted, ExpertID(layer=0, idx=0))
        self.assertEqual(self.pool.free_slots, 8)  # 10 - 3 + 1

    def test_touch(self):
        """Test LRU touch operation"""
        # Allocate experts
        for i in range(3):
            expert = ExpertID(layer=0, idx=i)
            residency = Residency(bitwidth="W4", location="HBM", bytes=100)
            self.pool.allocate(expert, residency)

        # Touch expert 0 multiple times
        self.pool.touch(ExpertID(layer=0, idx=0))

        # Evict should remove expert 1 (now least recently used)
        evicted = self.pool.evict_lru()
        self.assertEqual(evicted, ExpertID(layer=0, idx=1))

    def test_remove(self):
        """Test explicit removal"""
        expert = ExpertID(layer=0, idx=0)
        residency = Residency(bitwidth="W4", location="HBM", bytes=100)
        self.pool.allocate(expert, residency)

        result = self.pool.remove(expert)

        self.assertTrue(result)
        self.assertFalse(self.pool.contains(expert))
        self.assertEqual(self.pool.free_slots, 10)

    def test_get_usage(self):
        """Test usage calculation"""
        self.assertEqual(self.pool.get_usage(), 0.0)

        # Allocate half
        for i in range(5):
            expert = ExpertID(layer=0, idx=i)
            residency = Residency(bitwidth="W4", location="HBM", bytes=100)
            self.pool.allocate(expert, residency)

        self.assertAlmostEqual(self.pool.get_usage(), 0.5)


class TestMemoryManager(unittest.TestCase):
    """Test MemoryManager functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.memmgr = MemoryManager(
            hot_pool_gb=0.001,  # 1 MB
            cold_pool_gb=0.001,  # 1 MB
            transient_pool_mb=1.0,
            w4_expert_size_mb=0.1,  # 100 KB
            w2_expert_size_mb=0.05,  # 50 KB
        )

    def test_initialization(self):
        """Test manager initialization"""
        self.assertIsNotNone(self.memmgr.hot_pool)
        self.assertIsNotNone(self.memmgr.cold_pool)
        self.assertIsNotNone(self.memmgr.transient_pool)

    def test_reserve_hot(self):
        """Test hot pool reservation"""
        expert = ExpertID(layer=0, idx=0)

        result = self.memmgr.reserve_hot(expert, self.memmgr.w4_size)

        self.assertTrue(result)
        self.assertTrue(self.memmgr.hot_pool.contains(expert))

    def test_reserve_hot_with_eviction(self):
        """Test hot pool reservation with automatic eviction"""
        # Fill hot pool
        experts = []
        for i in range(self.memmgr.hot_pool.max_slots):
            expert = ExpertID(layer=0, idx=i)
            experts.append(expert)
            self.memmgr.reserve_hot(expert, self.memmgr.w4_size)

        # Reserve one more (should evict LRU)
        new_expert = ExpertID(layer=0, idx=100)
        result = self.memmgr.reserve_hot(new_expert, self.memmgr.w4_size)

        self.assertTrue(result)
        self.assertTrue(self.memmgr.hot_pool.contains(new_expert))
        # First expert should be evicted
        self.assertFalse(self.memmgr.hot_pool.contains(experts[0]))

    def test_reserve_cold(self):
        """Test cold pool reservation"""
        expert = ExpertID(layer=0, idx=0)

        result = self.memmgr.reserve_cold(expert, self.memmgr.w2_size)

        self.assertTrue(result)
        self.assertTrue(self.memmgr.cold_pool.contains(expert))

    def test_get_residency(self):
        """Test residency lookup"""
        expert = ExpertID(layer=0, idx=0)
        self.memmgr.reserve_hot(expert, self.memmgr.w4_size)

        residency = self.memmgr.get_residency(expert)

        self.assertIsNotNone(residency)
        self.assertEqual(residency.bitwidth, "W4")
        self.assertEqual(residency.location, "HBM")

    def test_hbm_pressure(self):
        """Test HBM pressure calculation"""
        # Empty pools
        self.assertEqual(self.memmgr.get_hbm_pressure(), 0.0)

        # Fill half of hot pool
        for i in range(self.memmgr.hot_pool.max_slots // 2):
            expert = ExpertID(layer=0, idx=i)
            self.memmgr.reserve_hot(expert, self.memmgr.w4_size)

        pressure = self.memmgr.get_hbm_pressure()
        self.assertGreater(pressure, 0.0)
        self.assertLessEqual(pressure, 1.0)

    def test_statistics(self):
        """Test statistics reporting"""
        stats = self.memmgr.get_statistics()

        self.assertIn("hot_pool", stats)
        self.assertIn("cold_pool", stats)
        self.assertIn("transient_pool", stats)
        self.assertIn("total_experts", stats)
        self.assertIn("hbm_pressure", stats)


if __name__ == '__main__':
    unittest.main()
