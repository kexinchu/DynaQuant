"""
Expert Cache for managing quantized expert weights with async swapping.
Maintains both W2 and W4 packed representations with warm pool and LRU eviction.
"""

import torch
import os
from typing import Dict, Tuple, Optional, List
from collections import OrderedDict
import pickle
import logging

logger = logging.getLogger(__name__)


class ExpertCache:
    """
    Cache for managing quantized expert weights with async GPU/CPU swapping.
    """

    def __init__(
        self,
        cache_dir: str,
        num_experts: int,
        warm_pool_size: int = 16,
        async_swap: bool = True,
        num_streams: int = 4,
        prepack_w2: bool = True,
        prepack_w4: bool = True,
    ):
        """
        Initialize expert cache.

        Args:
            cache_dir: Directory for storing cached expert weights
            num_experts: Total number of experts
            warm_pool_size: Number of experts to keep in GPU memory
            async_swap: Enable async H2D swap via CUDA streams
            num_streams: Number of CUDA streams for async operations
            prepack_w2: Store prepacked W2 weights
            prepack_w4: Store prepacked W4 weights
        """
        self.cache_dir = cache_dir
        self.num_experts = num_experts
        self.warm_pool_size = warm_pool_size
        self.async_swap = async_swap
        self.num_streams = num_streams
        self.prepack_w2 = prepack_w2
        self.prepack_w4 = prepack_w4

        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)

        # CUDA streams for async operations
        if async_swap and torch.cuda.is_available():
            self.streams = [torch.cuda.Stream() for _ in range(num_streams)]
        else:
            self.streams = []

        # GPU cache (LRU)
        self.gpu_cache = OrderedDict()  # expert_id -> (w2_weights, w4_weights, metadata)

        # Pinned CPU memory for faster transfers
        self.cpu_cache = {}  # expert_id -> (w2_weights, w4_weights, metadata)

        # Statistics
        self.cache_hits = 0
        self.cache_misses = 0
        self.num_swaps = 0

    def _get_expert_path(self, expert_id: int, precision: str) -> str:
        """Get file path for expert weights."""
        return os.path.join(self.cache_dir, f"expert_{expert_id}_{precision}.pkl")

    def save_expert(
        self,
        expert_id: int,
        w2_packed: Optional[torch.Tensor] = None,
        w2_scales: Optional[torch.Tensor] = None,
        w2_metadata: Optional[Dict] = None,
        w4_packed: Optional[torch.Tensor] = None,
        w4_scales: Optional[torch.Tensor] = None,
        w4_metadata: Optional[Dict] = None,
    ):
        """
        Save expert weights to disk.

        Args:
            expert_id: Expert ID
            w2_packed: Packed W2 weights
            w2_scales: W2 scales
            w2_metadata: W2 metadata
            w4_packed: Packed W4 weights
            w4_scales: W4 scales
            w4_metadata: W4 metadata
        """
        # Save W2
        if w2_packed is not None and self.prepack_w2:
            w2_path = self._get_expert_path(expert_id, "w2")
            data = {
                'packed': w2_packed.cpu(),
                'scales': w2_scales.cpu() if w2_scales is not None else None,
                'metadata': w2_metadata,
            }
            with open(w2_path, 'wb') as f:
                pickle.dump(data, f)

        # Save W4
        if w4_packed is not None and self.prepack_w4:
            w4_path = self._get_expert_path(expert_id, "w4")
            data = {
                'packed': w4_packed.cpu(),
                'scales': w4_scales.cpu() if w4_scales is not None else None,
                'metadata': w4_metadata,
            }
            with open(w4_path, 'wb') as f:
                pickle.dump(data, f)

    def load_expert_from_disk(self, expert_id: int, precision: str) -> Optional[Dict]:
        """
        Load expert weights from disk.

        Args:
            expert_id: Expert ID
            precision: Precision ("w2" or "w4")

        Returns:
            data: Dictionary with 'packed', 'scales', 'metadata' or None if not found
        """
        path = self._get_expert_path(expert_id, precision)

        if not os.path.exists(path):
            return None

        with open(path, 'rb') as f:
            data = pickle.load(f)

        return data

    def _evict_from_gpu_cache(self):
        """Evict least recently used expert from GPU cache."""
        if len(self.gpu_cache) >= self.warm_pool_size:
            # Remove oldest (least recently used)
            evicted_id, _ = self.gpu_cache.popitem(last=False)
            logger.debug(f"Evicted expert {evicted_id} from GPU cache")

    def get_expert(
        self,
        expert_id: int,
        precision: str,
        device: Optional[torch.device] = None,
    ) -> Optional[Dict]:
        """
        Get expert weights (from GPU cache, CPU cache, or disk).

        Args:
            expert_id: Expert ID
            precision: Precision ("w2" or "w4")
            device: Target device (defaults to cuda if available)

        Returns:
            data: Dictionary with 'packed', 'scales', 'metadata' or None if not found
        """
        if device is None:
            device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')

        cache_key = (expert_id, precision)

        # Check GPU cache
        if expert_id in self.gpu_cache:
            self.cache_hits += 1

            # Move to end (mark as recently used)
            w2_data, w4_data, metadata = self.gpu_cache.pop(expert_id)
            self.gpu_cache[expert_id] = (w2_data, w4_data, metadata)

            # Return requested precision
            if precision == "w2":
                return w2_data if w2_data is not None else None
            else:
                return w4_data if w4_data is not None else None

        # Cache miss
        self.cache_misses += 1

        # Check CPU cache
        if expert_id in self.cpu_cache:
            w2_data, w4_data, metadata = self.cpu_cache[expert_id]

            # Move to GPU cache
            self._evict_from_gpu_cache()

            # Transfer to GPU (async if enabled)
            if self.async_swap and self.streams:
                stream = self.streams[expert_id % len(self.streams)]
                with torch.cuda.stream(stream):
                    if w2_data is not None:
                        w2_data_gpu = {
                            'packed': w2_data['packed'].to(device, non_blocking=True),
                            'scales': w2_data['scales'].to(device, non_blocking=True) if w2_data['scales'] is not None else None,
                            'metadata': w2_data['metadata'],
                        }
                    else:
                        w2_data_gpu = None

                    if w4_data is not None:
                        w4_data_gpu = {
                            'packed': w4_data['packed'].to(device, non_blocking=True),
                            'scales': w4_data['scales'].to(device, non_blocking=True) if w4_data['scales'] is not None else None,
                            'metadata': w4_data['metadata'],
                        }
                    else:
                        w4_data_gpu = None

                stream.synchronize()
            else:
                # Synchronous transfer
                if w2_data is not None:
                    w2_data_gpu = {
                        'packed': w2_data['packed'].to(device),
                        'scales': w2_data['scales'].to(device) if w2_data['scales'] is not None else None,
                        'metadata': w2_data['metadata'],
                    }
                else:
                    w2_data_gpu = None

                if w4_data is not None:
                    w4_data_gpu = {
                        'packed': w4_data['packed'].to(device),
                        'scales': w4_data['scales'].to(device) if w4_data['scales'] is not None else None,
                        'metadata': w4_data['metadata'],
                    }
                else:
                    w4_data_gpu = None

            self.gpu_cache[expert_id] = (w2_data_gpu, w4_data_gpu, metadata)
            self.num_swaps += 1

            # Return requested precision
            if precision == "w2":
                return w2_data_gpu if w2_data_gpu is not None else None
            else:
                return w4_data_gpu if w4_data_gpu is not None else None

        # Load from disk
        logger.debug(
            f"Loading expert {expert_id} from disk (precision: {precision})")

        # Load both W2 and W4 if available (for caching)
        w2_data_disk = self.load_expert_from_disk(expert_id, "w2")
        w4_data_disk = self.load_expert_from_disk(expert_id, "w4")

        if w2_data_disk is None and w4_data_disk is None:
            logger.warning(f"Expert {expert_id} not found on disk")
            return None

        # Store in CPU cache (with pinned memory for faster transfers)
        if torch.cuda.is_available():
            if w2_data_disk is not None:
                w2_data_cpu = {
                    'packed': w2_data_disk['packed'].pin_memory(),
                    'scales': w2_data_disk['scales'].pin_memory() if w2_data_disk['scales'] is not None else None,
                    'metadata': w2_data_disk['metadata'],
                }
            else:
                w2_data_cpu = None

            if w4_data_disk is not None:
                w4_data_cpu = {
                    'packed': w4_data_disk['packed'].pin_memory(),
                    'scales': w4_data_disk['scales'].pin_memory() if w4_data_disk['scales'] is not None else None,
                    'metadata': w4_data_disk['metadata'],
                }
            else:
                w4_data_cpu = None
        else:
            w2_data_cpu = w2_data_disk
            w4_data_cpu = w4_data_disk

        self.cpu_cache[expert_id] = (w2_data_cpu, w4_data_cpu, {})

        # Move to GPU cache
        self._evict_from_gpu_cache()

        if w2_data_cpu is not None:
            w2_data_gpu = {
                'packed': w2_data_cpu['packed'].to(device),
                'scales': w2_data_cpu['scales'].to(device) if w2_data_cpu['scales'] is not None else None,
                'metadata': w2_data_cpu['metadata'],
            }
        else:
            w2_data_gpu = None

        if w4_data_cpu is not None:
            w4_data_gpu = {
                'packed': w4_data_cpu['packed'].to(device),
                'scales': w4_data_cpu['scales'].to(device) if w4_data_cpu['scales'] is not None else None,
                'metadata': w4_data_cpu['metadata'],
            }
        else:
            w4_data_gpu = None

        self.gpu_cache[expert_id] = (w2_data_gpu, w4_data_gpu, {})
        self.num_swaps += 1

        # Return requested precision
        if precision == "w2":
            return w2_data_gpu if w2_data_gpu is not None else None
        else:
            return w4_data_gpu if w4_data_gpu is not None else None

    def preload_experts(self, expert_ids: List[int]):
        """
        Preload experts into GPU cache.

        Args:
            expert_ids: List of expert IDs to preload
        """
        for expert_id in expert_ids:
            # Load both W2 and W4
            if self.prepack_w2:
                self.get_expert(expert_id, "w2")
            if self.prepack_w4:
                self.get_expert(expert_id, "w4")

    def clear_gpu_cache(self):
        """Clear GPU cache."""
        self.gpu_cache.clear()

    def clear_cpu_cache(self):
        """Clear CPU cache."""
        self.cpu_cache.clear()

    def get_statistics(self) -> Dict[str, any]:
        """Get cache statistics."""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0.0

        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
            'num_swaps': self.num_swaps,
            'gpu_cache_size': len(self.gpu_cache),
            'cpu_cache_size': len(self.cpu_cache),
            'warm_pool_size': self.warm_pool_size,
        }

    def reset_statistics(self):
        """Reset statistics counters."""
        self.cache_hits = 0
        self.cache_misses = 0
        self.num_swaps = 0


def test_expert_cache():
    """
    Unit tests for ExpertCache.
    """
    import logging
    import tempfile
    logger = logging.getLogger(__name__)

    logger.info("Testing ExpertCache...")

    # Test initialization
    logger.info("\n--- Testing initialization ---")

    with tempfile.TemporaryDirectory() as tmpdir:
        cache = ExpertCache(
            cache_dir=tmpdir,
            num_experts=64,
            warm_pool_size=4,
            async_swap=False,  # Disable for testing
            prepack_w2=True,
            prepack_w4=True,
        )

        logger.info(f"Cache directory: {cache.cache_dir}")
        logger.info(f"Warm pool size: {cache.warm_pool_size}")
        logger.info(f"✓ Initialization test passed")

        # Test save/load expert
        logger.info("\n--- Testing save/load expert ---")

        # Create fake expert weights
        torch.manual_seed(42)
        expert_id = 0

        w2_packed = torch.randint(0, 256, (512, 64), dtype=torch.uint8)
        w2_scales = torch.randn(512, 4)
        w2_metadata = {'bits': 2, 'group_size': 128}

        w4_packed = torch.randint(0, 256, (512, 128), dtype=torch.uint8)
        w4_scales = torch.randn(512, 4)
        w4_metadata = {'bits': 4, 'group_size': 128}

        # Save
        cache.save_expert(
            expert_id,
            w2_packed=w2_packed,
            w2_scales=w2_scales,
            w2_metadata=w2_metadata,
            w4_packed=w4_packed,
            w4_scales=w4_scales,
            w4_metadata=w4_metadata,
        )

        logger.info(f"Saved expert {expert_id}")

        # Load W2
        w2_data = cache.get_expert(expert_id, "w2", device=torch.device('cpu'))
        assert w2_data is not None
        assert torch.equal(w2_data['packed'], w2_packed)
        assert torch.equal(w2_data['scales'], w2_scales)
        assert w2_data['metadata'] == w2_metadata

        logger.info(f"Loaded W2 expert {expert_id}")
        logger.info(f"✓ W2 save/load test passed")

        # Load W4
        w4_data = cache.get_expert(expert_id, "w4", device=torch.device('cpu'))
        assert w4_data is not None
        assert torch.equal(w4_data['packed'], w4_packed)
        assert torch.equal(w4_data['scales'], w4_scales)
        assert w4_data['metadata'] == w4_metadata

        logger.info(f"Loaded W4 expert {expert_id}")
        logger.info(f"✓ W4 save/load test passed")

        # Test cache hit
        logger.info("\n--- Testing cache hit ---")
        stats_before = cache.get_statistics()

        # Load again (should be cache hit)
        w2_data_cached = cache.get_expert(
            expert_id, "w2", device=torch.device('cpu'))

        stats_after = cache.get_statistics()
        assert stats_after['cache_hits'] > stats_before['cache_hits']

        logger.info(f"Cache statistics: {stats_after}")
        logger.info(f"✓ Cache hit test passed")

        # Test LRU eviction
        logger.info("\n--- Testing LRU eviction ---")
        cache.clear_gpu_cache()
        cache.reset_statistics()

        # Save multiple experts
        for i in range(cache.warm_pool_size + 2):
            w2_packed_i = torch.randint(0, 256, (512, 64), dtype=torch.uint8)
            w2_scales_i = torch.randn(512, 4)

            cache.save_expert(
                i,
                w2_packed=w2_packed_i,
                w2_scales=w2_scales_i,
                w2_metadata=w2_metadata,
            )

        # Load all experts (should trigger evictions)
        for i in range(cache.warm_pool_size + 2):
            cache.get_expert(i, "w2", device=torch.device('cpu'))

        stats = cache.get_statistics()
        logger.info(f"GPU cache size: {stats['gpu_cache_size']}")
        logger.info(f"Num swaps: {stats['num_swaps']}")

        # GPU cache should be at most warm_pool_size
        assert stats['gpu_cache_size'] <= cache.warm_pool_size
        logger.info(f"✓ LRU eviction test passed")

        # Test preload
        logger.info("\n--- Testing preload ---")
        cache.clear_gpu_cache()
        cache.reset_statistics()

        expert_ids = [0, 1, 2]
        cache.preload_experts(expert_ids)

        stats = cache.get_statistics()
        logger.info(f"GPU cache size after preload: {stats['gpu_cache_size']}")
        assert stats['gpu_cache_size'] >= len(expert_ids)
        logger.info(f"✓ Preload test passed")

    logger.info("\n✓ All ExpertCache tests passed!")
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_expert_cache()
