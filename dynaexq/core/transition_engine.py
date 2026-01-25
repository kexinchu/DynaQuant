"""
TransitionEngine: non-blocking promotion/demotion pipeline.

Implements asynchronous transitions with stages: fetch, h2d, register, reclaim.
"""

from __future__ import annotations

import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Optional

import torch

from .config import Tier
from .memory_pool import PoolAllocator, PoolBlock
from .registry import ExpertHandle, ExpertKey, ExpertRegistry
from .scheduler import TransitionReq

# WeightStore is an abstract base - ModelWeightStore implements it
from .weight_store import ModelWeightStore

# For backward compatibility, alias WeightStore
WeightStore = ModelWeightStore


@dataclass
class TransitionStage:
    """Stage timing information."""
    fetch_ms: float = 0.0
    h2d_ms: float = 0.0
    register_ms: float = 0.0
    reclaim_ms: float = 0.0
    total_ms: float = 0.0


# WeightStore is now in weight_store.py


class TransitionEngine:
    """
    Executes expert precision transitions asynchronously.
    
    Pipeline stages:
    1. Fetch: Load weights from storage (SSD/DRAM)
    2. H2D Transfer: Copy to GPU using dedicated CUDA stream
    3. Register: Atomically update ExpertRegistry
    4. Reclaim: Free old block back to pool
    """
    
    def __init__(
        self,
        registry: ExpertRegistry,
        pool_allocator: PoolAllocator,
        weight_store: WeightStore,
        max_workers: int = 4,
        max_inflight: int = 4,
    ):
        """
        Args:
            registry: ExpertRegistry for handle updates
            pool_allocator: PoolAllocator for block allocation
            weight_store: WeightStore for loading weights
            max_workers: Max worker threads
            max_inflight: Max concurrent transitions
        """
        self.registry = registry
        self.pool_allocator = pool_allocator
        self.weight_store = weight_store
        self.max_workers = max_workers
        self.max_inflight = max_inflight
        
        # Queues
        self._promotion_queue: queue.Queue[TransitionReq] = queue.Queue()
        self._demotion_queue: queue.Queue[TransitionReq] = queue.Queue()
        
        # Executor for async execution
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._active_transitions: dict[ExpertKey, threading.Event] = {}
        self._transition_lock = threading.Lock()
        
        # CUDA stream for async transfers
        self._copy_stream = torch.cuda.Stream() if torch.cuda.is_available() else None
        
        # Statistics
        self._stats_lock = threading.Lock()
        self._total_promotions = 0
        self._total_demotions = 0
        self._stage_timings: list[TransitionStage] = []
    
    def enqueue(self, req: TransitionReq) -> bool:
        """
        Enqueue a transition request.
        
        Returns:
            True if enqueued, False if queue full
        """
        with self._transition_lock:
            if len(self._active_transitions) >= self.max_inflight:
                return False
            
            key = req.key
            if key in self._active_transitions:
                return False  # Already in progress
            
            self._active_transitions[key] = threading.Event()
        
        if req.dst == Tier.HI:
            self._promotion_queue.put(req)
        else:
            self._demotion_queue.put(req)
        
        # Submit async execution
        self._executor.submit(self._execute_transition, req)
        return True
    
    def _execute_transition(self, req: TransitionReq) -> None:
        """Execute a single transition (runs in background thread)."""
        key = req.key
        stage = TransitionStage()
        start_time = time.time()
        
        try:
            # Stage 1: Fetch weights
            fetch_start = time.time()
            weights = self.weight_store.load_weights(key, req.dst)
            stage.fetch_ms = (time.time() - fetch_start) * 1000
            
            # Stage 2: Allocate block and transfer
            h2d_start = time.time()
            block = self.pool_allocator.alloc(key.layer, req.dst)
            if block is None:
                raise RuntimeError(f"Failed to allocate block for {key}")
            
            # Copy weights to block (using async stream if available)
            # Flatten weights and view as uint8 for byte-level copy
            weights_flat = weights.flatten().contiguous()
            # View as uint8 bytes
            weights_bytes = weights_flat.view(torch.uint8)
            
            # Copy to block (truncate if needed)
            copy_len = min(weights_bytes.numel(), block.tensor.numel())
            
            if self._copy_stream is not None:
                with torch.cuda.stream(self._copy_stream):
                    block.tensor[:copy_len].copy_(weights_bytes[:copy_len])
                torch.cuda.synchronize(self._copy_stream)
            else:
                block.tensor[:copy_len].copy_(weights_bytes[:copy_len])
            
            stage.h2d_ms = (time.time() - h2d_start) * 1000
            
            # Stage 3: Register new handle
            register_start = time.time()
            new_handle = ExpertHandle(
                tier=req.dst,
                device_ptr=block.tensor,
                format=self._tier_to_format(req.dst),
                bytes=self.weight_store.get_byte_size(key, req.dst),
                version=0,  # Will be incremented by registry
            )
            
            # Get old handle for cleanup
            old_handle = self.registry.get_old_handle(key)
            
            # Register atomically
            self.registry.register(key, new_handle)
            stage.register_ms = (time.time() - register_start) * 1000
            
            # Stage 4: Reclaim old block
            reclaim_start = time.time()
            if old_handle is not None and old_handle.device_ptr is not None:
                # Find block ID (simplified - in practice need block tracking)
                # For now, just mark old handle as invalid
                pass
            
            # Free old block if we can identify it
            # (This is simplified; real implementation needs block tracking)
            stage.reclaim_ms = (time.time() - reclaim_start) * 1000
            
            stage.total_ms = (time.time() - start_time) * 1000
            
            # Update statistics
            with self._stats_lock:
                if req.dst == Tier.HI:
                    self._total_promotions += 1
                else:
                    self._total_demotions += 1
                self._stage_timings.append(stage)
        
        except Exception as e:
            # Log error (in real implementation)
            print(f"Transition failed for {key}: {e}")
        finally:
            # Mark transition complete
            with self._transition_lock:
                if key in self._active_transitions:
                    self._active_transitions[key].set()
                    del self._active_transitions[key]
    
    def wait_ready(self, key: ExpertKey, timeout: Optional[float] = None) -> bool:
        """Wait for a transition to complete."""
        with self._transition_lock:
            event = self._active_transitions.get(key)
        
        if event is None:
            return True  # No transition in progress
        
        return event.wait(timeout=timeout)
    
    def _tier_to_format(self, tier: Tier) -> str:
        """Convert tier to format string."""
        if tier == Tier.HI:
            return "fp16"  # or "int4" depending on config
        else:
            return "int4"  # or "int2" depending on config
    
    def get_stats(self) -> dict:
        """Get transition statistics."""
        with self._stats_lock:
            return {
                "total_promotions": self._total_promotions,
                "total_demotions": self._total_demotions,
                "stage_timings": [
                    {
                        "fetch_ms": s.fetch_ms,
                        "h2d_ms": s.h2d_ms,
                        "register_ms": s.register_ms,
                        "reclaim_ms": s.reclaim_ms,
                        "total_ms": s.total_ms,
                    }
                    for s in self._stage_timings
                ],
            }
    
    def shutdown(self) -> None:
        """Shutdown the transition engine."""
        self._executor.shutdown(wait=True)

