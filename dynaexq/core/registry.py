"""
ExpertRegistry: atomic "last stable representation" handle management.

Provides thread-safe access to expert handles used by forward pass.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

import torch

from .config import Tier


@dataclass
class ExpertKey:
    """Unique identifier for an expert."""
    layer: int
    expert: int
    
    def __hash__(self) -> int:
        return hash((self.layer, self.expert))
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, ExpertKey):
            return False
        return self.layer == other.layer and self.expert == other.expert


@dataclass
class ExpertHandle:
    """
    Handle to expert representation used by forward pass.
    
    This is the "last stable registered" representation that forward
    always binds to. Transitions update this atomically.
    """
    tier: Tier
    device_ptr: Optional[torch.Tensor] = None  # Tensor handle or storage
    format: str = ""  # "fp16", "int4", "int2"
    bytes: int = 0
    version: int = 0  # Monotonically increasing per expert
    
    def is_valid(self) -> bool:
        """Check if handle is valid."""
        return self.device_ptr is not None


class ExpertRegistry:
    """
    Thread-safe registry mapping ExpertKey -> ExpertHandle.
    
    Forward pass reads handles without blocking; transitions update
    handles atomically after completion.
    """
    
    def __init__(self):
        self._handles: dict[ExpertKey, ExpertHandle] = {}
        self._locks: dict[ExpertKey, threading.Lock] = {}
        self._global_lock = threading.Lock()
    
    def get_handle(self, key: ExpertKey) -> Optional[ExpertHandle]:
        """
        Get current handle for an expert (non-blocking read).
        
        Returns the last stable registered handle.
        """
        with self._global_lock:
            return self._handles.get(key)
    
    def register(self, key: ExpertKey, handle: ExpertHandle) -> None:
        """
        Register a new handle (atomic update).
        
        This is called by TransitionEngine after a transition completes.
        The handle becomes visible to forward pass immediately.
        """
        with self._global_lock:
            old_handle = self._handles.get(key)
            if old_handle is not None:
                # Increment version
                handle.version = old_handle.version + 1
            self._handles[key] = handle
    
    def get_old_handle(self, key: ExpertKey) -> Optional[ExpertHandle]:
        """Get the handle that will be replaced (for cleanup)."""
        with self._global_lock:
            return self._handles.get(key)
    
    def _get_lock(self, key: ExpertKey) -> threading.Lock:
        """Get per-expert lock (for transition engine use)."""
        with self._global_lock:
            if key not in self._locks:
                self._locks[key] = threading.Lock()
            return self._locks[key]
    
    def acquire_lock(self, key: ExpertKey) -> threading.Lock:
        """Acquire per-expert lock (used by TransitionEngine)."""
        return self._get_lock(key)

