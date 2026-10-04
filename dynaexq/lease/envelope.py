"""Adaptive lookahead envelope.

``S_t`` bounds which future layers can generate candidates. It does not reserve
bytes and it is not a lookahead quota. Stall lengthens it by one layer;
pollution shortens it by one layer; values inside the hysteresis band do nothing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EnvelopeState:
    depth: int
    stall_ema: float = 0.0
    pollution_ema: float = 0.0


def initial_depth(
    absent_per_layer: float,
    bytes_per_expert: float,
    bandwidth_bytes_s: float,
    layer_time_s: float,
    *,
    s_min: int,
    s_max: int,
) -> int:
    """Cold start from the work that overlaps one layer."""
    if s_min > s_max or s_min < 0:
        raise ValueError("envelope bounds are inconsistent")
    window = bandwidth_bytes_s * layer_time_s
    if absent_per_layer <= 0 or bytes_per_expert <= 0 or window <= 0:
        return s_min
    raw = math.ceil(absent_per_layer * bytes_per_expert / window)
    return min(s_max, max(s_min, raw))


def pollution_ratio(
    unused_bytes: int,
    reload_bytes: int,
    admitted_lookahead_bytes: int,
    block_bytes: int,
) -> float:
    """Unused lookahead plus eviction reloads, over lookahead traffic."""
    if block_bytes <= 0:
        raise ValueError("block size must be positive")
    return (unused_bytes + reload_bytes) / max(admitted_lookahead_bytes, block_bytes)


def update_envelope(
    state: EnvelopeState,
    *,
    stall: float,
    pollution: float,
    beta: float,
    lam: float,
    theta: float,
    s_min: int,
    s_max: int,
) -> EnvelopeState:
    """One hysteresis step. The depth changes by at most one layer."""
    if not 0.0 <= beta < 1.0:
        raise ValueError("beta must be in [0, 1)")
    if s_min > s_max:
        raise ValueError("envelope bounds are inconsistent")
    stall_ema = beta * state.stall_ema + (1.0 - beta) * stall
    pollution_ema = beta * state.pollution_ema + (1.0 - beta) * pollution
    signal = stall_ema - lam * pollution_ema
    depth = state.depth
    if signal > theta:
        depth = min(depth + 1, s_max)
    elif signal < -theta:
        depth = max(depth - 1, s_min)
    return EnvelopeState(depth, stall_ema, pollution_ema)
