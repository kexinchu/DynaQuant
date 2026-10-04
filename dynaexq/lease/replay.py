"""Single-stream deadline replay of expert transfers.

One migration stream serves every layer. An expert that is already published
is ready immediately. A prefetch is useful only when its copy completes before
the layer that consumes it. A demand miss is issued when the layer starts and
stalls that layer until the copy finishes.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class LayerResult:
    layer: int
    exposed_s: float
    demand_bytes: int
    prefetched_hits: int
    late_prefetches: int


@dataclass
class ForwardResult:
    exposed_s: float
    demand_bytes: int
    layers: list[LayerResult] = field(default_factory=list)

    @property
    def exposed_ms(self) -> float:
        return 1000.0 * self.exposed_s


def simulate_forward(
    active: list[set[int]],
    resident: list[dict[int, str]],
    prefetch_at: list[list[tuple[int, int, str]]],
    *,
    layer_time_s: float,
    bandwidth_bytes_s: float,
    lo_bytes: int,
    hi_bytes: int,
    default_miss_tier: str = "lo",
) -> ForwardResult:
    """Replay one request.

    ``prefetch_at[i]`` is issued while layer ``i`` computes. Each entry is
    ``(future_layer, expert, tier)``. ``resident[i]`` maps an expert to the
    tier that is already published there.
    """
    if len(resident) != len(active) or len(prefetch_at) != len(active):
        raise ValueError("active, resident, and prefetch plans must align")
    if layer_time_s < 0 or bandwidth_bytes_s <= 0:
        raise ValueError("layer time and bandwidth must be positive")
    if lo_bytes <= 0 or hi_bytes < lo_bytes:
        raise ValueError("tier sizes are inconsistent")

    def nbytes(tier: str) -> int:
        if tier == "lo":
            return lo_bytes
        if tier == "hi":
            return hi_bytes
        raise ValueError(f"unknown tier {tier}")

    clock = 0.0
    stream_free = 0.0
    pending: dict[tuple[int, int], float] = {}
    exposed = 0.0
    demand_bytes = 0
    layers: list[LayerResult] = []

    for index, needed in enumerate(active):
        arrivals: list[float] = []
        layer_demand = 0
        hits = 0
        late = 0
        for expert in sorted(needed):
            published = resident[index].get(expert)
            if published is not None:
                arrivals.append(0.0)
                continue
            ready = pending.get((index, expert))
            if ready is not None:
                arrivals.append(ready)
                if ready <= clock:
                    hits += 1
                else:
                    late += 1
                continue
            start = max(stream_free, clock)
            ready = start + nbytes(default_miss_tier) / bandwidth_bytes_s
            stream_free = ready
            arrivals.append(ready)
            layer_demand += nbytes(default_miss_tier)
        ready_count = 0
        miss_ready: list[float] = []
        for arrival, expert in zip(arrivals, sorted(needed)):
            published = resident[index].get(expert)
            pending_ready = pending.get((index, expert))
            on_time = published is not None or (
                pending_ready is not None and pending_ready <= clock
            )
            if on_time:
                ready_count += 1
            else:
                miss_ready.append(arrival)
        queue = max((ready - clock for ready in miss_ready), default=0.0)
        # Published experts in this layer run while the missing group waits.
        overlap = layer_time_s * ready_count / len(needed) if needed else 0.0
        stall = max(0.0, queue - overlap)
        exposed += stall
        demand_bytes += layer_demand
        layers.append(
            LayerResult(
                layer=index,
                exposed_s=stall,
                demand_bytes=layer_demand,
                prefetched_hits=hits,
                late_prefetches=late,
            )
        )
        compute_start = clock + stall
        for future, expert, tier in prefetch_at[index]:
            if future <= index or future >= len(active):
                continue
            if expert in resident[future] or (future, expert) in pending:
                continue
            start = max(stream_free, compute_start)
            ready = start + nbytes(tier) / bandwidth_bytes_s
            stream_free = ready
            pending[(future, expert)] = ready
        clock = compute_start + layer_time_s

    return ForwardResult(exposed_s=exposed, demand_bytes=demand_bytes, layers=layers)
