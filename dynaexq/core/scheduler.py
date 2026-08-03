"""
PrecisionScheduler: computes H_l and emits transition requests.

Implements top-n projection with δ-margin hysteresis (plan §III-B / §6).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .config import Tier
from .hotness_tracker import HotnessTracker
from .registry import ExpertKey


@dataclass
class TransitionReq:
    """Transition request for an expert."""
    key: ExpertKey
    src: Tier
    dst: Tier
    reason: str  # e.g., "enter_hi_topn" / "leave_hi_topn"
    issued_step: int
    score_gap: float = 0.0  # For prioritization


class PrecisionScheduler:
    """
    Schedules precision transitions based on hotness scores.

    Every T_u steps, computes the new HI residents per layer using a
    **δ-margin top-N projection** (plan §III-B / Phase 5):

    1. Each layer has a fixed quota ``n_hi[l]`` HI residents (paper
       §III-A budget initialization).
    2. **Cold start** (fewer residents than quota): fill the slack with
       the top scoring outsiders, no δ check. This is the warm-up that
       must happen on the first scheduler tick before any expert has
       been promoted.
    3. **Hot path** (quota full): we sort residents ascending and
       outsiders descending by score, then pair them up. An outsider
       displaces a resident **only if** ``S[outsider] > S[resident] +
       δ`` (the symmetric eviction condition ``S[resident] <
       S[outsider] - δ`` is the same inequality, just from the other
       side). The first pair that fails the δ check ends the swap loop
       — outsiders are sorted descending so no further outsider could
       beat the same resident either.

    The δ margin (``delta_score_margin``) suppresses oscillation when
    two experts trade places near the resident/outsider boundary.
    Default ``0.0`` reproduces the pre-Phase-5 plain top-N behavior so
    every existing test continues to pass without re-tuning.

    Rate limiting: if ``rate_limit`` is set, the engine emits at most
    that many requests per tick. A steady-state replacement is treated
    as an indivisible two-request unit (demotion followed by promotion),
    so rate limiting cannot accidentally change the layer quota by
    retaining only one side of a swap.
    """

    def __init__(
        self,
        num_layers: int,
        experts_per_layer: list[int] | int,
        n_hi: list[int],
        update_period_steps: int = 200,
        rate_limit: Optional[int] = None,
        delta_score_margin: float = 0.0,
    ):
        """
        Args:
            num_layers: Number of MoE layers
            experts_per_layer: Expert counts per layer
            n_hi: n_hi[l] per layer (fixed by budget init)
            update_period_steps: T_u - update period
            rate_limit: Optional rate limit R per period
            delta_score_margin: δ margin in score units. ``0.0`` = no
                hysteresis. Recommended to start around 5-10% of the
                expected steady-state score range; plan §10 "Hysteresis
                δ 调不出论文数字" calls out exposing this as a CLI flag.
        """
        self.num_layers = num_layers
        if isinstance(experts_per_layer, int):
            self.experts_per_layer = [experts_per_layer] * num_layers
        else:
            self.experts_per_layer = list(experts_per_layer)

        if len(n_hi) != num_layers:
            raise ValueError(f"n_hi length {len(n_hi)} != num_layers {num_layers}")
        self.n_hi = list(n_hi)

        if delta_score_margin < 0:
            raise ValueError(
                f"delta_score_margin must be non-negative, got {delta_score_margin}"
            )
        if update_period_steps <= 0:
            raise ValueError(
                f"update_period_steps must be positive, got {update_period_steps}"
            )
        if rate_limit is not None and rate_limit <= 0:
            raise ValueError(f"rate_limit must be positive or None, got {rate_limit}")

        self.update_period_steps = update_period_steps
        self.rate_limit = rate_limit
        self.delta_score_margin = delta_score_margin

        # Current tier assignments (for comparison)
        self._current_tiers: dict[ExpertKey, Tier] = {}
        self._last_update_step = -1

    def should_update(self, step: int) -> bool:
        """Check if scheduler should update at this step."""
        return step % self.update_period_steps == 0 and step != self._last_update_step

    def plan(
        self,
        step: int,
        tracker: HotnessTracker,
        current_tiers: Optional[dict[ExpertKey, Tier]] = None,
    ) -> list[TransitionReq]:
        """
        Plan transitions for current step.

        Args:
            step: Current step number
            tracker: HotnessTracker with current scores
            current_tiers: Current tier assignments (if None, uses internal state)

        Returns:
            List of TransitionReq
        """
        if not self.should_update(step):
            return []

        if current_tiers is not None:
            self._current_tiers = current_tiers.copy()

        requests: list[TransitionReq] = []
        delta = self.delta_score_margin

        for layer in range(self.num_layers):
            scores = tracker.get_layer_scores(layer)
            n_hi_l = self.n_hi[layer]
            n_experts_l = self.experts_per_layer[layer]

            if len(scores) == 0 or n_hi_l <= 0:
                continue

            target_residents = self._project_layer(
                layer, scores, n_hi_l, n_experts_l, delta
            )

            # Diff against current assignments → emit transition requests
            for expert_id in range(n_experts_l):
                key = ExpertKey(layer=layer, expert=expert_id)
                want_hi = expert_id in target_residents
                have_hi = self._current_tiers.get(key, Tier.LO) == Tier.HI

                if want_hi and not have_hi:
                    score = float(scores[expert_id]) if expert_id < len(scores) else 0.0
                    boundary = (
                        float(min(scores[i] for i in target_residents))
                        if target_residents
                        else 0.0
                    )
                    requests.append(
                        TransitionReq(
                            key=key,
                            src=Tier.LO,
                            dst=Tier.HI,
                            reason="enter_hi_topn",
                            issued_step=step,
                            score_gap=score - boundary,
                        )
                    )
                elif not want_hi and have_hi:
                    score = float(scores[expert_id]) if expert_id < len(scores) else 0.0
                    boundary = (
                        float(min(scores[i] for i in target_residents))
                        if target_residents
                        else 0.0
                    )
                    requests.append(
                        TransitionReq(
                            key=key,
                            src=Tier.HI,
                            dst=Tier.LO,
                            reason="leave_hi_topn",
                            issued_step=step,
                            score_gap=boundary - score,
                        )
                    )

        requests = self._order_and_limit_requests(requests)

        for req in requests:
            self._current_tiers[req.key] = req.dst

        self._last_update_step = step
        return requests

    def _order_and_limit_requests(
        self, requests: list[TransitionReq]
    ) -> list[TransitionReq]:
        """Order evictions before upgrades and preserve complete swaps.

        Requests are paired within a layer because the projection keeps a
        fixed HI quota once warm-up has filled it.  Unpaired requests are
        possible only during cold start or when externally supplied state
        violates the quota; those remain valid one-request units.
        """
        units: list[tuple[float, list[TransitionReq]]] = []
        for layer in range(self.num_layers):
            layer_requests = [r for r in requests if r.key.layer == layer]
            demotions = sorted(
                (r for r in layer_requests if r.dst == Tier.LO),
                key=lambda r: r.score_gap,
                reverse=True,
            )
            promotions = sorted(
                (r for r in layer_requests if r.dst == Tier.HI),
                key=lambda r: r.score_gap,
                reverse=True,
            )
            paired = min(len(demotions), len(promotions))
            for index in range(paired):
                demote = demotions[index]
                promote = promotions[index]
                priority = max(demote.score_gap, promote.score_gap)
                # Reclaim a resident slot before attempting to consume it.
                units.append((priority, [demote, promote]))
            for req in demotions[paired:] + promotions[paired:]:
                units.append((req.score_gap, [req]))

        units.sort(key=lambda item: item[0], reverse=True)
        if self.rate_limit is None:
            return [req for _, unit in units for req in unit]

        selected: list[TransitionReq] = []
        for _, unit in units:
            if len(selected) + len(unit) > self.rate_limit:
                continue
            selected.extend(unit)
        return selected

    # ------------------------------------------------------------------
    # δ-margin projection (plan §III-B / Phase 5)
    # ------------------------------------------------------------------

    def _project_layer(
        self,
        layer: int,
        scores: np.ndarray,
        n_hi_l: int,
        n_experts_l: int,
        delta: float,
    ) -> set[int]:
        """
        Compute the **target** HI resident set for a single layer under
        the δ-margin policy.

        Returns the set of expert ids that *should* be HI after this
        tick. The caller diffs this against ``self._current_tiers`` to
        produce promote/demote requests.
        """
        n_scored = len(scores)

        # Degenerate case: quota covers all (or more than) the experts.
        if n_hi_l >= n_scored:
            return set(range(n_scored))

        # Identify current residents within this layer that are still
        # in the score-bearing range (some experts may have stale ids).
        current_residents = {
            e
            for e in range(n_experts_l)
            if e < n_scored
            and self._current_tiers.get(ExpertKey(layer, e), Tier.LO) == Tier.HI
        }

        # Cold-start path: fewer residents than quota → fill with the
        # top scorers, no δ check (we never need hysteresis to *create*
        # the first set; we only need it to *swap* an existing one).
        if len(current_residents) < n_hi_l:
            top = np.argsort(-scores, kind="stable")[:n_hi_l]
            return set(int(i) for i in top.tolist())

        # Quota is full → δ-margin swap path.
        # Sort residents ascending (lowest score first) so we evict the
        # weakest first; sort outsiders descending (highest first) so we
        # promote the strongest first.
        residents_sorted = sorted(current_residents, key=lambda e: float(scores[e]))
        outsiders_sorted = sorted(
            (e for e in range(n_scored) if e not in current_residents),
            key=lambda e: -float(scores[e]),
        )

        target = set(current_residents)
        r_idx = 0
        for o in outsiders_sorted:
            if r_idx >= len(residents_sorted):
                break
            r = residents_sorted[r_idx]
            if float(scores[o]) - float(scores[r]) > delta:
                target.discard(r)
                target.add(o)
                r_idx += 1
            else:
                # outsiders sorted desc → no later outsider can beat r
                break
        return target
