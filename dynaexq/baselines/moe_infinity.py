"""Pinned MoE-Infinity identity and fail-closed runtime telemetry.

This module does not reimplement MoE-Infinity.  It verifies an official,
clean checkout and instruments the public Python prefetch path while preserving
the original call.  Formal artifacts therefore cannot be produced by the
project's simple LRU baseline or by an unpinned external source tree.
"""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


OFFICIAL_REPOSITORY = "https://github.com/EfficientMoE/MoE-Infinity"
PINNED_COMMIT = "ba5651897a80d9c9b7a1500cef2c68adaa63db0f"
PINNED_TREE = "6c463a9ab298f352b0c1e855961b82ce2c545a64"
PINNED_SOURCE_SHA256 = (
    "c9f83ea65a2ed83c3454af861560d666a5ada14134e4e8bcd6d389e8231db30b"
)
OPEN_SOURCE_VARIANT_NOTE = (
    "Current official open-source MoE-Infinity runtime; its README states "
    "that this code was redesigned and differs from the paper implementation."
)


def _git(repo: Path, *args: str, binary: bool = False) -> str | bytes:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if binary:
        return result.stdout
    return result.stdout.decode("utf-8").strip()


def _canonical_repository(value: str) -> str:
    normalized = value.strip().removesuffix(".git").rstrip("/")
    if normalized.startswith("git@github.com:"):
        normalized = (
            "https://github.com/"
            + normalized.removeprefix("git@github.com:")
        )
    return normalized


def committed_source_sha256(repo: Path) -> str:
    """Hash the recursive Git tree manifest, independent of checkout mode."""
    manifest = _git(repo, "ls-tree", "-r", "-z", "HEAD", binary=True)
    assert isinstance(manifest, bytes)
    return hashlib.sha256(manifest).hexdigest()


def verify_official_checkout(repo: Path) -> dict[str, Any]:
    """Return immutable source identity or reject a non-official checkout."""
    repo = repo.expanduser().resolve()
    if not (repo / ".git").exists():
        raise ValueError(f"not a Git checkout: {repo}")
    origin = str(_git(repo, "remote", "get-url", "origin"))
    if _canonical_repository(origin) != OFFICIAL_REPOSITORY:
        raise ValueError(
            f"unexpected MoE-Infinity origin: {origin!r}"
        )
    commit = str(_git(repo, "rev-parse", "HEAD"))
    if commit != PINNED_COMMIT:
        raise ValueError(
            f"MoE-Infinity must be checked out at {PINNED_COMMIT}, got {commit}"
        )
    tree = str(_git(repo, "rev-parse", "HEAD^{tree}"))
    if tree != PINNED_TREE:
        raise ValueError(f"unexpected MoE-Infinity source tree: {tree}")
    status = str(
        _git(repo, "status", "--porcelain", "--untracked-files=normal")
    )
    if status:
        raise ValueError("MoE-Infinity checkout is dirty")
    source_sha256 = committed_source_sha256(repo)
    if source_sha256 != PINNED_SOURCE_SHA256:
        raise ValueError(
            "MoE-Infinity committed source manifest does not match the pin"
        )
    return {
        "name": "MoE-Infinity",
        "repository": OFFICIAL_REPOSITORY,
        "origin": origin,
        "commit": commit,
        "tree": tree,
        "source_hash_algorithm": "sha256(git-ls-tree-r-z)",
        "source_hash": source_sha256,
        "clean": True,
        "paper_implementation_equivalent": False,
        "variant_note": OPEN_SOURCE_VARIANT_NOTE,
    }


def verify_import_from_checkout(module_file: str, repo: Path) -> str:
    """Prove that the imported package comes from the verified checkout."""
    source = Path(module_file).resolve()
    checkout = repo.expanduser().resolve()
    try:
        relative = source.relative_to(checkout)
    except ValueError as error:
        raise ValueError(
            f"imported moe_infinity is outside verified checkout: {source}"
        ) from error
    tracked = str(
        _git(checkout, "ls-files", "--error-unmatch", relative.as_posix())
    )
    if tracked != relative.as_posix():
        raise ValueError("imported moe_infinity module is not tracked")
    return relative.as_posix()


def count_offloaded_expert_tensors(runtime: Any) -> tuple[int, int]:
    """Count expert tensors and those the external engine marks offloaded."""
    engine = getattr(runtime, "engine", None)
    mapping = getattr(engine, "expert_tensor_map", None)
    archer = getattr(engine, "archer_engine", None)
    is_offloaded = getattr(archer, "is_tensor_offloaded", None)
    if not isinstance(mapping, dict) or not mapping or not callable(is_offloaded):
        raise RuntimeError(
            "MoE-Infinity offload-state API is unavailable at the pinned commit"
        )
    tensor_ids = sorted({int(tensor_id) for tensor_id in mapping.values()})
    count = sum(bool(is_offloaded(tensor_id)) for tensor_id in tensor_ids)
    return len(tensor_ids), count


@dataclass
class PrefetchTelemetry:
    """Measured-interval counters around the external prefetch call."""

    calls: int = 0
    requested_experts: int = 0
    layers: set[int] = field(default_factory=set)
    expert_ids: set[int] = field(default_factory=set)
    _restore: Callable[[], None] | None = field(default=None, repr=False)

    @classmethod
    def install(cls, runtime: Any) -> "PrefetchTelemetry":
        engine = getattr(runtime, "engine", None)
        prefetcher = getattr(engine, "expert_prefetcher", None)
        original = getattr(prefetcher, "prefetch_experts_list", None)
        if prefetcher is None or not callable(original):
            raise RuntimeError(
                "MoE-Infinity prefetch API is unavailable at the pinned commit"
            )
        telemetry = cls()

        def counted(layer_id: int, expert_list: Any) -> Any:
            experts = [int(value) for value in expert_list]
            telemetry.calls += 1
            telemetry.requested_experts += len(experts)
            telemetry.layers.add(int(layer_id))
            telemetry.expert_ids.update(experts)
            return original(layer_id, experts)

        setattr(prefetcher, "prefetch_experts_list", counted)

        def restore() -> None:
            setattr(prefetcher, "prefetch_experts_list", original)

        telemetry._restore = restore
        return telemetry

    def reset(self) -> None:
        self.calls = 0
        self.requested_experts = 0
        self.layers.clear()
        self.expert_ids.clear()

    def snapshot(
        self,
        *,
        total_expert_tensors: int,
        offloaded_expert_tensors: int,
    ) -> dict[str, Any]:
        return {
            "prefetch_calls": self.calls,
            "prefetch_requested_experts": self.requested_experts,
            "prefetch_layers_touched": sorted(self.layers),
            "prefetch_unique_experts": sorted(self.expert_ids),
            "total_expert_tensors": total_expert_tensors,
            "offloaded_expert_tensors": offloaded_expert_tensors,
        }

    def close(self) -> None:
        if self._restore is not None:
            self._restore()
            self._restore = None


def validate_runtime_configuration(runtime: Any) -> dict[str, bool]:
    """Verify the specific offload/prefetch features used by the paper run."""
    engine = getattr(runtime, "engine", None)
    config = getattr(engine, "archer_config", None)
    expected = {
        "prefetch": True,
        "speculative_prefetch": True,
        "speculative_prefetch_overlap": True,
        "use_native_engine": False,
    }
    observed = {
        key: bool(getattr(config, key, False))
        for key in expected
    }
    if observed != expected:
        raise RuntimeError(
            f"MoE-Infinity runtime feature mismatch: {observed}"
        )
    if not callable(getattr(runtime, "_configure_hook", None)):
        raise RuntimeError(
            "pinned MoE-Infinity request setup API is unavailable"
        )
    return observed
