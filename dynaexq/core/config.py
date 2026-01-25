"""
Configuration management for DynaExq.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Optional

import yaml


class Tier(Enum):
    """Precision tier for experts."""
    HI = auto()
    LO = auto()


@dataclass
class ModelConfig:
    """Model configuration."""
    name: str
    layers: int
    experts_per_layer: int
    topk: int


@dataclass
class PrecisionConfig:
    """Precision configuration."""
    hi: str  # e.g., "fp16", "int4"
    lo: str  # e.g., "int4", "int2"


@dataclass
class SchedulerConfig:
    """Scheduler configuration."""
    alpha: float = 0.9  # EMA smoothing factor
    update_period_steps: int = 200  # T_u: update period in steps
    rate_limit: Optional[int] = None  # Optional rate limit R


@dataclass
class MemoryConfig:
    """Memory configuration."""
    device_mem_bytes: int
    reserve_kv_bytes: int = 0
    reserve_act_bytes: int = 0
    reserve_dense_bytes: int = 0
    safety_margin_bytes: int = 0
    max_inflight: int = 4  # Max concurrent transitions


@dataclass
class ExperimentConfig:
    """Experiment configuration."""
    concurrency: int = 64
    input_tokens: int = 2048
    output_tokens: int = 256
    shift_phases: list[str] = field(default_factory=lambda: ["wikitext", "gsm8k", "humaneval"])
    phase_duration_s: int = 180
    cycles: int = 3


@dataclass
class DynaExqConfig:
    """Complete DynaExq configuration."""
    model: ModelConfig
    precision: PrecisionConfig
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    experiments: ExperimentConfig = field(default_factory=ExperimentConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> DynaExqConfig:
        """Load configuration from YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        
        return cls(
            model=ModelConfig(**data["model"]),
            precision=PrecisionConfig(**data["precision"]),
            scheduler=SchedulerConfig(**data.get("scheduler", {})),
            memory=MemoryConfig(**data.get("memory", {})),
            experiments=ExperimentConfig(**data.get("experiments", {})),
        )

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to YAML file."""
        data = {
            "model": {
                "name": self.model.name,
                "layers": self.model.layers,
                "experts_per_layer": self.model.experts_per_layer,
                "topk": self.model.topk,
            },
            "precision": {
                "hi": self.precision.hi,
                "lo": self.precision.lo,
            },
            "scheduler": {
                "alpha": self.scheduler.alpha,
                "update_period_steps": self.scheduler.update_period_steps,
                "rate_limit": self.scheduler.rate_limit,
            },
            "memory": {
                "device_mem_bytes": self.memory.device_mem_bytes,
                "reserve_kv_bytes": self.memory.reserve_kv_bytes,
                "reserve_act_bytes": self.memory.reserve_act_bytes,
                "reserve_dense_bytes": self.memory.reserve_dense_bytes,
                "safety_margin_bytes": self.memory.safety_margin_bytes,
                "max_inflight": self.memory.max_inflight,
            },
            "experiments": {
                "concurrency": self.experiments.concurrency,
                "input_tokens": self.experiments.input_tokens,
                "output_tokens": self.experiments.output_tokens,
                "shift_phases": self.experiments.shift_phases,
                "phase_duration_s": self.experiments.phase_duration_s,
                "cycles": self.experiments.cycles,
            },
        }
        
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)

