"""
Workload generation for experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import json


@dataclass
class PhaseConfig:
    """Configuration for a workload phase."""
    name: str
    dataset_path: str
    duration_s: int
    concurrency: int = 64


class WorkloadStream:
    """
    Generates workload streams for shift experiments.
    
    Supports phased workloads: A -> B -> C -> repeat
    """
    
    def __init__(
        self,
        phases: list[PhaseConfig],
        cycles: int = 3,
    ):
        """
        Args:
            phases: List of phase configurations
            cycles: Number of cycles to repeat
        """
        self.phases = phases
        self.cycles = cycles
    
    def load_requests(self, dataset_path: str) -> list[dict]:
        """Load requests from dataset file."""
        path = Path(dataset_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        
        requests = []
        if path.suffix == ".jsonl":
            with open(path, "r") as f:
                for line in f:
                    if line.strip():
                        requests.append(json.loads(line))
        elif path.suffix == ".json":
            with open(path, "r") as f:
                data = json.load(f)
                if isinstance(data, list):
                    requests = data
                else:
                    requests = [data]
        
        return requests
    
    def generate_phases(self) -> Iterator[tuple[str, list[dict]]]:
        """Generate phases in sequence."""
        for cycle in range(self.cycles):
            for phase in self.phases:
                requests = self.load_requests(phase.dataset_path)
                yield phase.name, requests

