"""
DynaExQ — Dynamic Expert Precision Orchestration.

This package contains the runtime components required to monitor expert
hotness, choose precision tiers, manage memory pools, and orchestrate
non-blocking upgrades/downgrades of expert weights across the storage
hierarchy.
"""

from . import runtime  # re-export runtime subpackage for convenience

__all__ = ["runtime"]

