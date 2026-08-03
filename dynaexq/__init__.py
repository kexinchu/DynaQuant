"""
DynaExQ — Dynamic Expert Precision Orchestration.

This package contains the runtime components required to monitor expert
hotness, choose precision tiers, manage memory pools, and orchestrate
asynchronous upgrades/downgrades of expert weights across the storage
hierarchy.
"""

from . import core

__all__ = ["core"]
