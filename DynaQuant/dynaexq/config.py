"""
Configuration management for DynaExQ
"""

import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class DynaExQConfig:
    """
    Configuration manager for DynaExQ runtime.

    Loads from YAML file and provides typed access to settings.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to config file (None = use default)
        """
        if config_path is None:
            # Use default config
            config_path = Path(__file__).parent / "configs" / "default.yaml"

        self.config_path = Path(config_path)
        self.config = self._load_config()

        logger.info(f"Loaded config from {self.config_path}")

    def _load_config(self) -> Dict[str, Any]:
        """Load YAML config file"""
        if not self.config_path.exists():
            logger.warning(
                f"Config file not found: {self.config_path}, using defaults")
            return self._default_config()

        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)

        return config

    def _default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "hotness": {
                "window": 300,
                "ewma_alpha": 0.2,
            },
            "thresholds": {
                "tau_h": 0.65,
                "tau_c": 0.45,
            },
            "pool": {
                "hot_w4_slots": 16,
                "hot_pool_gb": 10.0,
                "cold_pool_gb": 5.0,
                "transient_mb": 2048,
            },
            "expert_sizes": {
                "w4_expert_mb": 256,
                "w2_expert_mb": 64,
            },
            "model": {
                "num_layers": 32,
                "num_experts_per_layer": 64,
                "top_k": 2,
            },
            "storage": {
                "ssd_path": "/tmp/dynaexq/experts.bin",
                "index_path": "/tmp/dynaexq/experts.index",
                "enable_ssd": False,
            },
            "streams": {
                "memcpy_h2d": 2,
                "memcpy_d2h": 1,
                "compute": 2,
            },
            "prefetch": {
                "lookahead_layers": 1,
                "prefetch_top_k": 8,
            },
            "telemetry": {
                "enable": True,
                "output_file": "./telemetry.jsonl",
                "export_format": "jsonl",
            },
            "adaptive": {
                "enable": True,
                "min_ready_ratio": 0.99,
                "max_hbm_pressure": 0.90,
            },
            "safety": {
                "max_swap_timeout_sec": 5.0,
                "fallback_to_w2_on_miss": True,
            }
        }

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get config value with dot notation.

        Example: config.get("pool.hot_w4_slots")
        """
        keys = key.split(".")
        value = self.config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def to_dict(self) -> Dict[str, Any]:
        """Get config as flat dictionary for runtime initialization"""
        return {
            # Monitor
            "ewma_alpha": self.get("hotness.ewma_alpha"),
            "epoch_duration": self.get("hotness.window"),
            "num_layers": self.get("model.num_layers"),
            "num_experts_per_layer": self.get("model.num_experts_per_layer"),

            # Controller
            "tau_h": self.get("thresholds.tau_h"),
            "tau_c": self.get("thresholds.tau_c"),
            "max_w4_slots": self.get("pool.hot_w4_slots"),

            # Memory Manager
            "hot_pool_gb": self.get("pool.hot_pool_gb"),
            "cold_pool_gb": self.get("pool.cold_pool_gb"),
            "transient_pool_mb": self.get("pool.transient_mb"),
            "w4_expert_size_mb": self.get("expert_sizes.w4_expert_mb"),
            "w2_expert_size_mb": self.get("expert_sizes.w2_expert_mb"),

            # Swap Engine
            "num_h2d_streams": self.get("streams.memcpy_h2d"),
            "num_d2h_streams": self.get("streams.memcpy_d2h"),

            # Prefetch
            "lookahead_layers": self.get("prefetch.lookahead_layers"),
            "prefetch_top_k": self.get("prefetch.prefetch_top_k"),

            # Telemetry
            "telemetry_file": (
                self.get("telemetry.output_file")
                if self.get("telemetry.enable") else None
            ),

            # Storage
            "ssd_path": self.get("storage.ssd_path"),
            "index_path": self.get("storage.index_path"),
            "enable_ssd": self.get("storage.enable_ssd"),
        }

    def save(self, output_path: str):
        """Save current config to file"""
        with open(output_path, 'w') as f:
            yaml.dump(self.config, f, indent=2)
        logger.info(f"Saved config to {output_path}")

    def update(self, updates: Dict[str, Any]):
        """
        Update config with new values.

        Args:
            updates: Dictionary of updates (supports dot notation keys)
        """
        for key, value in updates.items():
            keys = key.split(".")
            target = self.config

            for k in keys[:-1]:
                if k not in target:
                    target[k] = {}
                target = target[k]

            target[keys[-1]] = value
            logger.debug(f"Updated config: {key} = {value}")


def load_config(config_path: Optional[str] = None) -> DynaExQConfig:
    """
    Convenience function to load configuration.

    Args:
        config_path: Path to config file (None = use default)

    Returns:
        DynaExQConfig instance
    """
    return DynaExQConfig(config_path)
