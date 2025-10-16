"""
SafeTensors Model Saver

Saves quantized models in SafeTensors format with proper index files,
compatible with HuggingFace format for easy inference.
"""

import torch
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from collections import OrderedDict

try:
    from safetensors.torch import save_file
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False
    logging.warning(
        "safetensors not available, install with: pip install safetensors")


logger = logging.getLogger(__name__)


class SafeTensorsSaver:
    """
    Saves quantized models in SafeTensors format with proper sharding and index files.

    Output structure matches HuggingFace format:
    - model-00001-of-000XX.safetensors (sharded weight files)
    - model.safetensors.index.json (weight map)
    - config.json (model config)
    - generation_config.json
    - tokenizer files
    """

    def __init__(
        self,
        max_shard_size: str = "5GB",
        safe_serialization: bool = True
    ):
        """
        Args:
            max_shard_size: Maximum size per shard (e.g., "5GB", "10GB")
            safe_serialization: Use safetensors (recommended)
        """
        if not SAFETENSORS_AVAILABLE and safe_serialization:
            raise ImportError(
                "safetensors not available. Install with: pip install safetensors")

        self.max_shard_size = self._parse_size(max_shard_size)
        self.safe_serialization = safe_serialization

    def _parse_size(self, size_str: str) -> int:
        """Parse size string like '5GB' to bytes"""
        size_str = size_str.upper()
        if size_str.endswith('GB'):
            return int(size_str[:-2]) * 1024**3
        elif size_str.endswith('MB'):
            return int(size_str[:-2]) * 1024**2
        elif size_str.endswith('KB'):
            return int(size_str[:-2]) * 1024
        else:
            return int(size_str)

    def save_model(
        self,
        model: torch.nn.Module,
        output_dir: str,
        source_model_dir: Optional[str] = None,
        quantization_config: Optional[Dict] = None
    ) -> None:
        """
        Save model in SafeTensors format with proper sharding.

        Args:
            model: Quantized model to save
            output_dir: Output directory
            source_model_dir: Source model directory (to copy config files)
            quantization_config: Quantization configuration to save
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving quantized model to {output_path}")

        # Get model state dict
        state_dict = model.state_dict()

        # Shard the state dict
        shards, index = self._shard_state_dict(state_dict)

        # Save each shard
        num_shards = len(shards)
        logger.info(f"Saving {num_shards} shards...")

        for shard_idx, shard_dict in enumerate(shards):
            shard_filename = self._get_shard_filename(shard_idx, num_shards)
            shard_path = output_path / shard_filename

            logger.info(
                f"Saving shard {shard_idx+1}/{num_shards}: {shard_filename}")

            if self.safe_serialization:
                # Save as safetensors
                save_file(shard_dict, str(shard_path))
            else:
                # Save as PyTorch
                torch.save(shard_dict, str(shard_path))

        # Save index file
        self._save_index_file(index, output_path, num_shards)

        # Copy or create config files
        self._save_config_files(
            output_path, source_model_dir, quantization_config)

        logger.info(f"✓ Model saved successfully to {output_path}")

    def _shard_state_dict(
        self,
        state_dict: Dict[str, torch.Tensor]
    ) -> Tuple[List[Dict[str, torch.Tensor]], Dict[str, str]]:
        """
        Shard state dict into multiple files based on max_shard_size.

        Returns:
            shards: List of weight dictionaries
            index: Weight map (weight_name -> shard_filename)
        """
        shards = []
        current_shard = OrderedDict()
        current_size = 0
        weight_map = {}
        shard_idx = 0

        # Sort by size to pack efficiently
        sorted_items = sorted(
            state_dict.items(),
            key=lambda x: x[1].numel() * x[1].element_size(),
            reverse=True
        )

        for name, tensor in sorted_items:
            tensor_size = tensor.numel() * tensor.element_size()

            # Check if we need a new shard
            if current_size > 0 and current_size + tensor_size > self.max_shard_size:
                shards.append(current_shard)
                current_shard = OrderedDict()
                current_size = 0
                shard_idx += 1

            # Add to current shard
            current_shard[name] = tensor.contiguous()
            current_size += tensor_size

            # Update weight map
            num_shards = len(shards) + 1
            shard_filename = self._get_shard_filename(shard_idx, num_shards)
            weight_map[name] = shard_filename

        # Add last shard
        if current_shard:
            shards.append(current_shard)

        # Update all weight maps with final shard count
        num_shards = len(shards)
        weight_map = {
            name: self._get_shard_filename(
                int(filename.split('-')[1].split('of')[0]) - 1,
                num_shards
            )
            for name, filename in weight_map.items()
        }

        return shards, weight_map

    def _get_shard_filename(self, shard_idx: int, num_shards: int) -> str:
        """Generate shard filename like 'model-00001-of-00016.safetensors'"""
        num_digits = len(str(num_shards))
        shard_name = f"model-{shard_idx+1:0{num_digits}d}-of-{num_shards:0{num_digits}d}"

        if self.safe_serialization:
            return f"{shard_name}.safetensors"
        else:
            return f"{shard_name}.bin"

    def _save_index_file(
        self,
        weight_map: Dict[str, str],
        output_path: Path,
        num_shards: int
    ) -> None:
        """Save model.safetensors.index.json"""

        # Calculate metadata
        total_size = 0
        for name, shard_file in weight_map.items():
            # This is approximate, actual size needs to be computed from tensors
            pass

        index_data = {
            "metadata": {
                "total_size": total_size
            },
            "weight_map": weight_map
        }

        if self.safe_serialization:
            index_filename = "model.safetensors.index.json"
        else:
            index_filename = "pytorch_model.bin.index.json"

        index_path = output_path / index_filename
        with open(index_path, 'w', encoding='utf-8') as f:
            json.dump(index_data, f, indent=2)

        logger.info(f"Saved index file: {index_filename}")

    def _save_config_files(
        self,
        output_path: Path,
        source_model_dir: Optional[str],
        quantization_config: Optional[Dict]
    ) -> None:
        """
        Save or copy config files from source model.

        Files to handle:
        - config.json
        - generation_config.json
        - tokenizer.json
        - tokenizer_config.json
        - vocab.json
        - merges.txt
        - special_tokens_map.json
        """

        if source_model_dir:
            source_path = Path(source_model_dir)

            # List of files to copy
            files_to_copy = [
                "config.json",
                "configuration.json",  # Some models use this
                "generation_config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "vocab.json",
                "merges.txt",
                "special_tokens_map.json",
                "tokenizer_model.json",
                "added_tokens.json"
            ]

            for filename in files_to_copy:
                source_file = source_path / filename
                if source_file.exists():
                    dest_file = output_path / filename
                    shutil.copy2(source_file, dest_file)
                    logger.info(f"Copied {filename}")

        # Save quantization config
        if quantization_config:
            quant_config_path = output_path / "quantization_config.json"
            with open(quant_config_path, 'w', encoding='utf-8') as f:
                json.dump(quantization_config, f, indent=2)
            logger.info("Saved quantization_config.json")

            # Also update config.json with quantization info
            config_path = output_path / "config.json"
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)

                # Add quantization info
                config["quantization_config"] = quantization_config

                with open(config_path, 'w', encoding='utf-8') as f:
                    json.dump(config, f, indent=2)

                logger.info("Updated config.json with quantization info")


def save_quantized_model_safetensors(
    model: torch.nn.Module,
    output_dir: str,
    source_model_dir: Optional[str] = None,
    quantization_config: Optional[Dict] = None,
    max_shard_size: str = "5GB"
) -> None:
    """
    Convenience function to save model in SafeTensors format.

    Args:
        model: Quantized model
        output_dir: Output directory
        source_model_dir: Source model directory (for config files)
        quantization_config: Quantization configuration
        max_shard_size: Maximum size per shard file

    Example:
        >>> save_quantized_model_safetensors(
        ...     model=quantized_model,
        ...     output_dir="/dev/shm/Qwen3-30B-A3B-W2A2",
        ...     source_model_dir="/dev/shm/Qwen3-30B-A3B",
        ...     quantization_config={
        ...         "expert_precision": "W2A2",
        ...         "router_precision": "W8A8",
        ...         "method": "mixed_precision_ptq"
        ...     }
        ... )
    """
    saver = SafeTensorsSaver(
        max_shard_size=max_shard_size,
        safe_serialization=True
    )

    saver.save_model(
        model=model,
        output_dir=output_dir,
        source_model_dir=source_model_dir,
        quantization_config=quantization_config
    )


def save_quantized_model_pytorch(
    model: torch.nn.Module,
    output_dir: str,
    source_model_dir: Optional[str] = None,
    quantization_config: Optional[Dict] = None,
    max_shard_size: str = "5GB"
) -> None:
    """
    Save model in PyTorch format (fallback if safetensors unavailable).

    Args:
        model: Quantized model
        output_dir: Output directory
        source_model_dir: Source model directory (for config files)
        quantization_config: Quantization configuration
        max_shard_size: Maximum size per shard file
    """
    saver = SafeTensorsSaver(
        max_shard_size=max_shard_size,
        safe_serialization=False
    )

    saver.save_model(
        model=model,
        output_dir=output_dir,
        source_model_dir=source_model_dir,
        quantization_config=quantization_config
    )
