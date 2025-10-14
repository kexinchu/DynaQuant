"""
SGLang integration hooks for injecting DynaQuant into model serving.
Provides hooks to replace MoE blocks with quantized versions.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, List, Tuple, Callable
import logging

from . import moe_wrapper

logger = logging.getLogger(__name__)


class SGLangHookManager:
    """
    Manager for SGLang hooks to inject DynaQuant quantization.
    """

    def __init__(
        self,
        model: nn.Module,
        config: Dict,
        enable_rcg: bool = True,
        enable_ps: bool = True,
        enable_ec: bool = True,
    ):
        """
        Initialize hook manager.

        Args:
            model: Model to inject hooks into
            config: Configuration dictionary
            enable_rcg: Enable Router-Consistency Guard
            enable_ps: Enable Precision Scheduler
            enable_ec: Enable Expert Cache
        """
        self.model = model
        self.config = config
        self.enable_rcg = enable_rcg
        self.enable_ps = enable_ps
        self.enable_ec = enable_ec

        self.wrapped_blocks = {}
        self.hooks = []

    def find_moe_blocks(self) -> List[Tuple[str, nn.Module]]:
        """
        Find all MoE blocks in the model.

        Returns:
            List of (name, module) tuples for MoE blocks
        """
        moe_blocks = []

        for name, module in self.model.named_modules():
            # Check if module is an MoE block
            # This is model-specific; adapt for your model architecture

            # Check if module has 'experts' attribute (common MoE pattern)
            if hasattr(module, 'experts') and hasattr(module, 'gate'):
                moe_blocks.append((name, module))
            # For Qwen3:
            elif 'mlp' in name.lower() and hasattr(module, 'experts'):
                moe_blocks.append((name, module))

            # For other models, add detection logic here
            # Example: if isinstance(module, MoELayer):

        return moe_blocks

    def wrap_moe_block(
        self,
        name: str,
        block: nn.Module,
        num_experts: int,
        hidden_dim: int,
        ffn_dim: int,
        top_k: int = 2,
    ) -> moe_wrapper.DynaQuantMoEWrapper:
        """
        Wrap an MoE block with DynaQuant quantization.

        Args:
            name: Block name
            block: Original MoE block
            num_experts: Number of experts
            hidden_dim: Hidden dimension
            ffn_dim: FFN dimension
            top_k: Top-k routing

        Returns:
            wrapper: DynaQuant MoE wrapper
        """
        logger.info(f"Wrapping MoE block: {name}")

        # Get configuration
        rcg_config = self.config.get('router_guard', {})
        ps_config = self.config.get('precision_scheduler', {})
        ec_config = self.config.get('expert_cache', {})
        group_size = self.config.get('quantization', {}).get(
            'weight', {}).get('group_size', 128)

        # Create wrapper
        wrapper = moe_wrapper.DynaQuantMoEWrapper(
            original_moe_block=block,
            num_experts=num_experts,
            hidden_dim=hidden_dim,
            ffn_dim=ffn_dim,
            top_k=top_k,
            enable_rcg=self.enable_rcg,
            enable_ps=self.enable_ps,
            enable_ec=self.enable_ec,
            rcg_config=rcg_config,
            ps_config=ps_config,
            ec_config=ec_config,
            group_size=group_size,
        )

        # Load expert weights
        for expert_id in range(num_experts):
            wrapper.load_expert_weights_from_original(expert_id)

        self.wrapped_blocks[name] = wrapper

        return wrapper

    def inject_hooks(self):
        """
        Inject hooks into the model to replace MoE blocks.
        """
        moe_blocks = self.find_moe_blocks()

        if not moe_blocks:
            logger.warning("No MoE blocks found in model")
            return

        logger.info(f"Found {len(moe_blocks)} MoE blocks")

        for name, block in moe_blocks:
            # Infer block parameters
            # This is model-specific
            try:
                if hasattr(block, 'experts'):
                    num_experts = len(block.experts)

                    # Get dimensions from first expert
                    if hasattr(block.experts[0], 'up_proj'):
                        hidden_dim = block.experts[0].up_proj.in_features
                        ffn_dim = block.experts[0].up_proj.out_features
                    else:
                        logger.warning(
                            f"Cannot infer dimensions for block {name}, skipping")
                        continue

                    # Get top_k
                    if hasattr(block, 'top_k'):
                        top_k = block.top_k
                    else:
                        top_k = 2  # Default

                    # Wrap block
                    wrapper = self.wrap_moe_block(
                        name, block, num_experts, hidden_dim, ffn_dim, top_k
                    )

                    # Replace in model
                    # Navigate to parent and replace
                    parts = name.split('.')
                    parent = self.model
                    for part in parts[:-1]:
                        parent = getattr(parent, part)
                    setattr(parent, parts[-1], wrapper)

                    logger.info(f"Replaced {name} with DynaQuant wrapper")

            except Exception as e:
                logger.error(f"Failed to wrap block {name}: {e}")
                continue

    def remove_hooks(self):
        """Remove all injected hooks."""
        # Restore original blocks
        for name, wrapper in self.wrapped_blocks.items():
            parts = name.split('.')
            parent = self.model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], wrapper.original_moe_block)

        self.wrapped_blocks.clear()
        logger.info("Removed all DynaQuant hooks")

    def get_statistics(self) -> Dict:
        """Get statistics from all wrapped blocks."""
        stats = {}

        for name, wrapper in self.wrapped_blocks.items():
            stats[name] = wrapper.get_statistics()

        return stats


def inject_dynaquant_into_sglang(
    model: nn.Module,
    config: Dict,
    enable_rcg: bool = True,
    enable_ps: bool = True,
    enable_ec: bool = True,
) -> SGLangHookManager:
    """
    Inject DynaQuant into an SGLang model.

    Args:
        model: SGLang model
        config: DynaQuant configuration
        enable_rcg: Enable Router-Consistency Guard
        enable_ps: Enable Precision Scheduler
        enable_ec: Enable Expert Cache

    Returns:
        hook_manager: Hook manager for controlling DynaQuant
    """
    logger.info("Injecting DynaQuant into SGLang model")

    hook_manager = SGLangHookManager(
        model=model,
        config=config,
        enable_rcg=enable_rcg,
        enable_ps=enable_ps,
        enable_ec=enable_ec,
    )

    hook_manager.inject_hooks()

    logger.info("DynaQuant injection complete")

    return hook_manager


def test_hooks():
    """
    Unit tests for SGLang hooks.
    """
    import logging
    logger = logging.getLogger(__name__)

    logger.info("Testing SGLang hooks...")

    # Create a dummy model with MoE blocks
    logger.info("\n--- Creating dummy model ---")

    class DummyExpert(nn.Module):
        def __init__(self, hidden_dim, ffn_dim):
            super().__init__()
            self.up_proj = nn.Linear(hidden_dim, ffn_dim, bias=False)
            self.gate_proj = nn.Linear(hidden_dim, ffn_dim, bias=False)
            self.down_proj = nn.Linear(ffn_dim, hidden_dim, bias=False)

    class DummyMoEBlock(nn.Module):
        def __init__(self, num_experts, hidden_dim, ffn_dim):
            super().__init__()
            self.experts = nn.ModuleList([
                DummyExpert(hidden_dim, ffn_dim) for _ in range(num_experts)
            ])
            self.gate = nn.Linear(hidden_dim, num_experts, bias=False)
            self.top_k = 2

    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = DummyMoEBlock(8, 512, 1024)
            self.layer2 = DummyMoEBlock(8, 512, 1024)

    model = DummyModel()
    logger.info(f"Created dummy model with 2 MoE blocks")

    # Create configuration
    logger.info("\n--- Creating configuration ---")
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        config = {
            'quantization': {
                'weight': {
                    'group_size': 128,
                }
            },
            'router_guard': {
                'temperature': 1.0,
                'clip_range': 10.0,
            },
            'precision_scheduler': {
                'vram_budget_gb': 80.0,
                'top_m_experts': 4,
            },
            'expert_cache': {
                'cache_dir': tmpdir,
                'warm_pool_size': 4,
                'async_swap': False,
            },
        }

        logger.info(f"Configuration: {config}")

        # Test hook manager
        logger.info("\n--- Testing hook manager ---")
        hook_manager = SGLangHookManager(
            model=model,
            config=config,
            enable_rcg=True,
            enable_ps=True,
            enable_ec=True,
        )

        # Find MoE blocks
        moe_blocks = hook_manager.find_moe_blocks()
        logger.info(f"Found {len(moe_blocks)} MoE blocks:")
        for name, _ in moe_blocks:
            logger.info(f"  - {name}")

        assert len(
            moe_blocks) == 2, f"Expected 2 MoE blocks, found {len(moe_blocks)}"
        logger.info(f"✓ MoE block detection test passed")

        # Inject hooks
        logger.info("\n--- Testing hook injection ---")
        hook_manager.inject_hooks()

        # Check that blocks are replaced
        assert isinstance(model.layer1, moe_wrapper.DynaQuantMoEWrapper)
        assert isinstance(model.layer2, moe_wrapper.DynaQuantMoEWrapper)

        logger.info(f"✓ Hook injection test passed")

        # Test forward pass
        logger.info("\n--- Testing forward pass ---")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)

        hidden_states = torch.randn(2, 8, 512, device=device)

        model.eval()
        with torch.no_grad():
            output1, _ = model.layer1(hidden_states)
            output2, _ = model.layer2(output1)

        logger.info(f"Input shape: {hidden_states.shape}")
        logger.info(f"Layer1 output shape: {output1.shape}")
        logger.info(f"Layer2 output shape: {output2.shape}")

        logger.info(f"✓ Forward pass test passed")

        # Test statistics
        logger.info("\n--- Testing statistics ---")
        stats = hook_manager.get_statistics()

        logger.info(f"Number of wrapped blocks: {len(stats)}")
        for block_name, block_stats in stats.items():
            logger.info(f"  {block_name}:")
            logger.info(f"    Forward count: {block_stats['forward_count']}")

        logger.info(f"✓ Statistics test passed")

        # Test hook removal
        logger.info("\n--- Testing hook removal ---")
        hook_manager.remove_hooks()

        assert isinstance(model.layer1, DummyMoEBlock)
        assert isinstance(model.layer2, DummyMoEBlock)

        logger.info(f"✓ Hook removal test passed")

        # Test inject_dynaquant_into_sglang function
        logger.info("\n--- Testing inject_dynaquant_into_sglang ---")
        model = DummyModel()  # Fresh model

        hook_manager2 = inject_dynaquant_into_sglang(
            model=model,
            config=config,
            enable_rcg=True,
            enable_ps=True,
            enable_ec=True,
        )

        assert isinstance(model.layer1, moe_wrapper.DynaQuantMoEWrapper)
        assert isinstance(model.layer2, moe_wrapper.DynaQuantMoEWrapper)

        logger.info(f"✓ inject_dynaquant_into_sglang test passed")

    logger.info("\n✓ All SGLang hooks tests passed!")
    return True


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    test_hooks()
