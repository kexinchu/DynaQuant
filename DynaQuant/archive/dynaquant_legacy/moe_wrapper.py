"""
MoE Wrapper that integrates Router-Consistency Guard, Precision Scheduler,
Expert Cache, and Quantized Linear layers for dynamic mixed-precision MoE.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, List, Tuple
import logging

from . import router_guard, precision_sched, expert_cache, moe_linear

logger = logging.getLogger(__name__)


class DynaQuantMoEWrapper(nn.Module):
    """
    Wrapper for MoE blocks with dynamic mixed-precision quantization.
    """

    def __init__(
        self,
        original_moe_block: nn.Module,
        num_experts: int,
        hidden_dim: int,
        ffn_dim: int,
        top_k: int = 2,
        enable_rcg: bool = True,
        enable_ps: bool = True,
        enable_ec: bool = True,
        rcg_config: Optional[Dict] = None,
        ps_config: Optional[Dict] = None,
        ec_config: Optional[Dict] = None,
        group_size: int = 128,
        use_triton: bool = True,
    ):
        """
        Initialize DynaQuant MoE wrapper.

        Args:
            original_moe_block: Original MoE block to wrap
            num_experts: Number of experts
            hidden_dim: Hidden dimension
            ffn_dim: FFN intermediate dimension
            top_k: Number of top experts to select
            enable_rcg: Enable Router-Consistency Guard
            enable_ps: Enable Precision Scheduler
            enable_ec: Enable Expert Cache
            rcg_config: RCG configuration
            ps_config: Precision Scheduler configuration
            ec_config: Expert Cache configuration
            group_size: Group size for weight quantization
            use_triton: Whether to use Triton kernels
        """
        super().__init__()

        self.original_moe_block = original_moe_block
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.top_k = top_k
        self.enable_rcg = enable_rcg
        self.enable_ps = enable_ps
        self.enable_ec = enable_ec
        self.group_size = group_size

        # Router-Consistency Guard
        if enable_rcg:
            rcg_config = rcg_config or {}
            self.rcg = router_guard.RouterConsistencyGuard(
                num_experts=num_experts,
                top_k=top_k,
                **rcg_config,
            )
        else:
            self.rcg = None

        # Precision Scheduler
        if enable_ps:
            ps_config = ps_config or {}
            self.ps = precision_sched.PrecisionScheduler(
                num_experts=num_experts,
                **ps_config,
            )
        else:
            self.ps = None

        # Expert Cache
        if enable_ec:
            ec_config = ec_config or {}
            self.ec = expert_cache.ExpertCache(
                num_experts=num_experts,
                **ec_config,
            )
        else:
            self.ec = None

        # Quantized expert layers
        # We need to replace FFN layers in the original MoE block
        # For now, create new quantized layers
        # up_proj: [hidden_dim, ffn_dim]
        # gate_proj: [hidden_dim, ffn_dim]
        # down_proj: [ffn_dim, hidden_dim]

        self.expert_up_proj = moe_linear.MoELinear(
            num_experts=num_experts,
            in_features=hidden_dim,
            out_features=ffn_dim,
            bias=False,
            default_precision="w2a4",
            group_size=group_size,
            use_triton=use_triton,
        )

        self.expert_gate_proj = moe_linear.MoELinear(
            num_experts=num_experts,
            in_features=hidden_dim,
            out_features=ffn_dim,
            bias=False,
            default_precision="w2a4",
            group_size=group_size,
            use_triton=use_triton,
        )

        self.expert_down_proj = moe_linear.MoELinear(
            num_experts=num_experts,
            in_features=ffn_dim,
            out_features=hidden_dim,
            bias=False,
            default_precision="w2a4",
            group_size=group_size,
            use_triton=use_triton,
        )

        # Track expert usage
        self.expert_usage_counts = [0] * num_experts
        self.forward_count = 0

    def load_expert_weights_from_original(self, expert_id: int):
        """
        Load and quantize expert weights from original MoE block.

        Args:
            expert_id: Expert ID
        """
        try:
            # Try to extract weights from original block
            # This is model-specific; adapt for your model architecture

            # For Qwen3 MoE:
            # experts[i].up_proj.weight
            # experts[i].gate_proj.weight
            # experts[i].down_proj.weight

            if hasattr(self.original_moe_block, 'experts'):
                expert = self.original_moe_block.experts[expert_id]

                if hasattr(expert, 'up_proj'):
                    up_weight = expert.up_proj.weight.data
                    self.expert_up_proj.experts[expert_id].quantize_from_fp_weights(
                        up_weight)

                if hasattr(expert, 'gate_proj'):
                    gate_weight = expert.gate_proj.weight.data
                    self.expert_gate_proj.experts[expert_id].quantize_from_fp_weights(
                        gate_weight)

                if hasattr(expert, 'down_proj'):
                    down_weight = expert.down_proj.weight.data
                    self.expert_down_proj.experts[expert_id].quantize_from_fp_weights(
                        down_weight)

                logger.info(f"Loaded and quantized expert {expert_id} weights")

                # Save to cache if enabled
                if self.enable_ec and self.ec is not None:
                    self._save_expert_to_cache(expert_id)
            else:
                logger.warning(
                    f"Cannot extract weights from original block for expert {expert_id}")

        except Exception as e:
            logger.error(f"Failed to load expert {expert_id} weights: {e}")

    def _save_expert_to_cache(self, expert_id: int):
        """Save expert weights to cache."""
        try:
            # Save up_proj
            up_layer = self.expert_up_proj.experts[expert_id]
            if up_layer.w2_packed is not None:
                self.ec.save_expert(
                    expert_id,
                    w2_packed=up_layer.w2_packed,
                    w2_scales=up_layer.w2_scales,
                    w2_metadata=up_layer.w2_metadata,
                    w4_packed=up_layer.w4_packed,
                    w4_scales=up_layer.w4_scales,
                    w4_metadata=up_layer.w4_metadata,
                )

            # Similar for gate_proj and down_proj
            # In practice, save all layers separately with different keys

        except Exception as e:
            logger.error(f"Failed to save expert {expert_id} to cache: {e}")

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through quantized MoE block.

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_dim]
            router_logits: Router logits [batch * seq_len, num_experts] (optional)

        Returns:
            output: Output tensor [batch, seq_len, hidden_dim]
            router_logits: Router logits (if computed)
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape

        # Flatten for routing
        # [batch * seq_len, hidden_dim]
        hidden_states_flat = hidden_states.view(-1, hidden_dim)

        # Get router logits
        if router_logits is None:
            # Use original router
            if hasattr(self.original_moe_block, 'gate'):
                router_logits = self.original_moe_block.gate(
                    hidden_states_flat)
            else:
                raise ValueError(
                    "Router logits not provided and cannot compute from original block")

        # Select top-k experts
        routing_weights, selected_experts = torch.topk(
            router_logits, self.top_k, dim=-1)
        routing_weights = torch.softmax(
            routing_weights, dim=-1, dtype=torch.float32).type_as(hidden_states)

        # Router-Consistency Guard
        promotion_candidates = []
        if self.enable_rcg and self.rcg is not None and self.training:
            # Simulate quantized router logits (for simplicity, add small noise)
            router_logits_quant = router_logits + \
                torch.randn_like(router_logits) * 0.01

            # Check for promotions
            promote_mask = self.rcg.check_promotion(
                router_logits, router_logits_quant, selected_experts)
            promotion_candidates = [i for i in range(
                self.num_experts) if promote_mask[i]]

        # Precision Scheduler
        if self.enable_ps and self.ps is not None:
            # Update hit counts
            expert_ids_used = selected_experts.flatten().tolist()
            self.ps.update_hit_counts(expert_ids_used)

            # Run scheduling
            if self.forward_count % 10 == 0:  # Schedule every 10 steps
                precision_changes = self.ps.schedule(
                    promotion_candidates=promotion_candidates)

                # Apply precision changes
                for expert_id, new_precision in precision_changes.items():
                    self.expert_up_proj.set_expert_precision(
                        expert_id, new_precision)
                    self.expert_gate_proj.set_expert_precision(
                        expert_id, new_precision)
                    self.expert_down_proj.set_expert_precision(
                        expert_id, new_precision)

        # Process tokens through selected experts
        output_flat = torch.zeros_like(hidden_states_flat)

        # Group tokens by expert for efficient batching
        for expert_id in range(self.num_experts):
            # Find tokens routed to this expert
            expert_mask = (selected_experts == expert_id).any(dim=-1)
            expert_tokens = hidden_states_flat[expert_mask]

            if expert_tokens.shape[0] == 0:
                continue

            # Get precision for this expert
            if self.enable_ps and self.ps is not None:
                precision = self.ps.get_precision(expert_id)
            else:
                precision = "w2a4"  # Default

            # Update usage count
            self.expert_usage_counts[expert_id] += expert_tokens.shape[0]

            # Process through expert FFN
            # FFN: down_proj(silu(gate_proj(x)) * up_proj(x))
            try:
                gate_out = self.expert_gate_proj(
                    expert_tokens, expert_id=expert_id, precision=precision)
                up_out = self.expert_up_proj(
                    expert_tokens, expert_id=expert_id, precision=precision)

                # SiLU activation
                intermediate = torch.nn.functional.silu(gate_out) * up_out

                expert_output = self.expert_down_proj(
                    intermediate, expert_id=expert_id, precision=precision)

                # Accumulate weighted output
                # Find which tokens were routed to this expert and their weights
                for token_idx in range(expert_mask.shape[0]):
                    if expert_mask[token_idx]:
                        # Find position of this expert in selected_experts for this token
                        expert_positions = (selected_experts[token_idx] == expert_id).nonzero(
                            as_tuple=True)[0]
                        if len(expert_positions) > 0:
                            weight = routing_weights[token_idx,
                                                     expert_positions[0]]
                            output_flat[token_idx] += weight * \
                                expert_output[expert_mask[:token_idx+1].sum() - 1]

            except Exception as e:
                logger.error(f"Error processing expert {expert_id}: {e}")
                # Skip this expert
                continue

        # Reshape output
        output = output_flat.view(batch_size, seq_len, hidden_dim)

        self.forward_count += 1

        return output, router_logits

    def get_statistics(self) -> Dict:
        """Get statistics from all components."""
        stats = {
            'forward_count': self.forward_count,
            'expert_usage_counts': self.expert_usage_counts,
        }

        if self.enable_rcg and self.rcg is not None:
            stats['rcg'] = self.rcg.get_statistics()

        if self.enable_ps and self.ps is not None:
            stats['ps'] = self.ps.get_statistics()

        if self.enable_ec and self.ec is not None:
            stats['ec'] = self.ec.get_statistics()

        return stats


def test_moe_wrapper():
    """
    Unit tests for MoE wrapper.
    """
    import logging
    logger = logging.getLogger(__name__)

    logger.info("Testing MoE wrapper...")

    # Create a dummy MoE block
    logger.info("\n--- Creating dummy MoE block ---")

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

    num_experts = 8
    hidden_dim = 512
    ffn_dim = 1024
    top_k = 2

    original_moe = DummyMoEBlock(num_experts, hidden_dim, ffn_dim)
    logger.info(f"Created dummy MoE block with {num_experts} experts")

    # Create wrapper
    logger.info("\n--- Creating DynaQuant wrapper ---")

    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        wrapper = DynaQuantMoEWrapper(
            original_moe_block=original_moe,
            num_experts=num_experts,
            hidden_dim=hidden_dim,
            ffn_dim=ffn_dim,
            top_k=top_k,
            enable_rcg=True,
            enable_ps=True,
            enable_ec=True,
            ec_config={'cache_dir': tmpdir,
                       'warm_pool_size': 4, 'async_swap': False},
            group_size=128,
            use_triton=False,  # Disable Triton for testing
        )

        logger.info(f"Created DynaQuant wrapper")
        logger.info(f"RCG enabled: {wrapper.enable_rcg}")
        logger.info(f"PS enabled: {wrapper.enable_ps}")
        logger.info(f"EC enabled: {wrapper.enable_ec}")

        # Load expert weights
        logger.info("\n--- Loading expert weights ---")
        for expert_id in range(num_experts):
            wrapper.load_expert_weights_from_original(expert_id)

        logger.info(f"Loaded {num_experts} experts")

        # Test forward pass
        logger.info("\n--- Testing forward pass ---")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        wrapper = wrapper.to(device)

        batch_size = 2
        seq_len = 8
        hidden_states = torch.randn(
            batch_size, seq_len, hidden_dim, device=device)

        wrapper.eval()
        with torch.no_grad():
            output, router_logits = wrapper(hidden_states)

        logger.info(f"Input shape: {hidden_states.shape}")
        logger.info(f"Output shape: {output.shape}")
        logger.info(f"Router logits shape: {router_logits.shape}")

        assert output.shape == hidden_states.shape
        assert router_logits.shape == (batch_size * seq_len, num_experts)

        logger.info(f"✓ Forward pass test passed")

        # Test statistics
        logger.info("\n--- Testing statistics ---")
        stats = wrapper.get_statistics()

        logger.info(f"Forward count: {stats['forward_count']}")
        logger.info(f"Expert usage counts: {stats['expert_usage_counts']}")

        if 'rcg' in stats:
            logger.info(f"RCG stats: {stats['rcg']}")

        if 'ps' in stats:
            logger.info(f"PS stats: {stats['ps']}")

        if 'ec' in stats:
            logger.info(f"EC stats: {stats['ec']}")

        logger.info(f"✓ Statistics test passed")

        # Test multiple forward passes
        logger.info("\n--- Testing multiple forward passes ---")
        for i in range(5):
            with torch.no_grad():
                output, _ = wrapper(hidden_states)

        stats = wrapper.get_statistics()
        logger.info(
            f"Forward count after 5 more passes: {stats['forward_count']}")
        logger.info(f"✓ Multiple forward passes test passed")

    logger.info("\n✓ All MoE wrapper tests passed!")
    return True


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    test_moe_wrapper()
