#!/usr/bin/env python3
"""
Simple test script for MoE-Quant components

This script tests the core components without requiring actual model loading.
"""

import torch
import sys
from pathlib import Path

# Add moe_quant to path
sys.path.insert(0, str(Path(__file__).parent))


def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    try:
        from moe_quant.quant.ebss import EBSSSampler, EBSSConfig
        from moe_quant.quant.agq import AGQuantizer, AGQConfig
        from moe_quant.quant.quantizers import W2A2Quantizer, W2A2Config
        from moe_quant.quant.router_guard_enhanced import EnhancedRouterGuard
        from moe_quant.losses.routing_losses import topk_consistency_loss, margin_loss
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def test_agq_quantizer():
    """Test AGQ quantizer on dummy data"""
    print("\nTesting AGQ quantizer...")
    try:
        from moe_quant.quant.agq import AGQuantizer, AGQConfig
        import torch.nn as nn

        config = AGQConfig(bit_width=2, group_size=64)
        quantizer = AGQuantizer(config)

        # Create dummy layer and data
        layer = nn.Linear(128, 128)
        X = torch.randn(32, 128)  # [batch, features]
        c = torch.rand(32)         # [batch] affinities

        # Quantize
        W_quant, scales, stats = quantizer.quantize_linear(layer, X, c)

        assert W_quant.shape == layer.weight.shape
        print(f"✓ AGQ quantization successful")
        print(f"  MSE: {stats['mse']:.6f}")
        print(f"  Weighted MSE: {stats['weighted_mse']:.6f}")
        return True

    except Exception as e:
        print(f"✗ AGQ test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_w2a2_quantizer():
    """Test W2A2 quantizer"""
    print("\nTesting W2A2 quantizer...")
    try:
        from moe_quant.quant.quantizers import W2A2Quantizer, W2A2Config
        import torch.nn as nn

        config = W2A2Config(
            w_bit=2,
            a_bit=2,
            use_rotation=True,
            use_whitening=True
        )
        quantizer = W2A2Quantizer(config)

        # Create dummy layer and calibration data
        layer = nn.Linear(128, 128)
        X_calib = torch.randn(64, 128)  # [samples, features]

        # Quantize
        W_quant, W_absorbed, stats = quantizer.quantize_linear_layer(
            layer, X_calib, layer_id=0
        )

        assert W_quant.shape == layer.weight.shape
        assert W_absorbed.shape == layer.weight.shape
        print(f"✓ W2A2 quantization successful")
        print(f"  MSE: {stats['mse']:.6f}")
        print(f"  Relative Error: {stats['relative_error']:.4f}")
        return True

    except Exception as e:
        print(f"✗ W2A2 test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_router_guard():
    """Test Enhanced Router Guard"""
    print("\nTesting Enhanced Router Guard...")
    try:
        from moe_quant.quant.router_guard_enhanced import EnhancedRouterGuard, EnhancedRouterConfig

        config = EnhancedRouterConfig(mode="fp16", top_k=2)
        guard = EnhancedRouterGuard(config)

        # Create dummy data
        batch, seq_len, hidden_dim = 2, 10, 128
        num_experts = 8

        x = torch.randn(batch, seq_len, hidden_dim)
        router_weight = torch.randn(num_experts, hidden_dim)
        router_bias = torch.randn(num_experts)

        # Forward
        logits, expert_ids = guard.forward_router_fp16(
            x, router_weight, router_bias)

        assert logits.shape == (batch, seq_len, num_experts)
        assert expert_ids.shape == (batch, seq_len, 2)

        # Test consistency check
        logits2, expert_ids2 = guard.forward_router_fp16(
            x, router_weight, router_bias)
        consistency = guard.check_consistency(
            expert_ids, expert_ids2, layer_id=0)

        print(f"✓ Router guard successful")
        print(f"  Exact match rate: {consistency['exact_match_rate']:.2%}")
        return True

    except Exception as e:
        print(f"✗ Router guard test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_routing_losses():
    """Test routing loss functions"""
    print("\nTesting routing losses...")
    try:
        from moe_quant.losses.routing_losses import (
            topk_consistency_loss, margin_loss, combined_routing_loss
        )

        batch, seq_len, num_experts = 2, 10, 8
        logits_fp = torch.randn(batch, seq_len, num_experts)
        logits_quant = logits_fp + torch.randn_like(logits_fp) * 0.1

        # Test losses
        consistency = topk_consistency_loss(logits_quant, logits_fp, k=2)
        margin = margin_loss(logits_quant, k=2)
        combined, loss_dict = combined_routing_loss(
            logits_quant, logits_fp, k=2)

        print(f"✓ Routing losses successful")
        print(f"  Consistency loss: {loss_dict['consistency']:.4f}")
        print(f"  Margin loss: {loss_dict['margin']:.4f}")
        print(f"  Total loss: {loss_dict['total']:.4f}")
        return True

    except Exception as e:
        print(f"✗ Routing losses test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("="*50)
    print("MoE-Quant Component Tests")
    print("="*50)

    results = []

    # Run tests
    results.append(("Imports", test_imports()))
    results.append(("AGQ Quantizer", test_agq_quantizer()))
    results.append(("W2A2 Quantizer", test_w2a2_quantizer()))
    results.append(("Router Guard", test_router_guard()))
    results.append(("Routing Losses", test_routing_losses()))

    # Summary
    print("\n" + "="*50)
    print("Test Summary")
    print("="*50)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
