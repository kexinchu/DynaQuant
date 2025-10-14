#!/usr/bin/env python3
"""
Quick test runner for DynaQuant modules.
"""

import logging
import sys

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def run_tests():
    """Run all module tests."""
    test_results = {}

    tests = [
        ("pack", "dynaquant.pack", "test_pack_unpack"),
        ("fake_quant", "dynaquant.fake_quant", "test_fake_quant"),
        ("router_guard", "dynaquant.router_guard", "test_router_guard"),
        ("precision_sched", "dynaquant.precision_sched", "test_precision_scheduler"),
        ("expert_cache", "dynaquant.expert_cache", "test_expert_cache"),
        ("moe_linear", "dynaquant.moe_linear", "test_moe_linear"),
        ("moe_wrapper", "dynaquant.moe_wrapper", "test_moe_wrapper"),
        ("hooks", "dynaquant.hooks", "test_hooks"),
    ]

    print("="*80)
    print("Running DynaQuant Tests")
    print("="*80)

    for name, module_name, test_func_name in tests:
        print(f"\nTesting {name}...", end=" ", flush=True)
        try:
            module = __import__(module_name, fromlist=[test_func_name])
            test_func = getattr(module, test_func_name)
            test_func()
            print("✅ PASS")
            test_results[name] = "PASS"
        except Exception as e:
            print(f"❌ FAIL: {e}")
            test_results[name] = "FAIL"

    # Summary
    print("\n" + "="*80)
    print("Test Summary")
    print("="*80)

    passed = sum(1 for r in test_results.values() if r == "PASS")
    failed = len(test_results) - passed

    for name, result in test_results.items():
        symbol = "✅" if result == "PASS" else "❌"
        print(f"{symbol} {name}: {result}")

    print("="*80)
    print(f"Total: {len(test_results)}, Passed: {passed}, Failed: {failed}")
    print("="*80)

    if failed == 0:
        print("\n🎉 All tests passed!")
        return True
    else:
        print(f"\n⚠️  {failed} test(s) failed")
        return False


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
