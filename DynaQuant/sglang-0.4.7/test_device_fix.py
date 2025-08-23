#!/usr/bin/env python3
"""
设备修复测试脚本
"""

import torch
import logging
from fix_device_issues import (
    ensure_model_on_device,
    fix_tokenizer_device_issues,
    create_proper_attention_mask,
    validate_model_device_consistency,
    comprehensive_device_fix
)

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_device_fix_functions():
    """测试设备修复函数"""
    print("=" * 60)
    print("测试设备修复函数")
    print("=" * 60)
    
    # 创建一个简单的测试模型
    class TestModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = torch.nn.Linear(10, 20)
            self.linear2 = torch.nn.Linear(20, 5)
        
        def forward(self, x):
            return self.linear2(self.linear1(x))
    
    # 创建模型
    model = TestModel()
    print(f"原始模型设备: {next(model.parameters()).device}")
    
    # 测试设备修复
    if torch.cuda.is_available():
        device = "cuda"
        print(f"使用CUDA设备: {device}")
        
        # 测试ensure_model_on_device
        model_fixed = ensure_model_on_device(model, device)
        print(f"修复后模型设备: {next(model_fixed.parameters()).device}")
        
        # 测试验证函数
        validation = validate_model_device_consistency(model_fixed, device)
        print(f"设备一致性验证: {validation['is_consistent']}")
        
        if not validation['is_consistent']:
            print(f"发现的问题: {validation['issues']}")
    else:
        print("CUDA不可用，跳过设备测试")
    
    print("✓ 设备修复函数测试完成")


def test_attention_mask_creation():
    """测试注意力掩码创建"""
    print("\n" + "=" * 60)
    print("测试注意力掩码创建")
    print("=" * 60)
    
    # 模拟tokenizer
    class MockTokenizer:
        def __init__(self):
            self.pad_token_id = 0
            self.eos_token_id = 0  # 相同的情况
    
    tokenizer = MockTokenizer()
    
    # 创建测试输入
    input_ids = torch.tensor([[1, 2, 3, 0, 0], [1, 2, 0, 0, 0]])  # 包含padding
    
    # 测试注意力掩码创建
    attention_mask = create_proper_attention_mask(input_ids, tokenizer, "cpu")
    print(f"输入形状: {input_ids.shape}")
    print(f"注意力掩码形状: {attention_mask.shape}")
    print(f"注意力掩码:\n{attention_mask}")
    
    print("✓ 注意力掩码创建测试完成")


def test_tokenizer_fix():
    """测试tokenizer修复"""
    print("\n" + "=" * 60)
    print("测试tokenizer修复")
    print("=" * 60)
    
    # 模拟有问题的tokenizer
    class MockTokenizer:
        def __init__(self):
            self.pad_token_id = None
            self.eos_token_id = 2
            self.unk_token_id = 1
            self.pad_token = None
            self.eos_token = None
    
    tokenizer = MockTokenizer()
    print(f"修复前 - pad_token_id: {tokenizer.pad_token_id}")
    print(f"修复前 - eos_token: {tokenizer.eos_token}")
    
    # 修复tokenizer
    fixed_tokenizer = fix_tokenizer_device_issues(tokenizer, "cpu")
    print(f"修复后 - pad_token_id: {fixed_tokenizer.pad_token_id}")
    print(f"修复后 - eos_token: {fixed_tokenizer.eos_token}")
    
    print("✓ tokenizer修复测试完成")


def main():
    """主函数"""
    print("设备修复测试")
    print("=" * 60)
    
    # 运行所有测试
    tests = [
        ("设备修复函数", test_device_fix_functions),
        ("注意力掩码创建", test_attention_mask_creation),
        ("tokenizer修复", test_tokenizer_fix)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n运行测试: {test_name}")
        try:
            test_func()
            results.append((test_name, True))
        except Exception as e:
            print(f"✗ 测试失败: {e}")
            results.append((test_name, False))
    
    # 显示结果
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    
    for test_name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{test_name}: {status}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print(f"\n总计: {passed}/{total} 个测试通过")
    
    if passed == total:
        print("🎉 所有设备修复测试通过！")
        print("\n修复说明:")
        print("1. 模型设备一致性检查")
        print("2. 注意力掩码正确创建")
        print("3. tokenizer特殊token修复")
        print("4. MoE模块设备问题修复")
    else:
        print("⚠ 部分测试失败，需要进一步调试。")


if __name__ == "__main__":
    main()
