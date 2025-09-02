#!/usr/bin/env python3
"""
快速诊断脚本
用于快速检查GPU使用率不均匀的原因
"""

import subprocess
import os
import time
import statistics

def check_environment():
    """检查环境变量"""
    print("=== 环境变量检查 ===")
    
    single_expert_mode = os.environ.get('SINGLE_EXPERT_MODE', '未设置')
    cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '未设置')
    
    print(f"SINGLE_EXPERT_MODE: {single_expert_mode}")
    print(f"CUDA_VISIBLE_DEVICES: {cuda_devices}")
    
    if single_expert_mode == 'dp':
        print("✅ Expert层使用DP模式")
    elif single_expert_mode == 'tp':
        print("✅ Expert层使用TP模式")
    else:
        print("⚠️  SINGLE_EXPERT_MODE未正确设置")
    
    return single_expert_mode

def check_gpu_utilization():
    """检查GPU使用率"""
    print("\n=== GPU使用率检查 ===")
    
    try:
        result = subprocess.run([
            'nvidia-smi', 
            '--query-gpu=index,utilization.gpu,memory.used,memory.total',
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode != 0:
            print("❌ 无法获取GPU信息")
            return None
        
        utilizations = []
        memory_usages = []
        
        print("GPU ID | 使用率 | 内存使用")
        print("-------|--------|----------")
        
        for line in result.stdout.strip().split('\n'):
            if line.strip():
                parts = line.split(', ')
                if len(parts) >= 4:
                    gpu_id = int(parts[0])
                    utilization = float(parts[1])
                    memory_used = float(parts[2])
                    memory_total = float(parts[3])
                    memory_usage_percent = (memory_used / memory_total) * 100
                    
                    utilizations.append(utilization)
                    memory_usages.append(memory_usage_percent)
                    
                    print(f"GPU {gpu_id:2d} | {utilization:6.1f}% | {memory_usage_percent:8.1f}%")
        
        # 计算统计信息
        if utilizations:
            mean_util = statistics.mean(utilizations)
            std_dev_util = statistics.stdev(utilizations) if len(utilizations) > 1 else 0
            cv_util = (std_dev_util / mean_util) * 100 if mean_util > 0 else 0
            
            mean_memory = statistics.mean(memory_usages)
            std_dev_memory = statistics.stdev(memory_usages) if len(memory_usages) > 1 else 0
            cv_memory = (std_dev_memory / mean_memory) * 100 if mean_memory > 0 else 0
            
            print(f"\n📊 统计信息:")
            print(f"  使用率 - 平均: {mean_util:.1f}%, 标准差: {std_dev_util:.1f}%, 变异系数: {cv_util:.1f}%")
            print(f"  内存使用 - 平均: {mean_memory:.1f}%, 标准差: {std_dev_memory:.1f}%, 变异系数: {cv_memory:.1f}%")
            
            # 判断均匀性
            if cv_util < 10:
                print("✅ GPU使用率均匀")
            elif cv_util < 15:
                print("⚠️  GPU使用率轻微不均")
            else:
                print("❌ GPU使用率明显不均")
            
            return {
                'utilizations': utilizations,
                'memory_usages': memory_usages,
                'cv_util': cv_util,
                'cv_memory': cv_memory
            }
        
    except Exception as e:
        print(f"❌ 检查GPU使用率失败: {e}")
    
    return None

def diagnose_issues(single_expert_mode, gpu_data):
    """诊断问题"""
    print("\n=== 问题诊断 ===")
    
    issues = []
    
    if not gpu_data:
        print("❌ 无法获取GPU数据")
        return issues
    
    cv_util = gpu_data['cv_util']
    utilizations = gpu_data['utilizations']
    
    # 检查使用率不均匀
    if cv_util > 15:
        issues.append("GPU使用率明显不均")
        print(f"❌ GPU使用率变异系数过高: {cv_util:.1f}%")
        
        # 找出问题GPU
        max_util = max(utilizations)
        min_util = min(utilizations)
        max_idx = utilizations.index(max_util)
        min_idx = utilizations.index(min_util)
        
        print(f"   - GPU {max_idx} 使用率最高: {max_util:.1f}%")
        print(f"   - GPU {min_idx} 使用率最低: {min_util:.1f}%")
        print(f"   - 差异: {max_util - min_util:.1f}%")
    
    # 检查配置问题
    if single_expert_mode == 'dp':
        print("🔍 当前使用DP模式，检查配置...")
        issues.append("DP模式配置可能有问题")
    elif single_expert_mode == 'tp':
        print("🔍 当前使用TP模式，检查配置...")
        issues.append("TP模式配置可能有问题")
    else:
        print("🔍 SINGLE_EXPERT_MODE未设置，这可能是问题所在")
        issues.append("SINGLE_EXPERT_MODE未正确设置")
    
    return issues

def suggest_solutions(issues, single_expert_mode):
    """建议解决方案"""
    print("\n=== 解决方案建议 ===")
    
    if not issues:
        print("✅ 未发现明显问题")
        return
    
    print(f"发现 {len(issues)} 个问题，建议解决方案:")
    
    for i, issue in enumerate(issues, 1):
        print(f"\n{i}. {issue}")
        
        if "使用率明显不均" in issue:
            print("   解决方案:")
            print("   - 使用纯EP配置: --tp-size 1 --dp-size 8")
            print("   - 或使用纯TP配置: --tp-size 8 --dp-size 1")
            print("   - 避免混合配置")
        
        if "DP模式配置" in issue:
            print("   解决方案:")
            print("   - 确保使用: --tp-size 1 --dp-size 8")
            print("   - 添加: --ep-dispatch-algorithm dynamic")
            print("   - 检查环境变量: export SINGLE_EXPERT_MODE=dp")
        
        if "TP模式配置" in issue:
            print("   解决方案:")
            print("   - 确保使用: --tp-size 8 --dp-size 1")
            print("   - 检查环境变量: export SINGLE_EXPERT_MODE=tp")
            print("   - 验证GPU间网络连接")
        
        if "SINGLE_EXPERT_MODE未正确设置" in issue:
            print("   解决方案:")
            print("   - 设置环境变量: export SINGLE_EXPERT_MODE=dp")
            print("   - 或设置: export SINGLE_EXPERT_MODE=tp")
            print("   - 重启服务器")

def main():
    """主函数"""
    print("🔍 快速诊断GPU使用率不均匀问题")
    print("="*50)
    
    # 1. 检查环境变量
    single_expert_mode = check_environment()
    
    # 2. 检查GPU使用率
    gpu_data = check_gpu_utilization()
    
    # 3. 诊断问题
    issues = diagnose_issues(single_expert_mode, gpu_data)
    
    # 4. 建议解决方案
    suggest_solutions(issues, single_expert_mode)
    
    print("\n" + "="*50)
    print("诊断完成！")
    
    if issues:
        print("\n💡 推荐立即尝试:")
        print("1. 修改test_single_expert_ep.py中的配置:")
        print("   --tp-size 1 --dp-size 8")
        print("2. 重新运行测试")
        print("3. 使用监控脚本验证效果:")
        print("   python gpu_utilization_monitor.py --single-check")

if __name__ == "__main__":
    main()

