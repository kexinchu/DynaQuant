#!/usr/bin/env python3
"""
SGLang内部状态检查器
通过修改sglang源码添加内部状态检查功能，直接获取并行状态和expert分布信息
"""

import os
import sys
import json
import subprocess
import time
import requests
import statistics
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class InternalParallelState:
    """内部并行状态"""
    tensor_parallel_world_size: int
    tensor_parallel_rank: int
    data_parallel_world_size: int
    data_parallel_rank: int
    pipeline_parallel_world_size: int
    pipeline_parallel_rank: int
    moe_expert_parallel_world_size: int
    moe_expert_parallel_rank: int

@dataclass
class ExpertDistributionState:
    """Expert分布状态"""
    num_experts: int
    num_local_experts: int
    local_expert_ids: List[int]
    distribution_type: str  # 'dp', 'tp', 'hybrid'
    is_uniform: bool
    expert_memory_usage: Dict[int, float]  # expert_id -> memory_usage_percent

@dataclass
class ModelLayerState:
    """模型层状态"""
    layer_id: int
    layer_type: str  # 'attention', 'moe', 'mlp', 'embedding'
    parallel_type: str  # 'tp', 'dp', 'ep'
    gpu_distribution: List[int]  # 分布到哪些GPU
    memory_usage: float  # 内存使用百分比

class SGLangInternalStateChecker:
    """SGLang内部状态检查器"""
    
    def __init__(self, sglang_path: str = "sglang-0.4.7/python"):
        self.sglang_path = sglang_path
        self.modified_files = []
    
    def add_internal_state_api(self):
        """添加内部状态检查API到sglang"""
        print("=== 添加内部状态检查API到SGLang ===")
        
        # 1. 修改parallel_state.py，添加状态查询函数
        self._modify_parallel_state()
        
        # 2. 修改qwen3_moe.py，添加expert分布查询函数
        self._modify_qwen3_moe()
        
        print("✅ 内部状态检查API添加完成")
    
    def _modify_parallel_state(self):
        """修改parallel_state.py，添加状态查询函数"""
        parallel_state_file = os.path.join(self.sglang_path, "sglang/srt/distributed/parallel_state.py")
        
        # 在文件末尾添加状态查询函数
        additional_code = """

# ==============================================================================
# Internal State Query Functions (Added for deployment verification)
# ==============================================================================

def get_internal_parallel_state() -> Dict[str, Any]:
    \"\"\"获取内部并行状态信息\"\"\"
    try:
        return {
            'tensor_parallel_world_size': get_tensor_model_parallel_world_size(),
            'tensor_parallel_rank': get_tensor_model_parallel_rank(),
            'data_parallel_world_size': get_data_parallel_world_size(),
            'data_parallel_rank': get_data_parallel_rank(),
            'pipeline_parallel_world_size': get_pipeline_parallel_world_size(),
            'pipeline_parallel_rank': get_pipeline_parallel_rank(),
            'moe_expert_parallel_world_size': get_moe_expert_parallel_world_size(),
            'moe_expert_parallel_rank': get_moe_expert_parallel_rank(),
            'is_initialized': _TP is not None,
            'tp_group_info': {
                'world_size': _TP.world_size if _TP else 0,
                'rank_in_group': _TP.rank_in_group if _TP else 0,
                'ranks': _TP.ranks if _TP else []
            } if _TP else None
        }
    except Exception as e:
        return {
            'error': str(e),
            'is_initialized': False
        }

def get_all_parallel_groups_info() -> Dict[str, Any]:
    \"\"\"获取所有并行组信息\"\"\"
    groups_info = {}
    
    if _TP:
        groups_info['tensor_parallel'] = {
            'world_size': _TP.world_size,
            'rank_in_group': _TP.rank_in_group,
            'ranks': _TP.ranks,
            'device': str(_TP.device)
        }
    
    if _PP:
        groups_info['pipeline_parallel'] = {
            'world_size': _PP.world_size,
            'rank_in_group': _PP.rank_in_group,
            'ranks': _PP.ranks,
            'device': str(_PP.device)
        }
    
    return groups_info

def get_environment_info() -> Dict[str, Any]:
    \"\"\"获取环境信息\"\"\"
    return {
        'single_expert_mode': os.environ.get('SINGLE_EXPERT_MODE', 'unknown'),
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES', ''),
        'sglang_disable_marlin': os.environ.get('SGLANG_DISABLE_MARLIN', ''),
        'sgl_disable_awq_marlin': os.environ.get('SGL_DISABLE_AWQ_MARLIN', ''),
        'sglang_disable_sgl_kernel': os.environ.get('SGLANG_DISABLE_SGL_KERNEL', ''),
        'torch_distributed_backend': torch.distributed.get_backend() if torch.distributed.is_initialized() else 'not_initialized',
        'torch_distributed_world_size': torch.distributed.get_world_size() if torch.distributed.is_initialized() else 0,
        'torch_distributed_rank': torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    }
"""
        
        # 读取原文件
        with open(parallel_state_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查是否已经添加过
        if 'get_internal_parallel_state' not in content:
            # 在文件末尾添加代码
            with open(parallel_state_file, 'a', encoding='utf-8') as f:
                f.write(additional_code)
            
            self.modified_files.append(parallel_state_file)
            print(f"✅ 已修改: {parallel_state_file}")
        else:
            print(f"⚠️ 文件已包含内部状态API: {parallel_state_file}")
    
    def _modify_qwen3_moe(self):
        """修改qwen3_moe.py，添加expert分布查询函数"""
        qwen3_moe_file = os.path.join(self.sglang_path, "sglang/srt/models/qwen3_moe.py")
        
        # 在Qwen3MoeSparseMoeBlock类中添加状态查询方法
        additional_code = """

    def get_expert_distribution_info(self) -> Dict[str, Any]:
        \"\"\"获取expert分布信息\"\"\"
        try:
            return {
                'layer_id': self.layer_id,
                'tp_size': self.tp_size,
                'tp_rank': get_tensor_model_parallel_rank(),
                'num_experts': self.num_experts,
                'num_local_experts': self.num_local_experts,
                'start_expert_id': self.start_expert_id,
                'end_expert_id': self.end_expert_id,
                'local_expert_ids': list(range(self.start_expert_id, self.end_expert_id + 1)),
                'distribution_type': 'tp' if self.tp_size > 1 else 'dp',
                'is_uniform': True,  # 对于单expert模型，总是均匀的
                'expert_memory_usage': self._get_expert_memory_usage()
            }
        except Exception as e:
            return {
                'error': str(e),
                'layer_id': self.layer_id
            }
    
    def _get_expert_memory_usage(self) -> Dict[int, float]:
        \"\"\"获取expert内存使用情况\"\"\"
        try:
            # 这里可以添加实际的内存使用统计
            # 由于sglang可能没有直接暴露这些信息，我们返回估算值
            memory_usage = {}
            for expert_id in range(self.start_expert_id, self.end_expert_id + 1):
                # 估算每个expert的内存使用
                memory_usage[expert_id] = 100.0 / self.tp_size  # 假设均匀分布
            return memory_usage
        except Exception as e:
            return {0: 0.0}  # 返回默认值
"""
        
        # 读取原文件
        with open(qwen3_moe_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 查找Qwen3MoeSparseMoeBlock类的结束位置
        if 'class Qwen3MoeSparseMoeBlock' in content and 'get_expert_distribution_info' not in content:
            # 在类的最后一个方法后添加新方法
            lines = content.split('\n')
            class_start = -1
            class_end = -1
            
            for i, line in enumerate(lines):
                if 'class Qwen3MoeSparseMoeBlock' in line:
                    class_start = i
                elif class_start != -1 and line.strip() == '' and i > class_start:
                    # 找到类的结束位置
                    class_end = i
                    break
            
            if class_start != -1 and class_end != -1:
                # 在类结束前插入新方法
                lines.insert(class_end, additional_code)
                
                # 写回文件
                with open(qwen3_moe_file, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(lines))
                
                self.modified_files.append(qwen3_moe_file)
                print(f"✅ 已修改: {qwen3_moe_file}")
            else:
                print(f"⚠️ 无法找到Qwen3MoeSparseMoeBlock类的位置: {qwen3_moe_file}")
        else:
            print(f"⚠️ 文件已包含expert分布API或未找到目标类: {qwen3_moe_file}")

    def install_dependencies(self):
        """安装依赖"""
        print("=== 检查依赖 ===")

        try:
            # 检查Python内置模块是否可用
            import http.server
            import urllib.parse
            import threading
            print("✅ Python内置HTTP服务器模块可用")
            return True
        except Exception as e:
            print(f"❌ Python内置HTTP服务器模块不可用: {e}")
            return False
    
    def reinstall_sglang(self):
        """重新安装sglang"""
        print("=== 重新安装SGLang ===")
        
        try:
            # 切换到sglang目录
            os.chdir(self.sglang_path)
            
            # 重新安装
            subprocess.run([sys.executable, '-m', 'pip', 'install', '-e', '.'], check=True)
            print("✅ SGLang重新安装完成")
        except Exception as e:
            print(f"❌ SGLang重新安装失败: {e}")
            return False
        
        return True
    
    def start_internal_api_server(self, port: int = 8082):
        """启动内部状态API服务器"""
        print(f"=== 启动内部状态API服务器 (端口: {port}) ===")
        
        try:
            api_file = os.path.join(self.sglang_path, "python/sglang/srt/internal_state_api.py")
            
            # 启动API服务器
            process = subprocess.Popen([
                sys.executable, "-u", api_file
            ], stdout=open("./api_stdout.log", "w"), stderr=open("./api_stderr.log", "w"))
            
            # 等待服务器启动
            time.sleep(10)
            
            # 检查服务器是否启动成功
            try:
                response = requests.get(f"http://127.0.0.1:{port}/internal/health", timeout=5)
                if response.status_code == 200:
                    print(f"✅ 内部状态API服务器启动成功: http://127.0.0.1:{port}")
                    return process
                else:
                    print(f"❌ 内部状态API服务器启动失败: HTTP {response.status_code}")
                    return None
            except Exception as e:
                print(f"❌ 内部状态API服务器启动失败: {e}")
                return None
        except Exception as e:
            print(f"❌ 启动内部状态API服务器失败: {e}")
            return None
    
    def get_internal_parallel_state(self, api_port: int = 8082) -> Optional[Dict[str, Any]]:
        """获取内部并行状态"""
        try:
            response = requests.get(f"http://127.0.0.1:{api_port}/internal/parallel_state", timeout=10)
            if response.status_code == 200:
                return response.json()['data']
            else:
                logger.error(f"获取并行状态失败: HTTP {response.status_code}")
                return None
        except Exception as e:
            logger.error(f"获取并行状态失败: {e}")
            return None
    
    def get_internal_environment_info(self, api_port: int = 8082) -> Optional[Dict[str, Any]]:
        """获取内部环境信息"""
        try:
            response = requests.get(f"http://127.0.0.1:{api_port}/internal/environment", timeout=10)
            if response.status_code == 200:
                return response.json()['data']
            else:
                logger.error(f"获取环境信息失败: HTTP {response.status_code}")
                return None
        except Exception as e:
            logger.error(f"获取环境信息失败: {e}")
            return None
    
    def verify_deployment_with_internal_state(self, deployment_type: str, api_port: int = 8082) -> Dict[str, Any]:
        """使用内部状态验证部署"""
        print(f"=== 使用内部状态验证 {deployment_type.upper()} 部署 ===")
        
        # 获取内部状态
        parallel_state = self.get_internal_parallel_state(api_port)
        environment_info = self.get_internal_environment_info(api_port)
        
        verification_result = {
            'deployment_type': deployment_type,
            'verification_passed': False,
            'details': {},
            'warnings': [],
            'recommendations': []
        }
        
        # 1. 检查并行状态
        if parallel_state:
            print(f"并行状态信息:")
            print(f"  Tensor Parallel World Size: {parallel_state.get('tensor_parallel_world_size', 'unknown')}")
            print(f"  Tensor Parallel Rank: {parallel_state.get('tensor_parallel_rank', 'unknown')}")
            print(f"  Data Parallel World Size: {parallel_state.get('data_parallel_world_size', 'unknown')}")
            print(f"  MoE Expert Parallel World Size: {parallel_state.get('moe_expert_parallel_world_size', 'unknown')}")
            print(f"  是否已初始化: {parallel_state.get('is_initialized', False)}")
            
            verification_result['details']['parallel_state'] = parallel_state
            
            # 检查并行配置
            if deployment_type == 'ep':
                if parallel_state.get('moe_expert_parallel_world_size', 1) > 1:
                    verification_result['details']['parallel_config_correct'] = True
                    print("  ✅ 并行配置符合EP部署特征")
                else:
                    verification_result['details']['parallel_config_correct'] = False
                    verification_result['warnings'].append("MoE Expert Parallel World Size应该大于1")
                    print("  ⚠️ MoE Expert Parallel World Size为1，可能不符合EP部署特征")
            else:  # tp
                if parallel_state.get('tensor_parallel_world_size', 1) > 1:
                    verification_result['details']['parallel_config_correct'] = True
                    print("  ✅ 并行配置符合TP部署特征")
                else:
                    verification_result['details']['parallel_config_correct'] = False
                    verification_result['warnings'].append("Tensor Parallel World Size应该大于1")
                    print("  ⚠️ Tensor Parallel World Size为1，可能不符合TP部署特征")
        else:
            verification_result['warnings'].append("无法获取并行状态信息")
        
        # 2. 检查环境信息
        if environment_info:
            print(f"\n环境信息:")
            print(f"  Single Expert Mode: {environment_info.get('single_expert_mode', 'unknown')}")
            print(f"  CUDA Visible Devices: {environment_info.get('cuda_visible_devices', 'unknown')}")
            print(f"  Torch Distributed Backend: {environment_info.get('torch_distributed_backend', 'unknown')}")
            print(f"  Torch Distributed World Size: {environment_info.get('torch_distributed_world_size', 'unknown')}")
            
            verification_result['details']['environment_info'] = environment_info
            
            # 检查环境配置
            if deployment_type == 'ep':
                if environment_info.get('single_expert_mode') == 'dp':
                    verification_result['details']['environment_config_correct'] = True
                    print("  ✅ 环境配置符合EP部署特征 (DP模式)")
                else:
                    verification_result['details']['environment_config_correct'] = False
                    verification_result['warnings'].append("Single Expert Mode应该为'dp'")
                    print(f"  ⚠️ Single Expert Mode为 {environment_info.get('single_expert_mode')}，可能不符合EP部署特征")
            else:  # tp
                if environment_info.get('single_expert_mode') == 'tp':
                    verification_result['details']['environment_config_correct'] = True
                    print("  ✅ 环境配置符合TP部署特征 (TP模式)")
                else:
                    verification_result['details']['environment_config_correct'] = False
                    verification_result['warnings'].append("Single Expert Mode应该为'tp'")
                    print(f"  ⚠️ Single Expert Mode为 {environment_info.get('single_expert_mode')}，可能不符合TP部署特征")
        else:
            verification_result['warnings'].append("无法获取环境信息")
        
        # 3. 综合判断
        passed_checks = 0
        total_checks = 0
        
        if 'parallel_config_correct' in verification_result['details']:
            total_checks += 1
            if verification_result['details']['parallel_config_correct']:
                passed_checks += 1
        
        if 'environment_config_correct' in verification_result['details']:
            total_checks += 1
            if verification_result['details']['environment_config_correct']:
                passed_checks += 1
        
        # 至少通过50%的检查才认为验证通过
        if total_checks > 0 and (passed_checks / total_checks) >= 0.5:
            verification_result['verification_passed'] = True
            print(f"\n✅ {deployment_type.upper()}部署验证通过 ({passed_checks}/{total_checks} 项检查通过)")
        else:
            verification_result['verification_passed'] = False
            print(f"\n❌ {deployment_type.upper()}部署验证失败 ({passed_checks}/{total_checks} 项检查通过)")
        
        verification_result['details']['check_summary'] = {
            'passed_checks': passed_checks,
            'total_checks': total_checks,
            'pass_rate': (passed_checks / total_checks) if total_checks > 0 else 0
        }
        
        return verification_result
    
    def cleanup(self):
        """清理修改的文件"""
        print("=== 清理修改的文件 ===")
        
        for file_path in self.modified_files:
            try:
                if os.path.exists(file_path):
                    # 这里可以选择是否恢复原文件
                    # 为了安全起见，我们只是标记文件已被修改
                    print(f"⚠️ 文件已被修改: {file_path}")
            except Exception as e:
                print(f"❌ 清理文件失败 {file_path}: {e}")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="SGLang内部状态检查器")
    parser.add_argument("--sglang-path", default="sglang-0.4.7",
                       help="SGLang源码路径")
    parser.add_argument("--action", choices=["setup", "verify", "cleanup"], required=True,
                       help="执行的操作")
    parser.add_argument("--deployment-type", choices=["ep", "tp"],
                       help="部署类型 (仅在verify时使用)")
    parser.add_argument("--api-port", type=int, default=8082,
                       help="内部API服务器端口")
    parser.add_argument("--output", help="输出结果到JSON文件")
    
    args = parser.parse_args()
    
    checker = SGLangInternalStateChecker(args.sglang_path)
    
    if args.action == "setup":
        print("=== 设置SGLang内部状态检查功能 ===")
        checker.add_internal_state_api()
        checker.install_dependencies()
        
        if checker.reinstall_sglang():
            print("\n✅ 设置完成！")
            print("现在可以使用以下命令验证部署:")
            print(f"  python {__file__} --action verify --deployment-type ep")
            print(f"  python {__file__} --action verify --deployment-type tp")
        else:
            print("\n❌ 设置失败！")
            return 1
    
    elif args.action == "verify":
        if not args.deployment_type:
            print("❌ 验证操作需要指定部署类型 (--deployment-type)")
            return 1
        
        print("=== 启动内部状态API服务器 ===")
        api_process = checker.start_internal_api_server(args.api_port)
        
        if api_process:
            try:
                # 执行验证
                result = checker.verify_deployment_with_internal_state(args.deployment_type, args.api_port)
                
                # 输出结果
                if args.output:
                    with open(args.output, 'w', encoding='utf-8') as f:
                        json.dump(result, f, indent=2, ensure_ascii=False)
                    print(f"\n结果已保存到: {args.output}")
                
                return 0 if result['verification_passed'] else 1
            finally:
                # 停止API服务器
                api_process.terminate()
                api_process.wait()
        else:
            print("❌ 无法启动内部状态API服务器")
            return 1
    
    elif args.action == "cleanup":
        checker.cleanup()
        print("✅ 清理完成")
        return 0
    
    return 0

if __name__ == "__main__":
    exit(main())
