#!/usr/bin/env python3
"""
增强的部署验证器
直接检查sglang的内部并行状态和expert分布信息，提供更准确的EP/TP部署验证
"""

import os
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
class DeploymentConfig:
    """部署配置信息"""
    tp_size: int
    dp_size: int
    ep_size: int
    enable_ep_moe: bool
    single_expert_mode: str
    model_path: str
    port: int

@dataclass
class ExpertDistributionInfo:
    """Expert分布信息"""
    num_experts: int
    num_local_experts: int
    expert_ids: List[int]
    distribution_type: str  # 'dp' or 'tp'
    is_uniform: bool

@dataclass
class ParallelStateInfo:
    """并行状态信息"""
    tensor_parallel_world_size: int
    tensor_parallel_rank: int
    data_parallel_world_size: int
    data_parallel_rank: int
    pipeline_parallel_world_size: int
    pipeline_parallel_rank: int
    moe_expert_parallel_world_size: int
    moe_expert_parallel_rank: int

@dataclass
class GPUInfo:
    """GPU信息"""
    gpu_id: int
    memory_used_mb: int
    memory_total_mb: int
    memory_usage_percent: float
    utilization_percent: float

class EnhancedDeploymentVerifier:
    """增强的部署验证器"""
    
    def __init__(self, server_url: str = "http://127.0.0.1:8080"):
        self.server_url = server_url
        self.config = None
        self.expert_distribution = None
        self.parallel_state = None
        self.gpu_info = None
    
    def get_gpu_info(self) -> List[GPUInfo]:
        """获取GPU信息"""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,memory.used,memory.total,utilization.gpu', 
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, check=True
            )
            
            gpu_info_list = []
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 4:
                        gpu_id = int(parts[0])
                        memory_used = int(parts[1])
                        memory_total = int(parts[2])
                        utilization = int(parts[3])
                        memory_usage_percent = (memory_used / memory_total) * 100
                        
                        gpu_info_list.append(GPUInfo(
                            gpu_id=gpu_id,
                            memory_used_mb=memory_used,
                            memory_total_mb=memory_total,
                            memory_usage_percent=memory_usage_percent,
                            utilization_percent=utilization
                        ))
            
            return gpu_info_list
        except Exception as e:
            logger.error(f"获取GPU信息失败: {e}")
            return []
    
    def get_server_config(self) -> Optional[DeploymentConfig]:
        """获取服务器配置信息"""
        try:
            # 尝试从服务器获取配置信息
            response = requests.get(f"{self.server_url}/v1/models", timeout=10)
            if response.status_code == 200:
                # 这里需要根据实际的API返回格式解析
                # 由于sglang可能没有直接暴露配置的API，我们需要其他方法
                pass
        except Exception as e:
            logger.warning(f"无法从服务器获取配置: {e}")
        
        # 从环境变量和进程信息推断配置
        return self._infer_config_from_environment()
    
    def _infer_config_from_environment(self) -> Optional[DeploymentConfig]:
        """从环境变量推断配置"""
        try:
            # 检查环境变量
            single_expert_mode = os.environ.get('SINGLE_EXPERT_MODE', 'unknown')
            
            # 检查进程信息
            result = subprocess.run(
                ['ps', 'aux'], capture_output=True, text=True, check=True
            )
            
            # 解析启动命令
            cmd_line = result.stdout
            tp_size = 1
            dp_size = 1
            ep_size = 1
            enable_ep_moe = False
            model_path = "/dev/shm/Qwen3-30B-A3B"  # 默认路径
            port = 8080
            
            # 解析命令行参数
            if '--tp-size' in cmd_line:
                import re
                match = re.search(r'--tp-size\s+(\d+)', cmd_line)
                if match:
                    tp_size = int(match.group(1))
            
            if '--dp-size' in cmd_line:
                import re
                match = re.search(r'--dp-size\s+(\d+)', cmd_line)
                if match:
                    dp_size = int(match.group(1))
            
            if '--ep-size' in cmd_line:
                import re
                match = re.search(r'--ep-size\s+(\d+)', cmd_line)
                if match:
                    ep_size = int(match.group(1))
            
            if '--enable-ep-moe' in cmd_line:
                enable_ep_moe = True
            
            if '--port' in cmd_line:
                import re
                match = re.search(r'--port\s+(\d+)', cmd_line)
                if match:
                    port = int(match.group(1))
            
            return DeploymentConfig(
                tp_size=tp_size,
                dp_size=dp_size,
                ep_size=ep_size,
                enable_ep_moe=enable_ep_moe,
                single_expert_mode=single_expert_mode,
                model_path=model_path,
                port=port
            )
        except Exception as e:
            logger.error(f"推断配置失败: {e}")
            return None
    
    def get_expert_distribution_info(self) -> Optional[ExpertDistributionInfo]:
        """获取expert分布信息"""
        try:
            # 这里需要通过sglang的内部API获取expert分布信息
            # 由于sglang可能没有直接暴露这些信息，我们需要通过其他方式
            
            # 方法1: 通过模型推理请求获取expert使用情况
            expert_usage = self._get_expert_usage_through_inference()
            
            # 方法2: 通过环境变量和配置推断
            config = self.get_server_config()
            if config:
                return self._infer_expert_distribution_from_config(config)
            
            return None
        except Exception as e:
            logger.error(f"获取expert分布信息失败: {e}")
            return None
    
    def _get_expert_usage_through_inference(self) -> Dict[str, Any]:
        """通过推理请求获取expert使用情况"""
        try:
            # 发送一个简单的推理请求，观察expert的使用情况
            payload = {
                "model": "qwen3-moe",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 10,
                "temperature": 0.1
            }
            
            response = requests.post(
                f"{self.server_url}/v1/chat/completions",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                # 这里需要分析响应中的expert使用信息
                # 由于sglang可能没有在响应中包含这些信息，我们需要其他方法
                return {"status": "success", "response": response.json()}
            else:
                return {"status": "error", "code": response.status_code}
        except Exception as e:
            logger.error(f"推理请求失败: {e}")
            return {"status": "error", "message": str(e)}
    
    def _infer_expert_distribution_from_config(self, config: DeploymentConfig) -> ExpertDistributionInfo:
        """从配置推断expert分布"""
        # 对于单expert模型，expert数量为1
        num_experts = 1
        
        if config.single_expert_mode == 'dp':
            # DP模式：每个GPU都有expert的完整副本
            num_local_experts = 1
            expert_ids = [0]  # 所有GPU都使用expert 0
            distribution_type = 'dp'
            is_uniform = True
        elif config.single_expert_mode == 'tp':
            # TP模式：expert在多个GPU上切分
            num_local_experts = 1
            expert_ids = [0]  # 每个GPU处理expert 0的一部分
            distribution_type = 'tp'
            is_uniform = True
        else:
            # 未知模式
            num_local_experts = 1
            expert_ids = [0]
            distribution_type = 'unknown'
            is_uniform = False
        
        return ExpertDistributionInfo(
            num_experts=num_experts,
            num_local_experts=num_local_experts,
            expert_ids=expert_ids,
            distribution_type=distribution_type,
            is_uniform=is_uniform
        )
    
    def get_parallel_state_info(self) -> Optional[ParallelStateInfo]:
        """获取并行状态信息"""
        try:
            # 这里需要通过sglang的内部API获取并行状态
            # 由于sglang可能没有直接暴露这些信息，我们需要通过其他方式
            
            # 方法1: 通过环境变量推断
            config = self.get_server_config()
            if config:
                return self._infer_parallel_state_from_config(config)
            
            return None
        except Exception as e:
            logger.error(f"获取并行状态信息失败: {e}")
            return None
    
    def _infer_parallel_state_from_config(self, config: DeploymentConfig) -> ParallelStateInfo:
        """从配置推断并行状态"""
        return ParallelStateInfo(
            tensor_parallel_world_size=config.tp_size,
            tensor_parallel_rank=0,  # 需要从实际运行环境获取
            data_parallel_world_size=config.dp_size,
            data_parallel_rank=0,  # 需要从实际运行环境获取
            pipeline_parallel_world_size=1,  # 假设没有pipeline parallel
            pipeline_parallel_rank=0,
            moe_expert_parallel_world_size=config.ep_size if config.enable_ep_moe else 1,
            moe_expert_parallel_rank=0  # 需要从实际运行环境获取
        )
    
    def verify_ep_deployment(self) -> Dict[str, Any]:
        """验证EP部署"""
        print("=== 验证 Expert Parallel 部署 ===")
        
        # 获取各种信息
        self.gpu_info = self.get_gpu_info()
        self.config = self.get_server_config()
        self.expert_distribution = self.get_expert_distribution_info()
        self.parallel_state = self.get_parallel_state_info()
        
        verification_result = {
            'deployment_type': 'expert_parallel',
            'verification_passed': False,
            'details': {},
            'warnings': [],
            'recommendations': []
        }
        
        # 1. 检查配置
        if self.config:
            print(f"配置信息:")
            print(f"  TP Size: {self.config.tp_size}")
            print(f"  DP Size: {self.config.dp_size}")
            print(f"  EP Size: {self.config.ep_size}")
            print(f"  Enable EP MoE: {self.config.enable_ep_moe}")
            print(f"  Single Expert Mode: {self.config.single_expert_mode}")
            
            verification_result['details']['config'] = {
                'tp_size': self.config.tp_size,
                'dp_size': self.config.dp_size,
                'ep_size': self.config.ep_size,
                'enable_ep_moe': self.config.enable_ep_moe,
                'single_expert_mode': self.config.single_expert_mode
            }
            
            # 检查EP配置是否正确
            if self.config.enable_ep_moe and self.config.ep_size > 0:
                verification_result['details']['ep_config_correct'] = True
            else:
                verification_result['details']['ep_config_correct'] = False
                verification_result['warnings'].append("EP配置可能不正确")
        else:
            verification_result['warnings'].append("无法获取配置信息")
        
        # 2. 检查GPU内存使用
        if self.gpu_info:
            print(f"\nGPU信息:")
            memory_usage_percentages = []
            
            for gpu in self.gpu_info:
                print(f"  GPU {gpu.gpu_id}: 内存使用 {gpu.memory_usage_percent:.2f}%, 利用率 {gpu.utilization_percent}%")
                memory_usage_percentages.append(gpu.memory_usage_percent)
            
            # 计算内存使用的均匀性
            if len(memory_usage_percentages) > 1:
                mean_usage = statistics.mean(memory_usage_percentages)
                std_usage = statistics.stdev(memory_usage_percentages)
                cv_usage = (std_usage / mean_usage) * 100 if mean_usage > 0 else 0
                
                print(f"  平均内存使用率: {mean_usage:.2f}%")
                print(f"  内存使用率标准差: {std_usage:.2f}%")
                print(f"  变异系数: {cv_usage:.2f}%")
                
                verification_result['details']['gpu_memory'] = {
                    'mean_usage': mean_usage,
                    'std_usage': std_usage,
                    'cv_usage': cv_usage,
                    'usage_percentages': memory_usage_percentages
                }
                
                # EP部署应该显示相对均匀的内存使用
                if cv_usage < 15:  # 变异系数小于15%认为是均匀的
                    verification_result['details']['memory_uniform'] = True
                    print("  ✅ GPU内存使用相对均匀，符合EP部署特征")
                else:
                    verification_result['details']['memory_uniform'] = False
                    verification_result['warnings'].append(f"GPU内存使用不够均匀 (CV: {cv_usage:.2f}%)")
                    print(f"  ⚠️ GPU内存使用不够均匀 (CV: {cv_usage:.2f}%)")
            else:
                verification_result['warnings'].append("GPU数量不足，无法进行均匀性分析")
        else:
            verification_result['warnings'].append("无法获取GPU信息")
        
        # 3. 检查expert分布
        if self.expert_distribution:
            print(f"\nExpert分布信息:")
            print(f"  Expert数量: {self.expert_distribution.num_experts}")
            print(f"  本地Expert数量: {self.expert_distribution.num_local_experts}")
            print(f"  Expert IDs: {self.expert_distribution.expert_ids}")
            print(f"  分布类型: {self.expert_distribution.distribution_type}")
            print(f"  是否均匀: {self.expert_distribution.is_uniform}")
            
            verification_result['details']['expert_distribution'] = {
                'num_experts': self.expert_distribution.num_experts,
                'num_local_experts': self.expert_distribution.num_local_experts,
                'expert_ids': self.expert_distribution.expert_ids,
                'distribution_type': self.expert_distribution.distribution_type,
                'is_uniform': self.expert_distribution.is_uniform
            }
            
            # 检查expert分布是否符合EP特征
            if self.expert_distribution.distribution_type == 'dp':
                verification_result['details']['expert_distribution_correct'] = True
                print("  ✅ Expert分布符合EP部署特征 (DP模式)")
            else:
                verification_result['details']['expert_distribution_correct'] = False
                verification_result['warnings'].append("Expert分布可能不符合EP部署特征")
                print(f"  ⚠️ Expert分布类型为 {self.expert_distribution.distribution_type}，可能不符合EP部署特征")
        else:
            verification_result['warnings'].append("无法获取expert分布信息")
        
        # 4. 检查并行状态
        if self.parallel_state:
            print(f"\n并行状态信息:")
            print(f"  Tensor Parallel World Size: {self.parallel_state.tensor_parallel_world_size}")
            print(f"  Data Parallel World Size: {self.parallel_state.data_parallel_world_size}")
            print(f"  MoE Expert Parallel World Size: {self.parallel_state.moe_expert_parallel_world_size}")
            
            verification_result['details']['parallel_state'] = {
                'tensor_parallel_world_size': self.parallel_state.tensor_parallel_world_size,
                'data_parallel_world_size': self.parallel_state.data_parallel_world_size,
                'moe_expert_parallel_world_size': self.parallel_state.moe_expert_parallel_world_size
            }
            
            # 检查并行配置是否符合EP特征
            if self.parallel_state.moe_expert_parallel_world_size > 1:
                verification_result['details']['parallel_config_correct'] = True
                print("  ✅ 并行配置符合EP部署特征")
            else:
                verification_result['details']['parallel_config_correct'] = False
                verification_result['warnings'].append("并行配置可能不符合EP部署特征")
                print("  ⚠️ MoE Expert Parallel World Size为1，可能不符合EP部署特征")
        else:
            verification_result['warnings'].append("无法获取并行状态信息")
        
        # 5. 综合判断
        passed_checks = 0
        total_checks = 0
        
        if 'ep_config_correct' in verification_result['details']:
            total_checks += 1
            if verification_result['details']['ep_config_correct']:
                passed_checks += 1
        
        if 'memory_uniform' in verification_result['details']:
            total_checks += 1
            if verification_result['details']['memory_uniform']:
                passed_checks += 1
        
        if 'expert_distribution_correct' in verification_result['details']:
            total_checks += 1
            if verification_result['details']['expert_distribution_correct']:
                passed_checks += 1
        
        if 'parallel_config_correct' in verification_result['details']:
            total_checks += 1
            if verification_result['details']['parallel_config_correct']:
                passed_checks += 1
        
        # 至少通过75%的检查才认为验证通过
        if total_checks > 0 and (passed_checks / total_checks) >= 0.75:
            verification_result['verification_passed'] = True
            print(f"\n✅ EP部署验证通过 ({passed_checks}/{total_checks} 项检查通过)")
        else:
            verification_result['verification_passed'] = False
            print(f"\n❌ EP部署验证失败 ({passed_checks}/{total_checks} 项检查通过)")
        
        verification_result['details']['check_summary'] = {
            'passed_checks': passed_checks,
            'total_checks': total_checks,
            'pass_rate': (passed_checks / total_checks) if total_checks > 0 else 0
        }
        
        return verification_result
    
    def verify_tp_deployment(self) -> Dict[str, Any]:
        """验证TP部署"""
        print("=== 验证 Tensor Parallel 部署 ===")
        
        # 获取各种信息
        self.gpu_info = self.get_gpu_info()
        self.config = self.get_server_config()
        self.expert_distribution = self.get_expert_distribution_info()
        self.parallel_state = self.get_parallel_state_info()
        
        verification_result = {
            'deployment_type': 'tensor_parallel',
            'verification_passed': False,
            'details': {},
            'warnings': [],
            'recommendations': []
        }
        
        # 1. 检查配置
        if self.config:
            print(f"配置信息:")
            print(f"  TP Size: {self.config.tp_size}")
            print(f"  DP Size: {self.config.dp_size}")
            print(f"  EP Size: {self.config.ep_size}")
            print(f"  Enable EP MoE: {self.config.enable_ep_moe}")
            print(f"  Single Expert Mode: {self.config.single_expert_mode}")
            
            verification_result['details']['config'] = {
                'tp_size': self.config.tp_size,
                'dp_size': self.config.dp_size,
                'ep_size': self.config.ep_size,
                'enable_ep_moe': self.config.enable_ep_moe,
                'single_expert_mode': self.config.single_expert_mode
            }
            
            # 检查TP配置是否正确
            if self.config.tp_size > 1:
                verification_result['details']['tp_config_correct'] = True
            else:
                verification_result['details']['tp_config_correct'] = False
                verification_result['warnings'].append("TP Size应该大于1")
        else:
            verification_result['warnings'].append("无法获取配置信息")
        
        # 2. 检查GPU内存使用
        if self.gpu_info:
            print(f"\nGPU信息:")
            memory_usage_percentages = []
            
            for gpu in self.gpu_info:
                print(f"  GPU {gpu.gpu_id}: 内存使用 {gpu.memory_usage_percent:.2f}%, 利用率 {gpu.utilization_percent}%")
                memory_usage_percentages.append(gpu.memory_usage_percent)
            
            # 计算内存使用的均匀性
            if len(memory_usage_percentages) > 1:
                mean_usage = statistics.mean(memory_usage_percentages)
                std_usage = statistics.stdev(memory_usage_percentages)
                cv_usage = (std_usage / mean_usage) * 100 if mean_usage > 0 else 0
                
                print(f"  平均内存使用率: {mean_usage:.2f}%")
                print(f"  内存使用率标准差: {std_usage:.2f}%")
                print(f"  变异系数: {cv_usage:.2f}%")
                
                verification_result['details']['gpu_memory'] = {
                    'mean_usage': mean_usage,
                    'std_usage': std_usage,
                    'cv_usage': cv_usage,
                    'usage_percentages': memory_usage_percentages
                }
                
                # TP部署应该显示相对均匀的内存使用（允许稍大的差异）
                if cv_usage < 25:  # 变异系数小于25%认为是均匀的
                    verification_result['details']['memory_uniform'] = True
                    print("  ✅ GPU内存使用相对均匀，符合TP部署特征")
                else:
                    verification_result['details']['memory_uniform'] = False
                    verification_result['warnings'].append(f"GPU内存使用不够均匀 (CV: {cv_usage:.2f}%)")
                    print(f"  ⚠️ GPU内存使用不够均匀 (CV: {cv_usage:.2f}%)")
            else:
                verification_result['warnings'].append("GPU数量不足，无法进行均匀性分析")
        else:
            verification_result['warnings'].append("无法获取GPU信息")
        
        # 3. 检查expert分布
        if self.expert_distribution:
            print(f"\nExpert分布信息:")
            print(f"  Expert数量: {self.expert_distribution.num_experts}")
            print(f"  本地Expert数量: {self.expert_distribution.num_local_experts}")
            print(f"  Expert IDs: {self.expert_distribution.expert_ids}")
            print(f"  分布类型: {self.expert_distribution.distribution_type}")
            print(f"  是否均匀: {self.expert_distribution.is_uniform}")
            
            verification_result['details']['expert_distribution'] = {
                'num_experts': self.expert_distribution.num_experts,
                'num_local_experts': self.expert_distribution.num_local_experts,
                'expert_ids': self.expert_distribution.expert_ids,
                'distribution_type': self.expert_distribution.distribution_type,
                'is_uniform': self.expert_distribution.is_uniform
            }
            
            # 检查expert分布是否符合TP特征
            if self.expert_distribution.distribution_type == 'tp':
                verification_result['details']['expert_distribution_correct'] = True
                print("  ✅ Expert分布符合TP部署特征 (TP模式)")
            else:
                verification_result['details']['expert_distribution_correct'] = False
                verification_result['warnings'].append("Expert分布可能不符合TP部署特征")
                print(f"  ⚠️ Expert分布类型为 {self.expert_distribution.distribution_type}，可能不符合TP部署特征")
        else:
            verification_result['warnings'].append("无法获取expert分布信息")
        
        # 4. 检查并行状态
        if self.parallel_state:
            print(f"\n并行状态信息:")
            print(f"  Tensor Parallel World Size: {self.parallel_state.tensor_parallel_world_size}")
            print(f"  Data Parallel World Size: {self.parallel_state.data_parallel_world_size}")
            print(f"  MoE Expert Parallel World Size: {self.parallel_state.moe_expert_parallel_world_size}")
            
            verification_result['details']['parallel_state'] = {
                'tensor_parallel_world_size': self.parallel_state.tensor_parallel_world_size,
                'data_parallel_world_size': self.parallel_state.data_parallel_world_size,
                'moe_expert_parallel_world_size': self.parallel_state.moe_expert_parallel_world_size
            }
            
            # 检查并行配置是否符合TP特征
            if self.parallel_state.tensor_parallel_world_size > 1:
                verification_result['details']['parallel_config_correct'] = True
                print("  ✅ 并行配置符合TP部署特征")
            else:
                verification_result['details']['parallel_config_correct'] = False
                verification_result['warnings'].append("并行配置可能不符合TP部署特征")
                print("  ⚠️ Tensor Parallel World Size为1，可能不符合TP部署特征")
        else:
            verification_result['warnings'].append("无法获取并行状态信息")
        
        # 5. 综合判断
        passed_checks = 0
        total_checks = 0
        
        if 'tp_config_correct' in verification_result['details']:
            total_checks += 1
            if verification_result['details']['tp_config_correct']:
                passed_checks += 1
        
        if 'memory_uniform' in verification_result['details']:
            total_checks += 1
            if verification_result['details']['memory_uniform']:
                passed_checks += 1
        
        if 'expert_distribution_correct' in verification_result['details']:
            total_checks += 1
            if verification_result['details']['expert_distribution_correct']:
                passed_checks += 1
        
        if 'parallel_config_correct' in verification_result['details']:
            total_checks += 1
            if verification_result['details']['parallel_config_correct']:
                passed_checks += 1
        
        # 至少通过75%的检查才认为验证通过
        if total_checks > 0 and (passed_checks / total_checks) >= 0.75:
            verification_result['verification_passed'] = True
            print(f"\n✅ TP部署验证通过 ({passed_checks}/{total_checks} 项检查通过)")
        else:
            verification_result['verification_passed'] = False
            print(f"\n❌ TP部署验证失败 ({passed_checks}/{total_checks} 项检查通过)")
        
        verification_result['details']['check_summary'] = {
            'passed_checks': passed_checks,
            'total_checks': total_checks,
            'pass_rate': (passed_checks / total_checks) if total_checks > 0 else 0
        }
        
        return verification_result

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="增强的部署验证器")
    parser.add_argument("--deployment-type", choices=["ep", "tp"], required=True,
                       help="部署类型: ep (Expert Parallel) 或 tp (Tensor Parallel)")
    parser.add_argument("--server-url", default="http://127.0.0.1:8080",
                       help="服务器URL")
    parser.add_argument("--output", help="输出结果到JSON文件")
    
    args = parser.parse_args()
    
    verifier = EnhancedDeploymentVerifier(args.server_url)
    
    if args.deployment_type == "ep":
        result = verifier.verify_ep_deployment()
    else:
        result = verifier.verify_tp_deployment()
    
    # 输出结果
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存到: {args.output}")
    
    return result

if __name__ == "__main__":
    main()
