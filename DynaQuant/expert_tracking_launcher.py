#!/usr/bin/env python3
"""
Expert Tracking 完整启动器
启动Qwen3-235B-A22B服务，启用expert tracking，使用sharegpt数据集测试，并在退出时导出结果
"""

import os
import sys
import time
import json
import signal
import logging
import threading
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional
import requests

# 添加SGLang路径
sys.path.insert(0, 'sglang-0.4.7/python')

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ExpertTrackingLauncher:
    """Expert Tracking 启动器"""
    
    def __init__(self):
        self.sglang_process = None
        self.expert_tracker = None
        self.shutdown_event = threading.Event()
        self.test_results = []
        
        # 注册信号处理器
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """信号处理器"""
        logger.info(f"收到信号 {signum}，开始优雅关闭...")
        self.shutdown_event.set()
        self.cleanup()
        sys.exit(0)
    
    def start_sglang_service(self):
        """启动SGLang服务"""
        try:
            logger.info("启动Qwen3-235B-A22B服务...")
            
            # 检查启动脚本是否存在
            script_path = "Qwen3-235B-A22B.sh"
            if not os.path.exists(script_path):
                logger.error(f"启动脚本不存在: {script_path}")
                return False
            
            # 启动服务
            self.sglang_process = subprocess.Popen(
                ["bash", script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid
            )
            
            # 等待服务启动
            logger.info("等待服务启动...")
            time.sleep(30)  # 等待30秒让服务完全启动
            
            # 检查服务是否正常运行
            if self.check_service_health():
                logger.info("✓ SGLang服务启动成功")
                return True
            else:
                logger.error("✗ SGLang服务启动失败")
                return False
                
        except Exception as e:
            logger.error(f"启动服务失败: {e}")
            return False
    
    def check_service_health(self) -> bool:
        """检查服务健康状态"""
        try:
            # 尝试连接服务
            response = requests.get("http://127.0.0.1:8080/health", timeout=10)
            return response.status_code == 200
        except Exception as e:
            logger.debug(f"服务健康检查失败: {e}")
            return False
    
    def enable_expert_tracking(self):
        """启用expert tracking"""
        try:
            logger.info("启用Expert Tracking功能...")
            
            from sglang.srt.model_loader.enhanced_mixed_precision_loader import (
                init_global_expert_tracker
            )
            
            # 初始化全局expert tracker
            self.expert_tracker = init_global_expert_tracker()
            logger.info("✓ Expert Tracking功能已启用")
            
            return True
            
        except Exception as e:
            logger.error(f"启用Expert Tracking失败: {e}")
            return False
    
    def load_sharegpt_dataset(self) -> List[Dict[str, Any]]:
        """加载ShareGPT数据集"""
        try:
            # 尝试从多个可能的路径加载数据集
            possible_paths = [
                "sharegpt_data.json",
                "sharegpt_data.jsonl", 
                "data/sharegpt_data.json",
                "datasets/sharegpt_data.json"
            ]
            
            dataset_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    dataset_path = path
                    break
            
            if not dataset_path:
                logger.warning("未找到ShareGPT数据集，创建示例数据")
                return self.create_sample_sharegpt_data()
            
            logger.info(f"加载ShareGPT数据集: {dataset_path}")
            
            if dataset_path.endswith('.jsonl'):
                # 读取JSONL格式
                data = []
                with open(dataset_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            data.append(json.loads(line))
            else:
                # 读取JSON格式
                with open(dataset_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            
            logger.info(f"✓ 成功加载 {len(data)} 条ShareGPT数据")
            return data
            
        except Exception as e:
            logger.error(f"加载ShareGPT数据集失败: {e}")
            logger.info("使用示例数据继续测试")
            return self.create_sample_sharegpt_data()
    
    def create_sample_sharegpt_data(self) -> List[Dict[str, Any]]:
        """创建示例ShareGPT数据"""
        sample_data = [
            {
                "id": "sample_001",
                "conversations": [
                    {"from": "human", "value": "请介绍一下人工智能的发展历史"},
                    {"from": "gpt", "value": "人工智能的发展可以追溯到20世纪50年代..."}
                ]
            },
            {
                "id": "sample_002", 
                "conversations": [
                    {"from": "human", "value": "什么是机器学习？请用通俗易懂的语言解释"},
                    {"from": "gpt", "value": "机器学习是人工智能的一个分支..."}
                ]
            },
            {
                "id": "sample_003",
                "conversations": [
                    {"from": "human", "value": "深度学习与传统机器学习有什么区别？"},
                    {"from": "gpt", "value": "深度学习是机器学习的一个子领域..."}
                ]
            }
        ]
        
        logger.info(f"创建了 {len(sample_data)} 条示例数据")
        return sample_data
    
    def test_with_sharegpt_data(self, dataset: List[Dict[str, Any]]):
        """使用ShareGPT数据集测试模型"""
        try:
            logger.info("开始使用ShareGPT数据集测试模型...")
            
            # 选择前几条数据进行测试
            test_data = dataset[:3]  # 只测试前3条，避免过长时间
            
            for i, item in enumerate(test_data):
                if self.shutdown_event.is_set():
                    break
                
                logger.info(f"测试数据 {i+1}/{len(test_data)}: {item.get('id', f'item_{i}')}")
                
                # 提取对话内容
                conversations = item.get('conversations', [])
                if not conversations:
                    continue
                
                # 构建测试请求
                messages = []
                for conv in conversations:
                    if conv.get('from') == 'human':
                        messages.append({
                            "role": "user",
                            "content": conv.get('value', '')
                        })
                    elif conv.get('from') == 'gpt':
                        messages.append({
                            "role": "assistant", 
                            "content": conv.get('value', '')
                        })
                
                if not messages:
                    continue
                
                # 发送请求到模型
                try:
                    response = self.send_chat_request(messages)
                    if response:
                        logger.info(f"  ✓ 请求成功，响应长度: {len(response)}")
                        self.test_results.append({
                            'id': item.get('id', f'item_{i}'),
                            'status': 'success',
                            'response_length': len(response)
                        })
                    else:
                        logger.warning(f"  ⚠ 请求失败")
                        self.test_results.append({
                            'id': item.get('id', f'item_{i}'),
                            'status': 'failed'
                        })
                except Exception as e:
                    logger.error(f"  ✗ 请求异常: {e}")
                    self.test_results.append({
                        'id': item.get('id', f'item_{i}'),
                        'response_length': 0
                    })
                
                # 等待一段时间，让expert tracker记录激活情况
                time.sleep(5)
            
            logger.info("✓ ShareGPT数据集测试完成")
            
        except Exception as e:
            logger.error(f"测试失败: {e}")
    
    def send_chat_request(self, messages: List[Dict[str, str]]) -> Optional[str]:
        """发送聊天请求到模型"""
        try:
            data = {
                "model": "qwen3-235b-a22b",
                "messages": messages,
                "max_tokens": 256,
                "temperature": 0.7,
                "top_p": 0.9
            }
            
            response = requests.post(
                "http://127.0.0.1:8080/v1/chat/completions",
                json=data,
                headers={
                    'Content-Type': 'application/json',
                    'Authorization': 'Bearer sk-local'
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                if 'choices' in result and result['choices']:
                    return result['choices'][0]['message']['content']
            
            return None
            
        except Exception as e:
            logger.error(f"发送请求失败: {e}")
            return None
    
    def calculate_hot_cold_scores(self) -> Dict[str, Any]:
        """计算hot-cold分数（基于激活次数）"""
        try:
            if not self.expert_tracker:
                return {}
            
            logger.info("计算expert hot-cold分数...")
            
            # 获取所有expert的激活次数
            expert_stats = self.expert_tracker.get_expert_stats()
            
            if not expert_stats:
                logger.warning("没有expert统计数据")
                return {}
            
            # 按层分组统计
            layer_experts = {}
            for key, info in expert_stats.items():
                layer_id = info['layer_id']
                if layer_id not in layer_experts:
                    layer_experts[layer_id] = []
                layer_experts[layer_id].append({
                    'expert_id': info['expert_id'],
                    'activation_count': info['activation_count'],
                    'total_tokens': info['total_tokens_processed']
                })
            
            # 计算每层的hot-cold分数
            hot_cold_analysis = {}
            for layer_id, experts in layer_experts.items():
                if not experts:
                    continue
                
                # 按激活次数排序
                experts.sort(key=lambda x: x['activation_count'], reverse=True)
                
                # 计算分数
                max_count = experts[0]['activation_count']
                min_count = experts[-1]['activation_count']
                
                layer_analysis = {
                    'layer_id': layer_id,
                    'total_experts': len(experts),
                    'max_activations': max_count,
                    'min_activations': min_count,
                    'experts': {}
                }
                
                for expert in experts:
                    if max_count == min_count:
                        # 如果所有expert激活次数相同
                        hot_cold_score = 1.0
                    else:
                        # 线性插值计算分数
                        hot_cold_score = (expert['activation_count'] - min_count) / (max_count - min_count)
                    
                    layer_analysis['experts'][expert['expert_id']] = {
                        'activation_count': expert['activation_count'],
                        'total_tokens': expert['total_tokens'],
                        'hot_cold_score': round(hot_cold_score, 4)
                    }
                
                hot_cold_analysis[f'layer_{layer_id}'] = layer_analysis
            
            logger.info(f"✓ 计算完成，共 {len(hot_cold_analysis)} 层")
            return hot_cold_analysis
            
        except Exception as e:
            logger.error(f"计算hot-cold分数失败: {e}")
            return {}
    
    def export_expert_analysis(self):
        """导出expert分析结果"""
        try:
            if not self.expert_tracker:
                logger.warning("Expert tracker未初始化，跳过导出")
                return
            
            logger.info("导出expert分析结果...")
            
            # 计算hot-cold分数
            hot_cold_analysis = self.calculate_hot_cold_scores()
            
            # 获取详细统计
            expert_stats = self.expert_tracker.get_expert_stats()
            top_experts = self.expert_tracker.get_top_experts(20)
            
            # 构建完整报告
            report = {
                'export_time': time.time(),
                'export_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'summary': {
                    'total_experts': len(expert_stats),
                    'total_activations': len(self.expert_tracker.activation_history),
                    'total_requests': len(self.expert_tracker.request_history),
                    'test_results': self.test_results
                },
                'hot_cold_analysis': hot_cold_analysis,
                'expert_stats': expert_stats,
                'top_experts': top_experts
            }
            
            # 导出到文件
            output_file = "expert_analysis.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✓ Expert分析结果已导出到: {output_file}")
            
            # 显示摘要
            self.display_analysis_summary(report)
            
        except Exception as e:
            logger.error(f"导出expert分析失败: {e}")
    
    def display_analysis_summary(self, report: Dict[str, Any]):
        """显示分析摘要"""
        try:
            print("\n" + "=" * 60)
            print("Expert Analysis 摘要")
            print("=" * 60)
            
            summary = report['summary']
            print(f"总Expert数: {summary['total_experts']}")
            print(f"总激活次数: {summary['total_activations']}")
            print(f"总请求数: {summary['total_requests']}")
            print(f"测试数据数: {len(summary['test_results'])}")
            
            hot_cold_analysis = report['hot_cold_analysis']
            if hot_cold_analysis:
                print(f"\n层数: {len(hot_cold_analysis)}")
                
                for layer_key, layer_info in list(hot_cold_analysis.items())[:3]:  # 显示前3层
                    print(f"\n{layer_key}:")
                    print(f"  Expert数: {layer_info['total_experts']}")
                    print(f"  最大激活: {layer_info['max_activations']}")
                    print(f"  最小激活: {layer_info['min_activations']}")
                    
                    # 显示前5个expert的分数
                    experts = list(layer_info['experts'].items())[:5]
                    for expert_id, expert_info in experts:
                        print(f"    Expert {expert_id}: {expert_info['hot_cold_score']:.4f} ({expert_info['activation_count']})")
            
            print("\n" + "=" * 60)
            
        except Exception as e:
            logger.error(f"显示摘要失败: {e}")
    
    def cleanup(self):
        """清理资源"""
        try:
            logger.info("开始清理资源...")
            
            # 导出expert分析结果
            self.export_expert_analysis()
            
            # 停止SGLang服务
            if self.sglang_process:
                logger.info("停止SGLang服务...")
                try:
                    os.killpg(os.getpgid(self.sglang_process.pid), signal.SIGTERM)
                    self.sglang_process.wait(timeout=10)
                except:
                    try:
                        os.killpg(os.getpgid(self.sglang_process.pid), signal.SIGKILL)
                    except:
                        pass
                
                logger.info("✓ SGLang服务已停止")
            
            logger.info("✓ 资源清理完成")
            
        except Exception as e:
            logger.error(f"清理资源失败: {e}")
    
    def run(self):
        """运行主流程"""
        try:
            logger.info("=" * 60)
            logger.info("Expert Tracking 完整启动器")
            logger.info("=" * 60)
            
            # 1. 启动SGLang服务
            if not self.start_sglang_service():
                logger.error("无法启动SGLang服务")
                return
            
            # 2. 启用expert tracking
            if not self.enable_expert_tracking():
                logger.error("无法启用Expert Tracking")
                return
            
            # 3. 加载ShareGPT数据集
            dataset = self.load_sharegpt_dataset()
            
            # 4. 使用数据集测试模型
            self.test_with_sharegpt_data(dataset)
            
            # 5. 等待一段时间，让expert tracker收集更多数据
            logger.info("等待expert tracker收集数据...")
            time.sleep(10)
            
            # 6. 导出分析结果
            self.export_expert_analysis()
            
            logger.info("✓ 所有流程完成！")
            
            # 等待用户中断或自动退出
            logger.info("按 Ctrl+C 退出...")
            while not self.shutdown_event.is_set():
                time.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("收到用户中断信号")
        except Exception as e:
            logger.error(f"运行过程中出现错误: {e}")
        finally:
            self.cleanup()


def main():
    """主函数"""
    launcher = ExpertTrackingLauncher()
    launcher.run()


if __name__ == "__main__":
    main()
