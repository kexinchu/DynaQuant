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
    
    def __init__(self, max_workers: int = 16):
        self.sglang_process = None
        self.expert_tracker = None
        self.shutdown_event = threading.Event()
        self.test_results = []
        self.max_workers = max_workers  # 最大线程数
        
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
            
            # 设置环境变量启用expert tracking
            env = os.environ.copy()
            env['ENABLE_EXPERT_DISTRIBUTION_METRICS'] = 'true'
            env['ENABLE_MOE_TRACKING'] = 'true'
            env['ENABLE_EXPERT_TRACKING'] = 'true'
            
            logger.info("设置环境变量启用expert tracking...")
            
            # 启动服务
            self.sglang_process = subprocess.Popen(
                ["bash", script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid if hasattr(os, 'setsid') else None,
                env=env
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
            logger.info("✓ 全局Expert Tracker已初始化")
            
            # 尝试通过API启用expert distribution recording
            try:
                self.enable_expert_distribution_recording()
                logger.info("✓ 通过API启用expert distribution recording成功")
            except Exception as e:
                logger.warning(f"通过API启用失败: {e}")
            
            logger.info("✓ Expert Tracking功能已启用")

            # 测试记录功能
            self.expert_tracker.record_expert_activation(0, 1, activation_strength=1.0)
            test_stats = self.expert_tracker.get_expert_stats()
            logger.info(f"✓ 测试记录成功，当前统计: {len(test_stats)} 条")
            
            return True
            
        except Exception as e:
            logger.error(f"启用Expert Tracking失败: {e}")
            return False
    
    def enable_expert_distribution_recording(self):
        """通过API启用expert distribution recording"""
        try:
            # 等待服务完全启动
            time.sleep(5)
            
            # 发送请求启用expert distribution recording
            response = requests.post(
                "http://127.0.0.1:8080/start_expert_distribution_record",
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info("✓ Expert distribution recording已启动")
            else:
                logger.warning(f"启动expert distribution recording失败: {response.status_code}")
                
        except Exception as e:
            logger.warning(f"启用expert distribution recording失败: {e}")
    
    def load_sharegpt_dataset(self) -> List[Dict[str, Any]]:
        """加载ShareGPT数据集"""
        try:
            # 尝试从多个可能的路径加载数据集
            possible_paths = [
                "/root/code/ShareGPT_V3_unfiltered_cleaned_split.json",
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
            },
            {
                "id": "sample_004",
                "conversations": [
                    {"from": "human", "value": "请解释一下神经网络的工作原理"},
                    {"from": "gpt", "value": "神经网络是一种模仿生物神经系统的计算模型..."}
                ]
            },
            {
                "id": "sample_005",
                "conversations": [
                    {"from": "human", "value": "混合精度推理有什么优势？"},
                    {"from": "gpt", "value": "混合精度推理结合了不同精度的数值表示..."}
                ]
            }
        ]
        
        logger.info(f"创建了 {len(sample_data)} 条示例数据")
        return sample_data
    
    def test_with_sharegpt_data(self, dataset: List[Dict[str, Any]]):
        """使用ShareGPT数据集测试模型（多线程并行）"""
        try:
            logger.info("开始使用ShareGPT数据集测试模型（16线程并行）...")
            
            # 选择更多数据进行测试（多线程可以处理更多数据）
            test_data = dataset[:min(100, len(dataset))]  # 最多max_workers条
            
            # 创建线程池
            from concurrent.futures import ThreadPoolExecutor, as_completed
            import threading
            
            # 线程安全的测试结果列表
            test_results_lock = threading.Lock()
            
            def process_single_request(item_data):
                """处理单个请求的函数"""
                try:
                    item_id = item_data.get('id', 'unknown')
                    logger.debug(f"线程 {threading.current_thread().name} 处理: {item_id}")
                    
                    # 提取对话内容
                    conversations = item_data.get('conversations', [])
                    if not conversations:
                        return {
                            'id': item_id,
                            'status': 'skipped',
                            'response_length': 0,
                            'error': 'No conversations found'
                        }
                    
                    # 构建测试请求
                    messages = []
                    for conv in conversations:
                        if conv.get('from') == 'human':
                            messages.append({
                                "role": "user",
                                "content": conv.get('value', '')
                            })
                        elif conv.get('value', '') and conv.get('from') == 'gpt':
                            messages.append({
                                "role": "assistant", 
                                "content": conv.get('value', '')
                            })
                    
                    if not messages:
                        return {
                            'id': item_id,
                            'status': 'skipped',
                            'response_length': 0,
                            'error': 'No valid messages found'
                        }
                    
                    # 发送请求到模型
                    response = self.send_chat_request(messages)
                    if response:
                        logger.debug(f"线程 {threading.current_thread().name} - {item_id}: ✓ 请求成功，响应长度: {len(response)}")
                        return {
                            'id': item_id,
                            'status': 'success',
                            'response_length': len(response),
                            'error': None
                        }
                    else:
                        logger.warning(f"线程 {threading.current_thread().name} - {item_id}: ⚠ 请求失败")
                        return {
                            'id': item_id,
                            'status': 'failed',
                            'response_length': 0,
                            'error': 'Request failed'
                        }
                        
                except Exception as e:
                    logger.error(f"线程 {threading.current_thread().name} - {item_id}: ✗ 请求异常: {e}")
                    return {
                        'id': item_id,
                        'status': 'error',
                        'response_length': 0,
                        'error': str(e)
                    }
            
            # 使用配置的线程数并行处理
            max_workers = min(self.max_workers, len(test_data))
            logger.info(f"启动 {max_workers} 个线程并行处理 {len(test_data)} 条数据")
            
            with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="LLM_Worker") as executor:
                # 提交所有任务
                future_to_item = {
                    executor.submit(process_single_request, item): item 
                    for item in test_data
                }
                
                # 收集结果
                completed_count = 0
                for future in as_completed(future_to_item):
                    item = future_to_item[future]
                    try:
                        result = future.result()
                        with test_results_lock:
                            self.test_results.append(result)
                        
                        completed_count += 1
                        logger.info(f"进度: {completed_count}/{len(test_data)} - {result['id']}: {result['status']}")
                        
                        # 检查是否需要停止
                        if self.shutdown_event.is_set():
                            logger.info("收到停止信号，取消剩余任务")
                            break
                            
                    except Exception as e:
                        logger.error(f"获取任务结果失败: {e}")
                        with test_results_lock:
                            self.test_results.append({
                                'id': item.get('id', 'unknown'),
                                'status': 'error',
                                'response_length': 0,
                                'error': f'Result retrieval failed: {e}'
                            })
            
            # 统计测试结果
            self._print_test_summary()
            logger.info("✓ ShareGPT数据集多线程测试完成")
            
        except Exception as e:
            logger.error(f"多线程测试失败: {e}")
            import traceback
            traceback.print_exc()
    
    def _print_test_summary(self):
        """打印测试结果摘要"""
        try:
            if not self.test_results:
                logger.info("没有测试结果")
                return
            
            # 统计各种状态
            status_counts = {}
            total_response_length = 0
            success_count = 0
            
            for result in self.test_results:
                status = result.get('status', 'unknown')
                status_counts[status] = status_counts.get(status, 0) + 1
                
                if status == 'success':
                    success_count += 1
                    total_response_length += result.get('response_length', 0)
            
            # 打印统计信息
            logger.info("\n" + "=" * 50)
            logger.info("测试结果摘要")
            logger.info("=" * 50)
            logger.info(f"总测试数: {len(self.test_results)}")
            
            for status, count in status_counts.items():
                percentage = (count / len(self.test_results)) * 100
                logger.info(f"{status}: {count} ({percentage:.1f}%)")
            
            if success_count > 0:
                avg_response_length = total_response_length / success_count
                logger.info(f"成功请求平均响应长度: {avg_response_length:.1f} 字符")
            
            logger.info("=" * 50)
            
        except Exception as e:
            logger.error(f"打印测试摘要失败: {e}")
    
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
                timeout=600
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
                    # 'test_results': self.test_results
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
                    if hasattr(os, 'killpg') and hasattr(os, 'getpgid'):
                        os.killpg(os.getpgid(self.sglang_process.pid), signal.SIGTERM)
                        self.sglang_process.wait(timeout=10)
                    else:
                        # Windows兼容性处理
                        self.sglang_process.terminate()
                        self.sglang_process.wait(timeout=10)
                except Exception as e:
                    logger.warning(f"优雅关闭失败: {e}")
                    try:
                        if hasattr(os, 'killpg') and hasattr(os, 'getpgid'):
                            os.killpg(os.getpgid(self.sglang_process.pid), signal.SIGKILL)
                        else:
                            self.sglang_process.kill()
                    except Exception as e2:
                        logger.warning(f"强制关闭失败: {e2}")
                
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
    import argparse
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Expert Tracking 启动器')
    parser.add_argument('--workers', type=int, default=32, 
                       help='并行工作线程数 (默认: 32)')
    parser.add_argument('--test-data', type=int, default=None,
                       help='测试数据数量 (默认: 使用所有可用数据)')
    
    args = parser.parse_args()
    
    print(f"启动Expert Tracking，使用 {args.workers} 个并行线程")
    
    # 创建启动器实例
    launcher = ExpertTrackingLauncher(max_workers=args.workers)
    
    # 如果指定了测试数据数量，更新数据集选择
    if args.test_data:
        launcher.test_data_limit = args.test_data
    
    launcher.run()


if __name__ == "__main__":
    main()
