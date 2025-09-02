#!/usr/bin/env python3
"""
Qwen3-235B-A22B模型服务测试程序
读取测试数据文件，发送请求给模型，并记录结果到JSONL文件
"""

import json
import requests
import time
import argparse
import os
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class QwenServiceClient:
    """Qwen3-235B-A22B模型服务客户端"""
    
    def __init__(self, base_url: str = "http://127.0.0.1:8080", api_key: str = "sk-local"):
        """
        初始化客户端
        
        Args:
            base_url: API服务器地址
            api_key: API密钥
        """
        self.base_url = base_url
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}'
        })
    
    def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=10)
            return response.json()
        except Exception as e:
            logger.error(f"健康检查失败: {e}")
            return {"error": str(e)}
    
    def chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """
        发送聊天完成请求
        
        Args:
            messages: 消息列表，格式为 [{"role": "user", "content": "..."}]
            **kwargs: 其他参数如max_tokens, temperature等
            
        Returns:
            模型响应结果
        """
        data = {
            "model": "qwen3-235b-a22b",
            "messages": messages,
            **kwargs
        }
        
        try:
            start_time = time.time()
            response = self.session.post(
                f"{self.base_url}/v1/chat/completions",
                json=data,
                timeout=300  # 5分钟超时
            )
            end_time = time.time()
            
            if response.status_code == 200:
                result = response.json()
                result['request_time'] = end_time - start_time
                return result
            else:
                logger.error(f"请求失败，状态码: {response.status_code}")
                return {
                    "error": f"HTTP {response.status_code}",
                    "response": response.text,
                    "request_time": end_time - start_time
                }
                
        except Exception as e:
            logger.error(f"请求异常: {e}")
            return {"error": str(e)}
    
    def test_connection(self) -> bool:
        """测试连接"""
        try:
            health = self.health_check()
            if "error" not in health:
                logger.info("服务连接正常")
                return True
            else:
                logger.error("服务连接失败")
                return False
        except Exception as e:
            logger.error(f"连接测试失败: {e}")
            return False


class TestDataProcessor:
    """测试数据处理器"""
    
    def __init__(self, client: QwenServiceClient):
        self.client = client
    
    def read_txt_file(self, file_path: str) -> List[str]:
        """读取TXT文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f if line.strip()]
            logger.info(f"从TXT文件读取了 {len(lines)} 行数据")
            return lines
        except Exception as e:
            logger.error(f"读取TXT文件失败: {e}")
            return []
    
    def read_jsonl_file(self, file_path: str) -> List[Dict[str, Any]]:
        """读取JSONL文件"""
        try:
            data = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        try:
                            item = json.loads(line)
                            data.append(item)
                        except json.JSONDecodeError as e:
                            logger.warning(f"第{line_num}行JSON解析失败: {e}")
                            continue
            logger.info(f"从JSONL文件读取了 {len(data)} 条数据")
            return data
        except Exception as e:
            logger.error(f"读取JSONL文件失败: {e}")
            return []
    
    def process_txt_data(self, lines: List[str]) -> List[Dict[str, Any]]:
        """处理TXT数据，转换为标准格式"""
        data = []
        for i, line in enumerate(lines):
            data.append({
                "id": f"txt_{i+1:04d}",
                "type": "txt",
                "content": line,
                "messages": [
                    {"role": "user", "content": line}
                ]
            })
        return data
    
    def process_jsonl_data(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """处理JSONL数据，转换为标准格式"""
        data = []
        for i, item in enumerate(items):
            # 尝试从不同字段中提取内容
            content = None
            if isinstance(item, dict):
                if "content" in item:
                    content = item["content"]
                elif "text" in item:
                    content = item["text"]
                elif "prompt" in item:
                    content = item["prompt"]
                elif "question" in item:
                    content = item["question"]
                elif "input" in item:
                    content = item["input"]
                else:
                    # 如果都没有，使用整个item作为内容
                    content = str(item)
            else:
                content = str(item)
            
            data.append({
                "id": f"jsonl_{i+1:04d}",
                "type": "jsonl",
                "original_data": item,
                "content": content,
                "messages": [
                    {"role": "user", "content": content}
                ]
            })
        return data


class ResultRecorder:
    """结果记录器"""
    
    def __init__(self, output_file: str):
        self.output_file = output_file
        self.results = []
    
    def add_result(self, result: Dict[str, Any]):
        """添加结果"""
        self.results.append(result)
    
    def save_to_jsonl(self):
        """保存结果到JSONL文件"""
        try:
            with open(self.output_file, 'w', encoding='utf-8') as f:
                for result in self.results:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
            logger.info(f"结果已保存到: {self.output_file}")
        except Exception as e:
            logger.error(f"保存结果失败: {e}")
    
    def get_summary(self) -> Dict[str, Any]:
        """获取结果摘要"""
        total_requests = len(self.results)
        successful_requests = len([r for r in self.results if "error" not in r])
        failed_requests = total_requests - successful_requests
        
        total_time = sum(r.get('request_time', 0) for r in self.results if "error" not in r)
        avg_time = total_time / successful_requests if successful_requests > 0 else 0
        
        return {
            "total_requests": total_requests,
            "successful_requests": successful_requests,
            "failed_requests": failed_requests,
            "success_rate": successful_requests / total_requests if total_requests > 0 else 0,
            "total_time": total_time,
            "average_time": avg_time
        }


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Qwen3-235B-A22B模型服务测试程序')
    parser.add_argument('--input', '-i', required=True, help='输入文件路径 (支持.txt和.jsonl)')
    parser.add_argument('--output', '-o', default='test_results.jsonl', help='输出文件路径')
    parser.add_argument('--host', default='127.0.0.1', help='服务主机地址')
    parser.add_argument('--port', default='8080', help='服务端口')
    parser.add_argument('--max-tokens', type=int, default=512, help='最大生成token数')
    parser.add_argument('--temperature', type=float, default=0.7, help='生成温度')
    parser.add_argument('--top-p', type=float, default=0.9, help='top-p采样参数')
    parser.add_argument('--delay', type=float, default=1.0, help='请求间隔时间(秒)')
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not os.path.exists(args.input):
        logger.error(f"输入文件不存在: {args.input}")
        return
    
    # 创建客户端
    client = QwenServiceClient(f"http://{args.host}:{args.port}")
    
    # 测试连接
    if not client.test_connection():
        logger.error("无法连接到模型服务，请确保服务已启动")
        return
    
    # 创建数据处理器
    processor = TestDataProcessor(client)
    
    # 读取数据
    if args.input.endswith('.txt'):
        lines = processor.read_txt_file(args.input)
        test_data = processor.process_txt_data(lines)
    elif args.input.endswith('.jsonl'):
        items = processor.read_jsonl_file(args.input)
        test_data = processor.process_jsonl_data(items)
    else:
        logger.error("不支持的文件格式，请使用.txt或.jsonl文件")
        return
    
    if not test_data:
        logger.error("没有读取到有效的测试数据")
        return
    
    # 创建结果记录器
    recorder = ResultRecorder(args.output)
    
    logger.info(f"开始处理 {len(test_data)} 条测试数据...")
    
    # 处理每条测试数据
    for i, data in enumerate(test_data, 1):
        logger.info(f"处理第 {i}/{len(test_data)} 条数据: {data['id']}")
        
        # 发送请求
        result = client.chat_completion(
            messages=data['messages'],
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p
        )
        
        # 构建完整结果记录
        record = {
            "timestamp": datetime.now().isoformat(),
            "request_id": data['id'],
            "request_type": data['type'],
            "user_request": {
                "content": data['content'],
                "messages": data['messages'],
                "parameters": {
                    "max_tokens": args.max_tokens,
                    "temperature": args.temperature,
                    "top_p": args.top_p
                }
            },
            "model_response": result,
            "processing_info": {
                "request_number": i,
                "total_requests": len(test_data)
            }
        }
        
        # 添加原始数据（如果是JSONL格式）
        if data['type'] == 'jsonl':
            record['original_data'] = data['original_data']
        
        recorder.add_result(record)
        
        # 打印简要结果
        if "error" not in result:
            logger.info(f"  成功 - 生成时间: {result.get('request_time', 0):.2f}s")
            if 'choices' in result and result['choices']:
                content = result['choices'][0].get('message', {}).get('content', '')
                logger.info(f"  回答: {content[:100]}{'...' if len(content) > 100 else ''}")
        else:
            logger.error(f"  失败 - {result.get('error', '未知错误')}")
        
        # 请求间隔
        if i < len(test_data):
            time.sleep(args.delay)
    
    # 保存结果
    recorder.save_to_jsonl()
    
    # 显示摘要
    summary = recorder.get_summary()
    logger.info("=" * 60)
    logger.info("测试完成摘要:")
    logger.info(f"  总请求数: {summary['total_requests']}")
    logger.info(f"  成功请求: {summary['successful_requests']}")
    logger.info(f"  失败请求: {summary['failed_requests']}")
    logger.info(f"  成功率: {summary['success_rate']:.2%}")
    logger.info(f"  总耗时: {summary['total_time']:.2f}秒")
    logger.info(f"  平均耗时: {summary['average_time']:.2f}秒")
    logger.info(f"  结果文件: {args.output}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
