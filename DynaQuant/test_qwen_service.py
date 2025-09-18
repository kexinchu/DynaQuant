#!/usr/bin/env python3
"""
Qwen3-235B-A22B模型服务测试程序（多线程版）
读取测试数据文件，发送请求给模型，并记录结果到JSONL文件
"""

import json
import requests
import time
import argparse
import os
from typing import Dict, Any, List
from datetime import datetime
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import random
from load_requests import read_chatGPT

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class QwenServiceClient:
    """Qwen3-235B-A22B模型服务客户端（每线程独立 Session）"""

    def __init__(self, base_url: str = "http://127.0.0.1:8080", api_key: str = "sk-local", pool_size: int = 64):
        self.base_url = base_url
        self.api_key = api_key
        self._tls = threading.local()
        self._pool_size = pool_size

    def _get_session(self) -> requests.Session:
        s = getattr(self._tls, "session", None)
        if s is None:
            s = requests.Session()
            adapter = requests.adapters.HTTPAdapter(pool_connections=self._pool_size,
                                                    pool_maxsize=self._pool_size,
                                                    max_retries=0)
            s.mount("http://", adapter)
            s.mount("https://", adapter)
            s.headers.update({
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {self.api_key}'
            })
            self._tls.session = s
        return s

    def chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        data = {
            "model": "qwen3-235b-a22b",
            "messages": messages,
            **kwargs
        }
        session = self._get_session()
        try:
            start_time = time.time()
            resp = session.post(
                f"{self.base_url}/v1/chat/completions",
                json=data,
                timeout=900
            )
            end_time = time.time()

            if resp.status_code == 200:
                result = resp.json()
                result['request_time'] = end_time - start_time
                return result
            else:
                logger.error(f"请求失败，状态码: {resp.status_code}")
                return {
                    "error": f"HTTP {resp.status_code}",
                    "response": resp.text,
                    "request_time": end_time - start_time
                }
        except Exception as e:
            logger.error(f"请求异常: {e}")
            return {"error": str(e)}


class TestDataProcessor:
    """测试数据处理器"""

    def __init__(self, client: QwenServiceClient):
        self.client = client

    def read_txt_file(self, file_path: str) -> List[str]:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f if line.strip()]
            logger.info(f"从TXT文件读取了 {len(lines)} 行数据")
            return lines
        except Exception as e:
            logger.error(f"读取TXT文件失败: {e}")
            return []

    def read_jsonl_file(self, file_path: str) -> List[Dict[str, Any]]:
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
        data = []
        for i, item in enumerate(items):
            content = None
            if isinstance(item, dict):
                for k in ("content", "text", "prompt", "question", "input"):
                    if k in item:
                        content = item[k]
                        break
                if content is None:
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
        self.results: List[Dict[str, Any]] = []

    def add_result(self, result: Dict[str, Any]):
        self.results.append(result)

    def save_to_jsonl(self):
        try:
            # 按请求序号排序，保证输出文件顺序稳定
            results_sorted = sorted(
                self.results,
                key=lambda r: r.get("processing_info", {}).get("request_number", 0)
            )
            with open(self.output_file, 'w', encoding='utf-8') as f:
                for result in results_sorted:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
            logger.info(f"结果已保存到: {self.output_file}")
        except Exception as e:
            logger.error(f"保存结果失败: {e}")

    def get_summary(self) -> Dict[str, Any]:
        total_requests = len(self.results)
        successful_requests = len([r for r in self.results if "error" not in r.get("model_response", {})])
        failed_requests = total_requests - successful_requests
        total_time = sum(r.get('model_response', {}).get('request_time', 0)
                         for r in self.results if "error" not in r.get("model_response", {}))
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
    parser = argparse.ArgumentParser(description='Qwen3-235B-A22B模型服务测试程序（多线程版）')
    parser.add_argument('--input', '-i', required=True, help='输入文件路径 (支持.txt和.jsonl)')
    parser.add_argument('--output', '-o', default='test_results.jsonl', help='输出文件路径')
    parser.add_argument('--host', default='127.0.0.1', help='服务主机地址')
    parser.add_argument('--port', default='8080', help='服务端口')
    parser.add_argument('--max-tokens', type=int, default=4096, help='最大生成token数')
    parser.add_argument('--temperature', type=float, default=0.9, help='生成温度')
    parser.add_argument('--top-p', type=float, default=0.9, help='top-p采样参数')
    parser.add_argument('--delay', type=float, default=0.0, help='并发场景的抖动（每任务随机睡 0~delay 秒）')
    parser.add_argument('--workers', type=int, default=16, help='并发线程数（默认16）')
    args = parser.parse_args()

    # 检查输入文件
    if not os.path.exists(args.input):
        logger.error(f"输入文件不存在: {args.input}")
        return

    # 客户端
    client = QwenServiceClient(f"http://{args.host}:{args.port}")

    # 数据处理器
    processor = TestDataProcessor(client)

    # 读取数据
    if args.input.endswith('.txt'):
        lines = processor.read_txt_file(args.input)
        test_data = processor.process_txt_data(lines)
    elif args.input.endswith('.jsonl'):
        items = processor.read_jsonl_file(args.input)
        test_data = processor.process_jsonl_data(items)
    else:
        items = read_chatGPT(args.input)
        test_data = []
        for i, (context, answer, session_id) in enumerate(items):
            test_data.append({
                "id": f"txt_{i+1:04d}",
                "type": "json",
                "content": context,
                "messages": [
                    {"role": "user", "content": context}
                ]
            })

    if not test_data:
        logger.error("没有读取到有效的测试数据")
        return
    print(len(test_data))

    recorder = ResultRecorder(args.output)
    total = len(test_data)
    logger.info(f"开始并发处理 {total} 条测试数据，线程数={args.workers} ...")

    def worker(i: int, data: Dict[str, Any]) -> Dict[str, Any]:
        # 抖动，平滑突刺
        if args.delay and args.delay > 0:
            time.sleep(random.uniform(0, args.delay))

        result = client.chat_completion(
            messages=data['messages'],
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p
        )

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
                "total_requests": total
            }
        }
        if data['type'] == 'jsonl':
            record['original_data'] = data['original_data']
        return record

    # 并发提交
    futures = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for i, data in enumerate(test_data, 1):
            futures.append(executor.submit(worker, i, data))

        # 收集结果
        for fut in as_completed(futures):
            record = fut.result()
            result = record["model_response"]
            recorder.add_result(record)

            # 简要日志
            i = record["processing_info"]["request_number"]
            rid = record["request_id"]
            if "error" not in result:
                logger.info(f"[{i}/{total}] {rid} 成功 - 生成时间: {result.get('request_time', 0):.2f}s")
                if 'choices' in result and result['choices']:
                    content = result['choices'][0].get('message', {}).get('content', '')
                    logger.info(f"  回答: {content[:100]}{'...' if len(content) > 100 else ''}")
            else:
                logger.error(f"[{i}/{total}] {rid} 失败 - {result.get('error', '未知错误')}")

    # 保存与摘要
    recorder.save_to_jsonl()
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
