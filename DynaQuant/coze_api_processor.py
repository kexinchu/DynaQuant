#!/usr/bin/env python3
"""
Coze API 结果解析程序
读取模型输出文件，请求远端服务，解析结果并提取结构化数据
"""

import json
import requests
import argparse
import os
import re
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CozeAPIClient:
    """Coze API 客户端"""
    
    def __init__(self, api_key: str, workflow_id: str):
        """
        初始化客户端
        
        Args:
            api_key: Coze API密钥
            workflow_id: 工作流ID
        """
        self.api_key = api_key
        self.workflow_id = workflow_id
        self.base_url = "https://api.coze.cn/v1/workflow/stream_run"
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        })
    
    def send_request(self, content: str, answer: str) -> Optional[Dict[str, Any]]:
        """
        发送请求到Coze API
        
        Args:
            content: 内容参数
            answer: 答案参数
            
        Returns:
            API响应结果
        """
        data = {
            "workflow_id": self.workflow_id,
            "parameters": {
                "content": content,
                "answer": answer
            }
        }
        
        # try:
        logger.info(f"发送请求到Coze API，工作流ID: {self.workflow_id}")
        response = self.session.post(
            self.base_url,
            json=data,
            stream=True,
            timeout=300
        )
        
        if response.status_code == 200:
            result = self._parse_sse_response(response)
            if not result:
                logger.warning("SSE解析失败，启用调试模式")
            return result
        else:
            logger.error(f"API请求失败，状态码: {response.status_code}")
            logger.error(f"响应内容: {response.text}")
            return None
                
        # except Exception as e:
        #     logger.error(f"请求异常: {e}")
        #     return None
    
    def _parse_sse_response(self, response: requests.Response) -> Optional[Dict[str, Any]]:
        """
        解析SSE流式响应
        
        Args:
            response: 响应对象
            
        Returns:
            解析后的结果
        """
        # try:
        # 读取完整的响应内容，避免按行分割破坏JSON结构
        raw_content = response.content.decode('utf-8', errors='ignore')
        logger.debug(f"原始响应长度: {len(raw_content)}")

        result = {}

        # 按SSE事件分割（以id:开头）
        events = []
        current_event = {}
        current_data = ""
        
        # 逐行读取响应
        lines = raw_content.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            line_str = line.decode('utf-8') if isinstance(line, bytes) else line
            line_str = line_str.strip()
            
            if not line.startswith('data: '):
                continue
            elif "debug_url" in line:
                continue
            # 开始累积data内容
            current_data = line[6:]  # 移除 'data: ' 前缀
            print("line: " + current_data)

        # 保存最后一个事件
        if current_data:
            current_event['data'] = current_data
            events.append(current_event)
        
        logger.debug(f"解析到 {len(events)} 个SSE事件")
        print(f"解析到 {len(events)} 个SSE事件")
            
        # 查找结束事件
        for event in events:
            if event.get('data'):
                try:
                    # 解析外层JSON
                    s_clean = re.sub(r'[\x00-\x1f\x7f]', '', event['data'])
                    data_json = json.loads(s_clean)
                    
                    # 检查是否是结束事件
                    if data_json.get('node_type') == 'End' and data_json.get('node_is_finish'):
                        content = data_json.get('content', '')
                        if content:
                            # 解析content字段中的JSON字符串
                            try:
                                # 处理可能的转义字符
                                content_clean = content.replace('\\"', '"').replace('\\n', '\n')
                                s_clean = re.sub(r'[\x00-\x1f\x7f]', '', content_clean)
                                content_data = json.loads(s_clean)
                                result = content_data
                                logger.info("成功解析SSE响应内容")
                                break
                            except json.JSONDecodeError as e:
                                logger.warning(f"无法解析content字段的JSON内容: {e}")
                                logger.debug(f"Content内容: {content[:200]}...")
                                continue
                
                except json.JSONDecodeError as e:
                    logger.warning(f"无法解析SSE数据: {e}")
                    logger.debug(f"Data内容: {event['data'][:200]}...")
                    continue
    
        return result
            
        # except Exception as e:
        #     logger.error(f"解析SSE响应失败: {e}")
        #     import traceback
        #     traceback.print_exc()
        #     return None

class ModelOutputReader:
    """模型输出文件读取器"""
    
    def __init__(self):
        pass
    
    def read_jsonl_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        读取JSONL格式的模型输出文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            读取的数据列表
        """
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
            
            logger.info(f"从文件读取了 {len(data)} 条记录")
            return data
            
        except Exception as e:
            logger.error(f"读取文件失败: {e}")
            return []
    
    def extract_content_and_answer(self, record: Dict[str, Any]) -> tuple:
        """
        从记录中提取content和answer
        
        Args:
            record: 记录数据
            
        Returns:
            (content, answer) 元组
        """
        try:
            # 提取用户请求内容
            user_request = record.get('user_request', {})
            content = user_request.get('content', '')
            
            # 提取模型回答
            model_response = record.get('model_response', {})
            answer = ""
            
            if 'choices' in model_response and model_response['choices']:
                choice = model_response['choices'][0]
                if 'message' in choice and 'content' in choice['message']:
                    answer = choice['message']['content']
            
            # 如果没有找到标准格式的answer，尝试其他字段
            if not answer:
                if 'generated_text' in model_response:
                    answer = model_response['generated_text']
                elif 'response' in model_response:
                    answer = model_response['response']
            
            return content, answer
            
        except Exception as e:
            logger.error(f"提取content和answer失败: {e}")
            return "", ""


class ResultProcessor:
    """结果处理器"""
    
    def __init__(self):
        pass
    
    def extract_dimensions_and_scores(self, api_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        从API结果中提取dimension_name和score
        
        Args:
            api_result: API返回的结果
            
        Returns:
            提取的维度评分列表
        """
        dimensions = []
        
        try:
            output_list = api_result.get('outputList', [])
            
            for item in output_list:
                dimension_name = item.get('dimension_name', '')
                score = item.get('score', 0)
                reasoning_content = item.get('reasoning_content', '')
                
                if dimension_name and score is not None:
                    dimensions.append({
                        'dimension_name': dimension_name,
                        'score': score,
                        'reasoning_content': reasoning_content
                    })
            
            logger.info(f"提取了 {len(dimensions)} 个维度的评分")
            return dimensions
            
        except Exception as e:
            logger.error(f"提取维度和评分失败: {e}")
            return []
    
    def calculate_overall_score(self, dimensions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        计算总体评分
        
        Args:
            dimensions: 维度评分列表
            
        Returns:
            总体评分统计
        """
        if not dimensions:
            return {}
        
        total_score = sum(item['score'] for item in dimensions)
        avg_score = total_score / len(dimensions)
        max_score = max(item['score'] for item in dimensions)
        min_score = min(item['score'] for item in dimensions)
        
        # 统计各分数段的数量
        score_distribution = {}
        for item in dimensions:
            score = item['score']
            score_distribution[score] = score_distribution.get(score, 0) + 1
        
        return {
            'total_dimensions': len(dimensions),
            'total_score': total_score,
            'average_score': round(avg_score, 2),
            'max_score': max_score,
            'min_score': min_score,
            'score_distribution': score_distribution
        }


class DataExporter:
    """数据导出器"""
    
    def __init__(self):
        pass
    
    def export_to_jsonl(self, data: List[Dict[str, Any]], output_file: str):
        """
        导出数据到JSONL文件
        
        Args:
            data: 要导出的数据
            output_file: 输出文件路径
        """
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
            logger.info(f"数据已导出到: {output_file}")
            
        except Exception as e:
            logger.error(f"导出数据失败: {e}")
    
    def export_summary_report(self, summary_data: List[Dict[str, Any]], output_file: str):
        """
        导出摘要报告
        
        Args:
            summary_data: 摘要数据
            output_file: 输出文件路径
        """
        try:
            report = {
                'timestamp': datetime.now().isoformat(),
                'total_records': len(summary_data),
                'summary': summary_data
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            
            logger.info(f"摘要报告已导出到: {output_file}")
            
        except Exception as e:
            logger.error(f"导出摘要报告失败: {e}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Coze API 结果解析程序')
    parser.add_argument('--input', '-i', required=True, help='输入文件路径 (JSONL格式)')
    parser.add_argument('--output', '-o', default='coze_results.jsonl', help='输出文件路径')
    parser.add_argument('--summary', '-s', default='summary_report.json', help='摘要报告文件路径')
    parser.add_argument('--api-key', required=True, help='Coze API密钥')
    parser.add_argument('--workflow-id', required=True, help='工作流ID')
    parser.add_argument('--delay', type=float, default=2.0, help='请求间隔时间(秒)')
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not os.path.exists(args.input):
        logger.error(f"输入文件不存在: {args.input}")
        return
    
    # 创建各个组件
    reader = ModelOutputReader()
    client = CozeAPIClient(args.api_key, args.workflow_id)
    processor = ResultProcessor()
    exporter = DataExporter()
    
    # 读取模型输出文件
    logger.info(f"读取模型输出文件: {args.input}")
    records = reader.read_jsonl_file(args.input)
    
    if not records:
        logger.error("没有读取到有效的记录")
        return
    
    # 处理每条记录
    results = []
    summary_data = []
    
    logger.info(f"开始处理 {len(records)} 条记录...")
    
    for i, record in enumerate(records, 1):
        logger.info(f"处理第 {i}/{len(records)} 条记录")
        
        # 提取content和answer
        content, answer = reader.extract_content_and_answer(record)
        
        if not content or not answer:
            logger.warning(f"第 {i} 条记录缺少content或answer，跳过")
            continue
        
        # 发送请求到Coze API
        api_result = client.send_request(content, answer)
        
        if api_result:
            # 提取维度和评分
            dimensions = processor.extract_dimensions_and_scores(api_result)
            
            if dimensions:
                # 计算总体评分
                overall_score = processor.calculate_overall_score(dimensions)
                
                # 构建结果记录
                result_record = {
                    'timestamp': datetime.now().isoformat(),
                    'record_id': record.get('request_id', f'record_{i}'),
                    'original_record': {
                        'content': content,
                        'answer': answer,
                        'request_info': record.get('user_request', {}),
                        'model_response': record.get('model_response', {})
                    },
                    'coze_api_result': {
                        'dimensions': dimensions,
                        'overall_score': overall_score,
                        'raw_api_response': api_result
                    }
                }
                
                results.append(result_record)
                
                # 构建摘要数据
                summary_record = {
                    'record_id': result_record['record_id'],
                    'content_preview': content[:100] + '...' if len(content) > 100 else content,
                    'answer_preview': answer[:100] + '...' if len(answer) > 100 else answer,
                    'dimensions_count': len(dimensions),
                    'average_score': overall_score.get('average_score', 0),
                    'total_score': overall_score.get('total_score', 0)
                }
                summary_data.append(summary_record)
                
                logger.info(f"  成功 - 维度数: {len(dimensions)}, 平均分: {overall_score.get('average_score', 0)}")
            else:
                logger.warning(f"  未提取到有效的维度评分")
        else:
            logger.error(f"  API请求失败")
        
        # 请求间隔
        if i < len(records):
            import time
            time.sleep(args.delay)
    
    # 导出结果
    if results:
        exporter.export_to_jsonl(results, args.output)
        exporter.export_summary_report(summary_data, args.summary)
        
        logger.info("=" * 60)
        logger.info("处理完成摘要:")
        logger.info(f"  总记录数: {len(records)}")
        logger.info(f"  成功处理: {len(results)}")
        logger.info(f"  失败记录: {len(records) - len(results)}")
        logger.info(f"  结果文件: {args.output}")
        logger.info(f"  摘要报告: {args.summary}")
        logger.info("=" * 60)
    else:
        logger.error("没有成功处理任何记录")


if __name__ == "__main__":
    main()
