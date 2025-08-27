#!/usr/bin/env python3
"""
SGLang内部状态API
提供内部并行状态和expert分布信息的查询接口
使用Python内置的http.server，无需Flask依赖
"""

import json
import logging
import threading
import time
from typing import Dict, Any, Optional
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import os

# 尝试导入sglang相关模块
try:
    from sglang.srt.distributed.parallel_state import (
        get_internal_parallel_state,
        get_all_parallel_groups_info,
        get_environment_info
    )
    SGLANG_AVAILABLE = True
except ImportError:
    SGLANG_AVAILABLE = False
    # 提供模拟函数
    def get_internal_parallel_state():
        return {"error": "SGLang not available"}
    
    def get_all_parallel_groups_info():
        return {"error": "SGLang not available"}
    
    def get_environment_info():
        return {"error": "SGLang not available"}

logger = logging.getLogger(__name__)

class InternalStateHandler(BaseHTTPRequestHandler):
    """内部状态API处理器"""
    
    def do_GET(self):
        """处理GET请求"""
        try:
            parsed_url = urlparse(self.path)
            path = parsed_url.path
            
            # 设置CORS头
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type')
            self.end_headers()
            
            if path == '/internal/parallel_state':
                response = self.get_parallel_state()
            elif path == '/internal/parallel_groups':
                response = self.get_parallel_groups()
            elif path == '/internal/environment':
                response = self.get_environment()
            elif path == '/internal/expert_distribution':
                response = self.get_expert_distribution()
            elif path == '/internal/model_state':
                response = self.get_model_state()
            elif path == '/internal/health':
                response = self.health_check()
            else:
                response = {
                    'status': 'error',
                    'error': f'Unknown endpoint: {path}'
                }
            
            self.wfile.write(json.dumps(response, ensure_ascii=False).encode('utf-8'))
            
        except Exception as e:
            logger.error(f"Error handling request: {e}")
            error_response = {
                'status': 'error',
                'error': str(e)
            }
            self.wfile.write(json.dumps(error_response, ensure_ascii=False).encode('utf-8'))
    
    def do_OPTIONS(self):
        """处理OPTIONS请求（CORS预检）"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def log_message(self, format, *args):
        """重写日志方法，使用logger"""
        logger.info(f"{self.address_string()} - {format % args}")
    
    def get_parallel_state(self) -> Dict[str, Any]:
        """获取并行状态信息"""
        try:
            if SGLANG_AVAILABLE:
                state = get_internal_parallel_state()
                return {
                    'status': 'success',
                    'data': state
                }
            else:
                return {
                    'status': 'error',
                    'error': 'SGLang not available'
                }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def get_parallel_groups(self) -> Dict[str, Any]:
        """获取所有并行组信息"""
        try:
            if SGLANG_AVAILABLE:
                groups = get_all_parallel_groups_info()
                return {
                    'status': 'success',
                    'data': groups
                }
            else:
                return {
                    'status': 'error',
                    'error': 'SGLang not available'
                }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def get_environment(self) -> Dict[str, Any]:
        """获取环境信息"""
        try:
            if SGLANG_AVAILABLE:
                env = get_environment_info()
                return {
                    'status': 'success',
                    'data': env
                }
            else:
                # 提供基本的环境信息
                env = {
                    'single_expert_mode': os.environ.get('SINGLE_EXPERT_MODE', 'unknown'),
                    'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES', ''),
                    'sglang_disable_marlin': os.environ.get('SGLANG_DISABLE_MARLIN', ''),
                    'sgl_disable_awq_marlin': os.environ.get('SGL_DISABLE_AWQ_MARLIN', ''),
                    'sglang_disable_sgl_kernel': os.environ.get('SGLANG_DISABLE_SGL_KERNEL', ''),
                    'torch_distributed_backend': 'not_available',
                    'torch_distributed_world_size': 0,
                    'torch_distributed_rank': 0
                }
                return {
                    'status': 'success',
                    'data': env
                }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def get_expert_distribution(self) -> Dict[str, Any]:
        """获取expert分布信息"""
        return {
            'status': 'success',
            'data': {
                'message': 'Expert distribution info needs to be accessed through model instances',
                'suggestion': 'Use /internal/model_state endpoint instead'
            }
        }
    
    def get_model_state(self) -> Dict[str, Any]:
        """获取模型状态信息"""
        return {
            'status': 'success',
            'data': {
                'message': 'Model state info needs to be accessed through model instances',
                'suggestion': 'Use the enhanced deployment verifier instead'
            }
        }
    
    def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        return {
            'status': 'success',
            'message': 'Internal state API is running',
            'sglang_available': SGLANG_AVAILABLE,
            'timestamp': time.time()
        }

def start_internal_api(host='127.0.0.1', port=8082):
    """启动内部状态API服务器"""
    try:
        server = HTTPServer((host, port), InternalStateHandler)
        logger.info(f"Starting internal state API server on {host}:{port}")
        
        # 在新线程中启动服务器
        def run_server():
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                logger.info("Shutting down internal state API server")
                server.shutdown()
        
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        
        # 等待服务器启动
        time.sleep(1)
        
        logger.info(f"Internal state API server started successfully on {host}:{port}")
        return server
        
    except Exception as e:
        logger.error(f"Failed to start internal state API server: {e}")
        return None

def stop_internal_api(server):
    """停止内部状态API服务器"""
    if server:
        try:
            server.shutdown()
            logger.info("Internal state API server stopped")
        except Exception as e:
            logger.error(f"Error stopping internal state API server: {e}")

if __name__ == '__main__':
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    # 启动服务器
    server = start_internal_api()
    
    if server:
        try:
            # 保持服务器运行
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            stop_internal_api(server)
    else:
        logger.error("Failed to start server")
