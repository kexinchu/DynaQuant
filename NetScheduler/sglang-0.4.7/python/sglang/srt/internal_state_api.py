#!/usr/bin/env python3
"""
SGLang内部状态API
提供内部并行状态和expert分布信息的查询接口
"""

import json
import logging
from typing import Dict, Any, Optional
from flask import Flask, jsonify, request

from sglang.srt.distributed.parallel_state import (
    get_internal_parallel_state,
    get_all_parallel_groups_info,
    get_environment_info
)

logger = logging.getLogger(__name__)

app = Flask(__name__)

@app.route('/internal/parallel_state', methods=['GET'])
def get_parallel_state():
    """获取并行状态信息"""
    try:
        state = get_internal_parallel_state()
        return jsonify({
            'status': 'success',
            'data': state
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/internal/parallel_groups', methods=['GET'])
def get_parallel_groups():
    """获取所有并行组信息"""
    try:
        groups = get_all_parallel_groups_info()
        return jsonify({
            'status': 'success',
            'data': groups
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/internal/environment', methods=['GET'])
def get_environment():
    """获取环境信息"""
    try:
        env = get_environment_info()
        return jsonify({
            'status': 'success',
            'data': env
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/internal/expert_distribution', methods=['GET'])
def get_expert_distribution():
    """获取expert分布信息"""
    try:
        # 这里需要从模型实例获取expert分布信息
        # 由于需要访问模型实例，这个功能可能需要通过其他方式实现
        return jsonify({
            'status': 'success',
            'data': {
                'message': 'Expert distribution info needs to be accessed through model instances',
                'suggestion': 'Use /internal/model_state endpoint instead'
            }
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/internal/model_state', methods=['GET'])
def get_model_state():
    """获取模型状态信息"""
    try:
        # 这里需要访问模型实例
        # 由于sglang的架构，这个功能可能需要通过其他方式实现
        return jsonify({
            'status': 'success',
            'data': {
                'message': 'Model state info needs to be accessed through model instances',
                'suggestion': 'Use the enhanced deployment verifier instead'
            }
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/internal/health', methods=['GET'])
def health_check():
    """健康检查"""
    return jsonify({
        'status': 'success',
        'message': 'Internal state API is running'
    })

def start_internal_api(host='127.0.0.1', port=8082):
    """启动内部状态API服务器"""
    logger.info(f"Starting internal state API server on {host}:{port}")
    app.run(host=host, port=port, debug=False)

if __name__ == '__main__':
    start_internal_api()
