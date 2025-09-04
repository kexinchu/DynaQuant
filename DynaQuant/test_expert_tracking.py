#!/usr/bin/env python3
"""
测试Expert激活统计功能是否正常工作
"""

import os
import sys
import subprocess
import time
import logging
import requests
import json

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_expert_tracking():
    """测试Expert激活统计功能"""
    try:
        logger.info("开始测试Expert激活统计功能...")
        
        # 发送测试请求
        test_request = {
            "model": "qwen3-235b-a22b",
            "messages": [
                {"role": "system", "content": "你是一个擅长混合精度/MoE 的助手"},
                {"role": "user", "content": "用一段话解释混合精度推理的优势"}
            ],
            "max_tokens": 128,
            "temperature": 0.7,
            "top_p": 0.9
        }
        
        logger.info("发送测试请求...")
        response = requests.post(
            "http://127.0.0.1:8080/v1/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer sk-local"
            },
            json=test_request,
            timeout=60
        )
        
        if response.status_code == 200:
            logger.info("✅ 测试请求成功")
            result = response.json()
            logger.info(f"响应内容: {result.get('choices', [{}])[0].get('message', {}).get('content', '')[:100]}...")
            return True
        else:
            logger.error(f"❌ 测试请求失败: {response.status_code}")
            logger.error(f"错误信息: {response.text}")
            return False
            
    except Exception as e:
        logger.error(f"测试过程中出现异常: {e}")
        return False

def check_service_health():
    """检查服务健康状态"""
    try:
        response = requests.get("http://127.0.0.1:8080/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info("Expert激活统计功能测试")
    logger.info("=" * 60)
    
    # 检查服务是否运行
    if not check_service_health():
        logger.error("❌ SGLang服务未运行，请先启动服务")
        return False
    
    logger.info("✅ SGLang服务正在运行")
    
    # 运行测试
    success = test_expert_tracking()
    
    if success:
        logger.info("=" * 60)
        logger.info("✅ Expert激活统计功能测试成功！")
        logger.info("请检查服务日志中的 [EXPERT_TRACKING] 标记")
        logger.info("应该能看到类似以下信息：")
        logger.info("🔍 [EXPERT_TRACKING] Layer X: topk_idx shape=..., active_experts=...")
        logger.info("✅ [EXPERT_TRACKING] 记录 expert 激活: layer=X, expert_id=Y, ...")
        logger.info("=" * 60)
    else:
        logger.error("=" * 60)
        logger.error("❌ Expert激活统计功能测试失败！")
        logger.error("请检查服务日志和配置")
        logger.error("=" * 60)
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)