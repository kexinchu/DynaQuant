from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uvicorn
import yaml
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging

from .mixed_precision_model import MixedPrecisionTransformerModel
from .expert_activation_tracker import get_global_tracker

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 请求模型
class GenerationRequest(BaseModel):
    prompt: str = Field(..., description="输入提示文本")
    max_new_tokens: int = Field(default=512, description="最大生成token数")
    temperature: float = Field(default=0.7, description="温度参数")
    top_p: float = Field(default=0.9, description="top-p采样参数")
    top_k: int = Field(default=50, description="top-k采样参数")
    do_sample: bool = Field(default=True, description="是否使用采样")

class BatchGenerationRequest(BaseModel):
    prompts: List[str] = Field(..., description="输入提示文本列表")
    max_new_tokens: int = Field(default=512, description="最大生成token数")
    temperature: float = Field(default=0.7, description="温度参数")
    top_p: float = Field(default=0.9, description="top-p采样参数")
    top_k: int = Field(default=50, description="top-k采样参数")
    do_sample: bool = Field(default=True, description="是否使用采样")

class WeightMappingRequest(BaseModel):
    weight_mapping: Dict[str, str] = Field(..., description="权重映射配置")

class ExpertStatsRequest(BaseModel):
    top_k: int = Field(default=10, description="返回前k个激活最多的专家")
    minutes: int = Field(default=5, description="最近几分钟的激活记录")

# 响应模型
class GenerationResponse(BaseModel):
    generated_text: str
    prompt: str
    generation_time: float
    model_info: Dict[str, Any]

class BatchGenerationResponse(BaseModel):
    generated_texts: List[str]
    prompts: List[str]
    generation_time: float
    model_info: Dict[str, Any]

class ModelInfoResponse(BaseModel):
    model_info: Dict[str, Any]
    weight_info: Dict[str, Any]

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    timestamp: float


class MixedPrecisionAPIServer:
    """混合精度模型API服务器"""
    
    def __init__(self, config_path: str):
        """
        初始化API服务器
        
        Args:
            config_path: 配置文件路径
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.server_config = self.config['model']['server']
        
        # 初始化模型
        self.model = None
        self.model_loaded = False
        
        # 创建线程池
        self.executor = ThreadPoolExecutor(max_workers=self.server_config['max_workers'])
        
        # 创建FastAPI应用
        self.app = FastAPI(
            title="Mixed Precision Transformer API",
            description="支持混合精度推理的Transformer模型API",
            version="1.0.0"
        )
        
        # 添加CORS中间件
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # 注册路由
        self._register_routes()
        
        # 初始化模型
        self._initialize_model()
    
    def _initialize_model(self):
        """初始化模型"""
        try:
            logger.info("Initializing mixed precision model...")
            self.model = MixedPrecisionTransformerModel("config/model_config.yaml")
            self.model_loaded = True
            logger.info("Model initialized successfully!")
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            self.model_loaded = False
    
    def _register_routes(self):
        """注册API路由"""
        
        @self.app.get("/", response_model=HealthResponse)
        async def health_check():
            """健康检查"""
            return HealthResponse(
                status="healthy" if self.model_loaded else "unhealthy",
                model_loaded=self.model_loaded,
                device=str(self.model.device) if self.model else "unknown",
                timestamp=time.time()
            )
        
        @self.app.post("/generate", response_model=GenerationResponse)
        async def generate_text(request: GenerationRequest):
            """单次文本生成"""
            if not self.model_loaded:
                raise HTTPException(status_code=503, detail="Model not loaded")
            
            try:
                start_time = time.time()
                
                # 在线程池中执行生成任务
                loop = asyncio.get_event_loop()
                generated_text = await loop.run_in_executor(
                    self.executor,
                    self.model.generate,
                    request.prompt,
                    request.max_new_tokens,
                    request.temperature,
                    request.top_p,
                    request.top_k,
                    request.do_sample
                )
                
                generation_time = time.time() - start_time
                
                return GenerationResponse(
                    generated_text=generated_text,
                    prompt=request.prompt,
                    generation_time=generation_time,
                    model_info=self.model.get_model_info()
                )
                
            except Exception as e:
                logger.error(f"Generation error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/batch_generate", response_model=BatchGenerationResponse)
        async def batch_generate_text(request: BatchGenerationRequest):
            """批量文本生成"""
            if not self.model_loaded:
                raise HTTPException(status_code=503, detail="Model not loaded")
            
            try:
                start_time = time.time()
                
                # 在线程池中执行批量生成任务
                loop = asyncio.get_event_loop()
                generated_texts = await loop.run_in_executor(
                    self.executor,
                    self.model.batch_generate,
                    request.prompts,
                    request.max_new_tokens,
                    request.temperature,
                    request.top_p,
                    request.top_k,
                    request.do_sample
                )
                
                generation_time = time.time() - start_time
                
                return BatchGenerationResponse(
                    generated_texts=generated_texts,
                    prompts=request.prompts,
                    generation_time=generation_time,
                    model_info=self.model.get_model_info()
                )
                
            except Exception as e:
                logger.error(f"Batch generation error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/model_info", response_model=ModelInfoResponse)
        async def get_model_info():
            """获取模型信息"""
            if not self.model_loaded:
                raise HTTPException(status_code=503, detail="Model not loaded")
            
            try:
                model_info = self.model.get_model_info()
                weight_info = self.model.weight_loader.get_weight_info()
                
                return ModelInfoResponse(
                    model_info=model_info,
                    weight_info=weight_info
                )
                
            except Exception as e:
                logger.error(f"Get model info error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/update_weight_mapping")
        async def update_weight_mapping(request: WeightMappingRequest):
            """更新权重映射配置"""
            if not self.model_loaded:
                raise HTTPException(status_code=503, detail="Model not loaded")
            
            try:
                self.model.update_weight_mapping(request.weight_mapping)
                return {"message": "Weight mapping updated successfully"}
                
            except Exception as e:
                logger.error(f"Update weight mapping error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/reload_weights")
        async def reload_weights():
            """重新加载权重"""
            if not self.model_loaded:
                raise HTTPException(status_code=503, detail="Model not loaded")
            
            try:
                # 在线程池中执行权重重载任务
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(self.executor, self.model.reload_weights)
                
                return {"message": "Weights reloaded successfully"}
                
            except Exception as e:
                logger.error(f"Reload weights error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/expert_stats")
        async def get_expert_stats():
            """获取专家激活统计"""
            if not self.model_loaded:
                raise HTTPException(status_code=503, detail="Model not loaded")
            
            try:
                tracker = get_global_tracker()
                
                stats = {
                    "summary": tracker.get_summary_stats(),
                    "layer_stats": tracker.get_layer_stats(),
                    "top_experts": tracker.get_top_experts(10),
                    "all_experts": tracker.get_all_expert_info()
                }
                
                return stats
                
            except Exception as e:
                logger.error(f"Get expert stats error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/expert_stats")
        async def get_expert_stats_with_params(request: ExpertStatsRequest):
            """获取专家激活统计（带参数）"""
            if not self.model_loaded:
                raise HTTPException(status_code=503, detail="Model not loaded")
            
            try:
                tracker = get_global_tracker()
                
                stats = {
                    "summary": tracker.get_summary_stats(),
                    "layer_stats": tracker.get_layer_stats(),
                    "top_experts": tracker.get_top_experts(request.top_k),
                    "recent_activations": tracker.get_recent_activations(request.minutes),
                    "all_experts": tracker.get_all_expert_info()
                }
                
                return stats
                
            except Exception as e:
                logger.error(f"Get expert stats error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/reset_expert_stats")
        async def reset_expert_stats():
            """重置专家激活统计"""
            if not self.model_loaded:
                raise HTTPException(status_code=503, detail="Model not loaded")
            
            try:
                tracker = get_global_tracker()
                tracker.reset()
                
                return {"message": "Expert statistics reset successfully"}
                
            except Exception as e:
                logger.error(f"Reset expert stats error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/export_expert_stats")
        async def export_expert_stats():
            """导出专家激活统计"""
            if not self.model_loaded:
                raise HTTPException(status_code=503, detail="Model not loaded")
            
            try:
                tracker = get_global_tracker()
                file_path = f"expert_stats_{int(time.time())}.json"
                tracker.export_stats(file_path)
                
                return {"message": f"Expert statistics exported to {file_path}"}
                
            except Exception as e:
                logger.error(f"Export expert stats error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
    
    def run(self):
        """运行服务器"""
        uvicorn.run(
            self.app,
            host=self.server_config['host'],
            port=self.server_config['port'],
            log_level="info"
        )


def create_app(config_path: str = "config/model_config.yaml"):
    """创建FastAPI应用实例"""
    server = MixedPrecisionAPIServer(config_path)
    return server.app


if __name__ == "__main__":
    # 直接运行服务器
    server = MixedPrecisionAPIServer("config/model_config.yaml")
    server.run()
