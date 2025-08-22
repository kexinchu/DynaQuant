import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Dict, Any, Optional, List
import yaml
from .weight_loader import MixedPrecisionWeightLoader


class MixedPrecisionTransformerModel:
    """支持混合精度推理的Transformer模型"""
    
    def __init__(self, config_path: str):
        """
        初始化混合精度模型
        
        Args:
            config_path: 配置文件路径
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.model_config = self.config['model']
        self.inference_config = self.model_config['inference']
        
        # 初始化权重加载器
        self.weight_loader = MixedPrecisionWeightLoader(config_path)
        
        # 加载模型和分词器
        self.model = None
        self.tokenizer = None
        self.device = None
        
        self._load_model()
    
    def _load_model(self):
        """加载模型和分词器"""
        print("Loading model and tokenizer...")
        
        # 加载配置
        model_name = self.model_config['name']
        base_path = self.model_config['base_path']
        
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_path,
            trust_remote_code=True
        )
        
        # 加载模型配置
        config = AutoConfig.from_pretrained(
            base_path,
            trust_remote_code=True
        )
        
        # 设置推理配置
        config.max_seq_length = self.inference_config['max_seq_length']
        
        # 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            base_path,
            config=config,
            torch_dtype=getattr(torch, self.inference_config['dtype']),
            device_map=self.inference_config['device_map'],
            trust_remote_code=True,
            load_in_8bit=self.inference_config['load_in_8bit'],
            load_in_4bit=self.inference_config['load_in_4bit']
        )
        
        # 加载混合精度权重
        self.weight_loader.load_model_weights(self.model)
        
        # 设置设备
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        
        # 设置为评估模式
        self.model.eval()
        
        print(f"Model loaded successfully on device: {self.device}")
    
    def _prepare_inputs(self, text: str, max_length: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """准备模型输入"""
        if max_length is None:
            max_length = self.inference_config['max_seq_length']
        
        # 编码输入文本
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding=True
        )
        
        # 移动到设备
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        return inputs
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None
    ) -> str:
        """
        生成文本
        
        Args:
            prompt: 输入提示
            max_new_tokens: 最大生成token数
            temperature: 温度参数
            top_p: top-p采样参数
            top_k: top-k采样参数
            do_sample: 是否使用采样
            pad_token_id: 填充token ID
            eos_token_id: 结束token ID
            
        Returns:
            生成的文本
        """
        # 准备输入
        inputs = self._prepare_inputs(prompt)
        
        # 设置生成参数
        generation_config = {
            'max_new_tokens': max_new_tokens,
            'temperature': temperature,
            'top_p': top_p,
            'top_k': top_k,
            'do_sample': do_sample,
            'pad_token_id': pad_token_id or self.tokenizer.pad_token_id,
            'eos_token_id': eos_token_id or self.tokenizer.eos_token_id,
            'use_cache': True
        }
        
        # 生成文本
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **generation_config
            )
        
        # 解码输出
        generated_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        return generated_text
    
    def batch_generate(
        self,
        prompts: List[str],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True
    ) -> List[str]:
        """
        批量生成文本
        
        Args:
            prompts: 输入提示列表
            max_new_tokens: 最大生成token数
            temperature: 温度参数
            top_p: top-p采样参数
            top_k: top-k采样参数
            do_sample: 是否使用采样
            
        Returns:
            生成的文本列表
        """
        # 批量编码输入
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            max_length=self.inference_config['max_seq_length'],
            truncation=True,
            padding=True
        )
        
        # 移动到设备
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 设置生成参数
        generation_config = {
            'max_new_tokens': max_new_tokens,
            'temperature': temperature,
            'top_p': top_p,
            'top_k': top_k,
            'do_sample': do_sample,
            'pad_token_id': self.tokenizer.pad_token_id,
            'eos_token_id': self.tokenizer.eos_token_id,
            'use_cache': True
        }
        
        # 批量生成
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **generation_config
            )
        
        # 解码输出
        generated_texts = []
        for i, output in enumerate(outputs):
            input_length = inputs['input_ids'][i].shape[0]
            generated_text = self.tokenizer.decode(
                output[input_length:],
                skip_special_tokens=True
            )
            generated_texts.append(generated_text)
        
        return generated_texts
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        info = {
            'model_name': self.model_config['name'],
            'device': str(self.device),
            'dtype': self.inference_config['dtype'],
            'max_seq_length': self.inference_config['max_seq_length'],
            'max_batch_size': self.inference_config['max_batch_size'],
            'weight_info': self.weight_loader.get_weight_info()
        }
        return info
    
    def update_weight_mapping(self, new_mapping: Dict[str, str]):
        """
        更新权重映射配置
        
        Args:
            new_mapping: 新的权重映射
        """
        self.weight_loader.weight_mapping.update(new_mapping)
        print("Weight mapping updated. Please reload the model to apply changes.")
    
    def reload_weights(self):
        """重新加载权重"""
        print("Reloading mixed precision weights...")
        self.weight_loader.load_model_weights(self.model)
        print("Weights reloaded successfully!")
