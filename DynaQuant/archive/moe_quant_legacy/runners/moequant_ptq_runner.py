"""
MoEQuant PTQ Runner - 完全遵循MoEQuant论文设计
支持 W8A8, W4A4, W2A2 三种量化精度

基于论文: MoEQuant - Expert-Balanced Self-Sampling + Affinity-Guided Quantization
"""

import argparse
import logging
import torch
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
import time

from ..quant.moequant_core import MoEQuantizer, MoEQuantConfig, create_moequant_quantizer
from ..quant.ebss import EBSSSampler, EBSSConfig
from ..quant.agq import AGQuantizer, AGQConfig
from ..models.load_moe import load_moe_model
from ..utils.safetensors_saver import save_quantized_model_safetensors

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MoEQuantPTQRunner:
    """
    MoEQuant PTQ 完整运行器

    实现流程:
    1. 加载模型
    2. EBSS生成专家均衡的校准数据
    3. 收集激活和affinity
    4. AGQ量化专家层
    5. 保存量化模型 (safetensors格式)
    """

    def __init__(
        self,
        model_path: str,
        precision: str = "w4a4",
        output_dir: str = "./output/moequant",
        device: str = "cuda",
        use_multi_gpu: bool = False,
        gpu_ids: Optional[List[int]] = None
    ):
        """
        Args:
            model_path: 模型路径
            precision: 量化精度 ("w8a8", "w4a4", "w2a2")
            output_dir: 输出目录
            device: 设备
            use_multi_gpu: 是否使用多GPU
            gpu_ids: 指定的GPU ID列表
        """
        self.model_path = model_path
        self.precision = precision.lower()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.use_multi_gpu = use_multi_gpu
        self.gpu_ids = gpu_ids

        # 组件
        self.model_loader = None
        self.model = None
        self.tokenizer = None
        self.quantizer = None
        self.ebss_sampler = None

        logger.info(f"初始化 MoEQuant PTQ Runner")
        logger.info(f"  模型: {model_path}")
        logger.info(f"  精度: {precision}")
        logger.info(f"  输出: {output_dir}")
        logger.info(f"  多GPU: {use_multi_gpu}")
        if use_multi_gpu and gpu_ids:
            logger.info(f"  GPU IDs: {gpu_ids}")

    def load_model(self):
        """加载MoE模型（支持单GPU和多GPU）"""
        logger.info(f"正在加载模型: {self.model_path}")

        if self.use_multi_gpu:
            # 多GPU模式：使用device_map='auto'
            logger.info("使用多GPU模式加载...")

            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer

                # 准备max_memory配置
                if self.gpu_ids:
                    max_memory = {i: "90GiB" for i in self.gpu_ids}
                    logger.info(f"为GPU {self.gpu_ids} 设置max_memory=90GiB")
                else:
                    # 使用所有可用GPU
                    num_gpus = torch.cuda.device_count()
                    max_memory = {i: "90GiB" for i in range(num_gpus)}
                    logger.info(f"为所有{num_gpus}个GPU设置max_memory=90GiB")

                # 使用device_map='auto'加载
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    device_map='auto',
                    torch_dtype=torch.float16,
                    trust_remote_code=True,
                    max_memory=max_memory
                )

                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_path,
                    trust_remote_code=True
                )

                # 显示设备分配
                if hasattr(self.model, 'hf_device_map'):
                    logger.info("模型设备分配:")
                    device_counts = {}
                    for name, dev in self.model.hf_device_map.items():
                        device_counts[dev] = device_counts.get(dev, 0) + 1

                    for dev, count in sorted(device_counts.items()):
                        logger.info(f"  {dev}: {count} 个模块")

                logger.info("✓ 模型已加载到多GPU")

                # 创建一个简单的model_loader包装
                class SimpleModelLoader:
                    def __init__(self, model, tokenizer):
                        self.model = model
                        self.tokenizer = tokenizer

                    def get_num_moe_layers(self):
                        # 简单估计
                        return sum(1 for n, m in model.named_modules() if 'moe' in n.lower())

                self.model_loader = SimpleModelLoader(
                    self.model, self.tokenizer)

            except Exception as e:
                logger.error(f"多GPU加载失败: {e}")
                raise
        else:
            # 单GPU模式：使用原有方法
            logger.info("使用单GPU模式加载...")

            self.model_loader = load_moe_model(
                model_name=self.model_path,
                device=self.device,
                torch_dtype=torch.float16
            )

            self.model = self.model_loader.model
            self.tokenizer = self.model_loader.tokenizer

            # 获取MoE信息
            num_layers = self.model_loader.get_num_moe_layers()
            logger.info(f"✓ 模型加载完成: {num_layers} MoE 层")

            # 打印内存使用
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated() / 1024**3
                logger.info(f"  GPU 内存使用: {memory_allocated:.2f} GB")

    def generate_ebss_calibration_data(
        self,
        seed_texts: List[str],
        ebss_config: Optional[EBSSConfig] = None
    ) -> List[str]:
        """
        步骤1: 使用EBSS生成专家均衡的校准数据

        Args:
            seed_texts: 种子文本
            ebss_config: EBSS配置

        Returns:
            生成的校准样本列表
        """
        logger.info("=" * 60)
        logger.info("步骤 1/4: EBSS 生成专家均衡校准数据")
        logger.info("=" * 60)

        if ebss_config is None:
            ebss_config = EBSSConfig(
                beam_width=4,
                tau=1.2,
                max_tokens=512,
                num_samples=len(seed_texts)
            )

        logger.info(f"  Beam Width: {ebss_config.beam_width}")
        logger.info(f"  Temperature (τ): {ebss_config.tau}")
        logger.info(f"  Max Tokens: {ebss_config.max_tokens}")
        logger.info(f"  样本数: {len(seed_texts)}")

        # 创建EBSS采样器
        self.ebss_sampler = EBSSSampler(
            model=self.model,
            tokenizer=self.tokenizer,
            config=ebss_config,
            device=self.device
        )

        # 生成样本
        logger.info("正在生成EBSS样本...")
        start_time = time.time()

        ebss_samples = []
        for idx, seed in enumerate(tqdm(seed_texts, desc="EBSS生成")):
            try:
                samples = self.ebss_sampler.generate([seed])
                ebss_samples.extend(samples)
            except Exception as e:
                logger.warning(f"EBSS生成失败 (样本 {idx}): {e}, 使用原始种子")
                ebss_samples.append(seed)

        elapsed = time.time() - start_time
        logger.info(f"✓ EBSS生成完成: {len(ebss_samples)} 样本, 耗时 {elapsed:.2f}秒")

        # 保存EBSS样本
        ebss_file = self.output_dir / "ebss_calibration_samples.txt"
        with open(ebss_file, 'w', encoding='utf-8') as f:
            for idx, sample in enumerate(ebss_samples):
                f.write(f"=== 样本 {idx+1} ===\n")
                f.write(sample)
                f.write("\n\n")

        logger.info(f"  已保存到: {ebss_file}")

        # 统计专家激活情况
        if self.ebss_sampler.expert_counts:
            logger.info("  专家激活统计:")
            sorted_experts = sorted(self.ebss_sampler.expert_counts.items())
            for expert_id, count in sorted_experts[:10]:  # 显示前10个
                logger.info(f"    专家 {expert_id}: {count} 次")
            if len(sorted_experts) > 10:
                logger.info(f"    ... 以及其他 {len(sorted_experts)-10} 个专家")

        return ebss_samples

    def collect_calibration_data(
        self,
        calibration_texts: List[str],
        batch_size: int = 1
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        步骤2: 收集激活和affinity数据

        Args:
            calibration_texts: 校准文本
            batch_size: 批次大小

        Returns:
            校准数据字典: {layer_name: {"inputs": tensor, "affinities": tensor}}
        """
        logger.info("=" * 60)
        logger.info("步骤 2/4: 收集激活和Affinity数据")
        logger.info("=" * 60)

        from ..runners.collect_calib import CalibrationCollector

        calib_collector = CalibrationCollector(
            model_loader=self.model_loader,
            num_samples=len(calibration_texts),
            max_seq_len=512
        )

        logger.info(f"正在处理 {len(calibration_texts)} 个校准样本...")
        calib_data = calib_collector.collect_from_dataset(
            calibration_texts,
            batch_size=batch_size
        )

        # 保存原始校准数据
        calib_file = self.output_dir / "calibration_data.pkl"
        calib_data.save(str(calib_file))
        logger.info(f"✓ 校准数据已保存: {calib_file}")

        # 转换为量化器需要的格式
        processed_data = self._process_calibration_data(calib_data)

        logger.info(f"✓ 收集完成: {len(processed_data)} 个层")

        return processed_data

    def _process_calibration_data(
        self,
        calib_data
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """处理校准数据为量化器所需格式"""
        processed = {}

        # 处理专家层数据
        for (layer_idx, expert_id) in calib_data.expert_activations:
            layer_name = f"layer_{layer_idx}.expert_{expert_id}"

            activations = calib_data.expert_activations[(layer_idx, expert_id)]
            affinities = calib_data.expert_affinities.get(
                (layer_idx, expert_id))

            if affinities is not None:
                processed[layer_name] = {
                    "inputs": activations,
                    "affinities": affinities
                }

        logger.info(f"  处理了 {len(processed)} 个专家层")

        return processed

    def quantize_model_with_agq(
        self,
        calibration_data: Dict[str, Dict[str, torch.Tensor]]
    ) -> Dict[str, any]:
        """
        步骤3: 使用AGQ量化模型

        Args:
            calibration_data: 校准数据

        Returns:
            量化统计信息
        """
        logger.info("=" * 60)
        logger.info("步骤 3/4: AGQ 量化专家层")
        logger.info("=" * 60)

        # 创建MoEQuant量化器
        self.quantizer = create_moequant_quantizer(
            precision=self.precision,
            ebss_beam_width=4,
            ebss_tau=1.2,
            ebss_num_samples=512,
            agq_use_error_compensation=True
        )

        logger.info(f"  量化精度: {self.precision.upper()}")
        logger.info(f"  权重位宽: {self.quantizer.config.weight_bits}")
        logger.info(f"  激活位宽: {self.quantizer.config.activation_bits}")
        logger.info(f"  分组大小: {self.quantizer.config.agq_group_size}")

        # 收集所有专家层
        expert_layers = {}
        for name, module in self.model.named_modules():
            if "expert" in name.lower() and isinstance(module, torch.nn.Linear):
                expert_layers[name] = module

        logger.info(f"找到 {len(expert_layers)} 个专家层")

        # 量化专家层
        logger.info("正在量化专家层...")
        quantized_results = {}

        for layer_name in tqdm(expert_layers.keys(), desc="量化进度"):
            if layer_name not in calibration_data:
                logger.warning(f"  跳过 {layer_name} (无校准数据)")
                continue

            layer = expert_layers[layer_name]
            calib = calibration_data[layer_name]

            try:
                W_quant, scales, stats = self.quantizer.quantize_expert_layer_with_agq(
                    layer=layer,
                    inputs=calib["inputs"],
                    affinities=calib["affinities"]
                )

                # 应用量化权重
                layer.weight.data = W_quant.to(
                    layer.weight.device, dtype=layer.weight.dtype)

                quantized_results[layer_name] = stats

            except Exception as e:
                logger.error(f"  量化 {layer_name} 失败: {e}")
                quantized_results[layer_name] = {"error": str(e)}

        # 保存量化统计
        stats_file = self.output_dir / "quantization_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(quantized_results, f, indent=2)

        logger.info(f"✓ 量化完成: {len(quantized_results)} 层")
        logger.info(f"  统计信息: {stats_file}")

        # 打印关键统计
        avg_mse = sum(s.get("weight_mse", 0) for s in quantized_results.values(
        ) if isinstance(s, dict) and "weight_mse" in s) / max(len(quantized_results), 1)
        avg_weighted_mse = sum(s.get("weighted_output_mse", 0) for s in quantized_results.values(
        ) if isinstance(s, dict) and "weighted_output_mse" in s) / max(len(quantized_results), 1)

        logger.info(f"  平均权重MSE: {avg_mse:.6f}")
        logger.info(f"  平均加权输出MSE: {avg_weighted_mse:.6f}")

        return quantized_results

    def save_quantized_model(self, quantization_stats: Dict):
        """
        步骤4: 保存量化模型 (SafeTensors格式)

        Args:
            quantization_stats: 量化统计信息
        """
        logger.info("=" * 60)
        logger.info("步骤 4/4: 保存量化模型")
        logger.info("=" * 60)

        # 准备量化配置
        quant_config = {
            "quantization_method": "MoEQuant",
            "precision": self.precision,
            "weight_bits": self.quantizer.config.weight_bits,
            "activation_bits": self.quantizer.config.activation_bits,
            "group_size": self.quantizer.config.agq_group_size,
            "ebss_beam_width": self.quantizer.config.ebss_beam_width,
            "ebss_tau": self.quantizer.config.ebss_tau,
            "agq_error_compensation": self.quantizer.config.agq_use_error_compensation,
        }

        # 保存模型 (safetensors格式)
        logger.info("正在保存模型 (SafeTensors格式)...")
        save_quantized_model_safetensors(
            model=self.model,
            output_dir=str(self.output_dir),
            source_model_dir=self.model_path,
            quantization_config=quant_config,
            max_shard_size="5GB"
        )

        logger.info(f"✓ 模型已保存到: {self.output_dir}")

        # 生成结果摘要
        summary = {
            "model_path": str(self.model_path),
            "output_dir": str(self.output_dir),
            "precision": self.precision,
            "quantization_config": quant_config,
            "quantization_stats_summary": {
                "total_layers": len(quantization_stats),
                "successful_layers": sum(1 for s in quantization_stats.values() if isinstance(s, dict) and "weight_mse" in s),
                "failed_layers": sum(1 for s in quantization_stats.values() if isinstance(s, dict) and "error" in s),
            }
        }

        summary_file = self.output_dir / "quantization_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"  摘要文件: {summary_file}")

    def run_full_pipeline(
        self,
        seed_texts: List[str],
        ebss_config: Optional[EBSSConfig] = None,
        calibration_batch_size: int = 1
    ) -> Dict:
        """
        运行完整的MoEQuant PTQ流程

        Args:
            seed_texts: 种子文本列表
            ebss_config: EBSS配置
            calibration_batch_size: 校准批次大小

        Returns:
            完整结果字典
        """
        logger.info("=" * 60)
        logger.info("MoEQuant PTQ 完整流程开始")
        logger.info("=" * 60)

        start_time = time.time()

        try:
            # 步骤1: 加载模型
            self.load_model()

            # 步骤2: EBSS生成校准数据
            ebss_samples = self.generate_ebss_calibration_data(
                seed_texts, ebss_config
            )

            # 步骤3: 收集校准数据
            calibration_data = self.collect_calibration_data(
                ebss_samples, calibration_batch_size
            )

            # 步骤4: AGQ量化
            quantization_stats = self.quantize_model_with_agq(calibration_data)

            # 步骤5: 保存模型
            self.save_quantized_model(quantization_stats)

            elapsed = time.time() - start_time
            logger.info("=" * 60)
            logger.info(f"✓ MoEQuant PTQ 完成! 总耗时: {elapsed/60:.2f} 分钟")
            logger.info("=" * 60)

            results = {
                "success": True,
                "output_dir": str(self.output_dir),
                "elapsed_time": elapsed,
                "quantization_stats": quantization_stats
            }

            return results

        except Exception as e:
            logger.error(f"❌ PTQ 流程失败: {e}")
            import traceback
            traceback.print_exc()

            return {
                "success": False,
                "error": str(e)
            }


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(
        description="MoEQuant PTQ - 完全遵循MoEQuant论文的量化流程"
    )

    # 模型参数
    parser.add_argument("--model", type=str, required=True,
                        help="模型路径")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="输出目录")

    # 量化精度
    parser.add_argument("--precision", type=str, choices=["w8a8", "w4a4", "w2a2"],
                        default="w4a4", help="量化精度")

    # 多GPU参数
    parser.add_argument("--multi-gpu", action="store_true",
                        help="使用多GPU模式")
    parser.add_argument("--gpu-ids", type=str, default=None,
                        help="逗号分隔的GPU IDs，如: 1,2,3,4")

    # EBSS参数
    parser.add_argument("--ebss-beam-width", type=int, default=4,
                        help="EBSS beam search 宽度")
    parser.add_argument("--ebss-tau", type=float, default=1.2,
                        help="EBSS 温度参数")
    parser.add_argument("--ebss-max-tokens", type=int, default=512,
                        help="EBSS 最大生成token数")

    # 校准参数
    parser.add_argument("--calib-size", type=int, default=512,
                        help="校准样本数量")
    parser.add_argument("--seed-text", type=str, default=None,
                        help="种子文本文件路径")
    parser.add_argument("--calibration-batch-size", type=int, default=1,
                        help="校准批次大小")

    # AGQ参数
    parser.add_argument("--agq-group-size", type=int, default=None,
                        help="AGQ 分组大小 (None则自动选择)")
    parser.add_argument("--no-agq-error-compensation", action="store_true",
                        help="禁用AGQ误差补偿")

    args = parser.parse_args()

    # 准备种子文本
    if args.seed_text and Path(args.seed_text).exists():
        with open(args.seed_text, 'r', encoding='utf-8') as f:
            seed_texts = [line.strip() for line in f if line.strip()]
        logger.info(f"从文件加载了 {len(seed_texts)} 个种子文本")
    else:
        # 默认种子文本
        seed_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Artificial intelligence is transforming the world.",
            "Machine learning models require large amounts of data.",
            "Natural language processing enables computers to understand human language.",
            "Deep learning has revolutionized computer vision and speech recognition.",
        ]
        logger.info(f"使用默认种子文本 ({len(seed_texts)} 个)")

    # 扩展到目标数量
    if len(seed_texts) < args.calib_size:
        seed_texts = (seed_texts * (args.calib_size //
                      len(seed_texts) + 1))[:args.calib_size]

    # 创建EBSS配置
    ebss_config = EBSSConfig(
        beam_width=args.ebss_beam_width,
        tau=args.ebss_tau,
        max_tokens=args.ebss_max_tokens,
        num_samples=len(seed_texts)
    )

    # 解析GPU IDs
    gpu_ids = None
    if args.gpu_ids:
        gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(',')]
        logger.info(f"指定GPU: {gpu_ids}")

    # 创建运行器
    runner = MoEQuantPTQRunner(
        model_path=args.model,
        precision=args.precision,
        output_dir=args.output_dir,
        use_multi_gpu=args.multi_gpu,
        gpu_ids=gpu_ids
    )

    # 运行完整流程
    results = runner.run_full_pipeline(
        seed_texts=seed_texts,
        ebss_config=ebss_config,
        calibration_batch_size=args.calibration_batch_size
    )

    if results["success"]:
        logger.info("\n" + "=" * 60)
        logger.info("🎉 量化成功完成!")
        logger.info("=" * 60)
        logger.info(f"输出目录: {results['output_dir']}")
        logger.info(f"总耗时: {results['elapsed_time']/60:.2f} 分钟")
        logger.info("\n下一步:")
        logger.info("1. 测试量化模型:")
        logger.info(
            f"   python -m transformers.models.auto --model {args.output_dir}")
        logger.info("2. 评估精度:")
        logger.info(f"   python -m moe_quant.eval --model {args.output_dir}")
    else:
        logger.error("\n❌ 量化失败!")
        logger.error(f"错误: {results.get('error', 'Unknown error')}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
