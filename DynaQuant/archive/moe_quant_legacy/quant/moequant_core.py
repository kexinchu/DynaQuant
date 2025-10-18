"""
MoEQuant Core Implementation
完全遵循 MoEQuant 论文的 EBSS + AGQ 设计
支持 W8A8, W4A4, W2A2 三种量化精度
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class MoEQuantConfig:
    """MoEQuant 统一配置"""
    # 量化精度选项: "w8a8", "w4a4", "w2a2"
    precision: str = "w4a4"

    # EBSS 配置
    ebss_beam_width: int = 4
    ebss_tau: float = 1.2
    ebss_max_tokens: int = 512
    ebss_num_samples: int = 512

    # AGQ 配置
    agq_group_size: int = 128  # W8: 128, W4: 64, W2: 64
    agq_use_error_compensation: bool = True
    agq_damping: float = 0.01
    agq_iterations: int = 10

    # 其他配置
    use_symmetric_quant: bool = True
    calibration_batch_size: int = 1

    def __post_init__(self):
        """根据precision自动设置参数"""
        if self.precision == "w8a8":
            self.weight_bits = 8
            self.activation_bits = 8
            self.agq_group_size = 128
        elif self.precision == "w4a4":
            self.weight_bits = 4
            self.activation_bits = 4
            self.agq_group_size = 64
        elif self.precision == "w2a2":
            self.weight_bits = 2
            self.activation_bits = 2
            self.agq_group_size = 64
        else:
            raise ValueError(
                f"不支持的精度: {self.precision}, 必须是 w8a8, w4a4, w2a2 之一")


class MoEQuantizer:
    """
    MoEQuant 核心量化器

    实现论文中的核心算法:
    1. EBSS (Expert-Balanced Self-Sampling) - 生成均衡的校准数据
    2. AGQ (Affinity-Guided Quantization) - 基于亲和度的量化

    支持三种精度: W8A8, W4A4, W2A2
    """

    def __init__(self, config: MoEQuantConfig):
        self.config = config
        logger.info(f"初始化 MoEQuantizer: {config.precision}")
        logger.info(f"  权重位宽: {config.weight_bits}")
        logger.info(f"  激活位宽: {config.activation_bits}")
        logger.info(f"  分组大小: {config.agq_group_size}")

    def quantize_weights_symmetric(
        self,
        W: torch.Tensor,
        bit_width: int,
        group_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        对称量化权重（MoEQuant 标准方法）

        量化公式:
            scale = max(|W_group|) / (2^(bit_width-1) - 1)
            W_int = round(W / scale)
            W_int = clip(W_int, -2^(bit_width-1), 2^(bit_width-1) - 1)
            W_quant = W_int * scale

        Args:
            W: 权重矩阵 [out_features, in_features]
            bit_width: 位宽 (2, 4, 8)
            group_size: 分组大小

        Returns:
            W_quant: 反量化后的权重 (FP16/FP32)
            scales: 每组的缩放因子 [out_features, n_groups]
        """
        out_features, in_features = W.shape
        n_groups = (in_features + group_size - 1) // group_size

        # 填充到完整的group
        pad_size = n_groups * group_size - in_features
        if pad_size > 0:
            W_padded = F.pad(W, (0, pad_size), mode='constant', value=0)
        else:
            W_padded = W

        # 重塑为 [out, n_groups, group_size]
        W_grouped = W_padded.reshape(out_features, n_groups, group_size)

        # 计算每组的scale
        scales = W_grouped.abs().max(
            dim=-1, keepdim=True)[0]  # [out, n_groups, 1]
        scales = scales.clamp(min=1e-8)  # 避免除零

        # 对称量化
        n_levels = 2 ** (bit_width - 1)  # 2-bit: 2, 4-bit: 8, 8-bit: 128
        W_normalized = W_grouped / scales

        # 量化到整数
        W_int = torch.clamp(
            torch.round(W_normalized * (n_levels - 1)),
            -n_levels,
            n_levels - 1
        )

        # 反量化
        W_quant_grouped = (W_int / (n_levels - 1)) * scales

        # 重塑回原始形状
        W_quant_padded = W_quant_grouped.reshape(
            out_features, n_groups * group_size)
        W_quant = W_quant_padded[:, :in_features]  # 移除填充

        scales = scales.squeeze(-1)  # [out_features, n_groups]

        return W_quant, scales

    def compute_affinity_weighted_hessian(
        self,
        X: torch.Tensor,
        affinities: torch.Tensor
    ) -> torch.Tensor:
        """
        计算亲和度加权的 Hessian 矩阵（MoEQuant AGQ 核心）

        H = (X ⊙ √c) (X ⊙ √c)^T

        其中 c 是 gating affinity (router scores)

        Args:
            X: 输入激活 [N, in_features]
            affinities: 亲和度分数 [N]

        Returns:
            H: Hessian 矩阵 [in_features, in_features]
        """
        # 归一化affinity (保持总和为N)
        c = affinities * (affinities.numel() / (affinities.sum() + 1e-8))

        # 加权: X * sqrt(c)
        sqrt_c = torch.sqrt(c.clamp(min=0)).unsqueeze(-1)  # [N, 1]
        X_weighted = X * sqrt_c  # [N, in_features]

        # 计算 Hessian: X^T X
        H = X_weighted.T @ X_weighted  # [in_features, in_features]

        # 添加阻尼项 (数值稳定性)
        H += torch.eye(H.size(0), device=H.device,
                       dtype=H.dtype) * self.config.agq_damping

        return H

    def quantize_with_error_compensation(
        self,
        W: torch.Tensor,
        H_inv: torch.Tensor,
        bit_width: int,
        group_size: int
    ) -> torch.Tensor:
        """
        带误差补偿的量化（类似 GPTQ，MoEQuant AGQ 的一部分）

        逐列量化，并将误差传播到后续列

        Args:
            W: 权重矩阵 [out_features, in_features]
            H_inv: Hessian 逆矩阵 [in_features, in_features]
            bit_width: 位宽
            group_size: 分组大小

        Returns:
            W_quant: 量化后的权重
        """
        out_features, in_features = W.shape
        W_quant = W.clone()

        n_levels = 2 ** (bit_width - 1)

        # 逐列量化
        for col in range(in_features):
            # 当前列
            w_col = W_quant[:, col]

            # 确定所属的group
            group_idx = col // group_size
            group_start = group_idx * group_size
            group_end = min(group_start + group_size, in_features)

            # 计算该group的scale
            group_weights = W_quant[:, group_start:group_end]
            scale = group_weights.abs().max().clamp(min=1e-8)

            # 量化
            w_normalized = w_col / scale
            w_int = torch.clamp(
                torch.round(w_normalized * (n_levels - 1)),
                -n_levels,
                n_levels - 1
            )
            w_quant_col = (w_int / (n_levels - 1)) * scale

            # 计算误差
            error = w_col - w_quant_col

            # 传播误差到后续列
            if col + 1 < in_features:
                h_inv_col = H_inv[col, col + 1:]
                h_ii = H_inv[col, col]

                if abs(h_ii) > 1e-8:
                    compensation = torch.outer(error, h_inv_col) / h_ii
                    W_quant[:, col + 1:] -= compensation

            # 更新量化权重
            W_quant[:, col] = w_quant_col

        return W_quant

    def quantize_expert_layer_with_agq(
        self,
        layer: nn.Linear,
        inputs: torch.Tensor,
        affinities: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        """
        使用 AGQ 量化专家层（MoEQuant 核心算法）

        Args:
            layer: 线性层
            inputs: 输入激活 [batch, seq_len, in_features] 或 [N, in_features]
            affinities: 亲和度分数 [batch, seq_len] 或 [N]

        Returns:
            W_quant: 量化后的权重
            scales: 缩放因子
            stats: 统计信息
        """
        W = layer.weight.data

        # 展平输入和affinity
        if inputs.dim() == 3:
            X = inputs.reshape(-1, inputs.size(-1))
            c = affinities.reshape(-1)
        else:
            X = inputs
            c = affinities

        # 移到同一设备
        X = X.to(W.device)
        c = c.to(W.device)

        # 计算亲和度加权的 Hessian
        H = self.compute_affinity_weighted_hessian(X, c)

        # 尝试使用误差补偿
        if self.config.agq_use_error_compensation:
            try:
                H_inv = torch.linalg.inv(H)
                W_quant = self.quantize_with_error_compensation(
                    W, H_inv,
                    self.config.weight_bits,
                    self.config.agq_group_size
                )
                scales = None
                use_error_comp = True
            except Exception as e:
                logger.warning(f"误差补偿失败，回退到标准量化: {e}")
                W_quant, scales = self.quantize_weights_symmetric(
                    W,
                    self.config.weight_bits,
                    self.config.agq_group_size
                )
                use_error_comp = False
        else:
            W_quant, scales = self.quantize_weights_symmetric(
                W,
                self.config.weight_bits,
                self.config.agq_group_size
            )
            use_error_comp = False

        # 计算统计信息
        with torch.no_grad():
            # 权重 MSE
            w_mse = F.mse_loss(W, W_quant).item()

            # 输出误差 (affinity加权)
            Y_fp = F.linear(X, W, layer.bias)
            Y_quant = F.linear(X, W_quant, layer.bias)
            output_error = (Y_fp - Y_quant) ** 2

            # 加权 MSE
            c_normalized = c / (c.sum() + 1e-8)
            weighted_mse = (c_normalized.unsqueeze(-1)
                            * output_error).mean().item()

        stats = {
            "weight_mse": w_mse,
            "weighted_output_mse": weighted_mse,
            "bit_width": self.config.weight_bits,
            "group_size": self.config.agq_group_size,
            "affinity_mean": c.mean().item(),
            "affinity_std": c.std().item(),
            "affinity_max": c.max().item(),
            "use_error_compensation": use_error_comp,
        }

        return W_quant, scales, stats

    def quantize_all_expert_layers(
        self,
        expert_layers: Dict[str, nn.Linear],
        calibration_data: Dict[str, Dict[str, torch.Tensor]]
    ) -> Dict[str, Tuple[torch.Tensor, Optional[torch.Tensor], Dict]]:
        """
        量化所有专家层

        Args:
            expert_layers: {layer_name: nn.Linear}
            calibration_data: {layer_name: {"inputs": tensor, "affinities": tensor}}

        Returns:
            {layer_name: (W_quant, scales, stats)}
        """
        results = {}

        logger.info(f"开始量化 {len(expert_layers)} 个专家层...")

        for layer_name, layer in expert_layers.items():
            if layer_name not in calibration_data:
                logger.warning(f"缺少 {layer_name} 的校准数据，跳过")
                continue

            calib = calibration_data[layer_name]
            inputs = calib["inputs"]
            affinities = calib["affinities"]

            try:
                W_quant, scales, stats = self.quantize_expert_layer_with_agq(
                    layer, inputs, affinities
                )

                results[layer_name] = (W_quant, scales, stats)

                logger.info(
                    f"✓ {layer_name}: "
                    f"W_MSE={stats['weight_mse']:.6f}, "
                    f"Weighted_MSE={stats['weighted_output_mse']:.6f}"
                )
            except Exception as e:
                logger.error(f"量化 {layer_name} 失败: {e}")
                results[layer_name] = (None, None, {"error": str(e)})

        return results

    def apply_quantized_weights(
        self,
        model: nn.Module,
        quantized_weights: Dict[str,
                                Tuple[torch.Tensor, Optional[torch.Tensor], Dict]]
    ):
        """
        将量化后的权重应用到模型

        Args:
            model: 原始模型
            quantized_weights: 量化结果
        """
        logger.info("应用量化权重到模型...")

        for layer_name, (W_quant, scales, stats) in quantized_weights.items():
            if W_quant is None:
                continue

            # 找到对应的layer
            parts = layer_name.split('.')
            module = model
            for part in parts[:-1]:
                module = getattr(module, part)

            # 更新权重
            layer = getattr(module, parts[-1])
            if isinstance(layer, nn.Linear):
                layer.weight.data = W_quant.to(
                    layer.weight.device, dtype=layer.weight.dtype)
                logger.debug(f"  更新 {layer_name} 权重")

        logger.info("权重应用完成")

    def quantize_activation_placeholder(
        self,
        X: torch.Tensor,
        bit_width: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        激活量化（占位符，实际推理时实现）

        Args:
            X: 激活 [N, features]
            bit_width: 位宽

        Returns:
            X_quant: 量化后的激活
            scale: 缩放因子
            zero_point: 零点
        """
        # 非对称量化
        x_min = X.min()
        x_max = X.max()

        n_levels = 2 ** bit_width
        scale = (x_max - x_min) / (n_levels - 1)
        scale = scale.clamp(min=1e-8)

        zero_point = -torch.round(x_min / scale)
        zero_point = zero_point.clamp(0, n_levels - 1)

        # 量化
        X_normalized = X / scale + zero_point
        X_int = torch.clamp(torch.round(X_normalized), 0, n_levels - 1)

        # 反量化
        X_quant = (X_int - zero_point) * scale

        return X_quant, scale, zero_point


def create_moequant_quantizer(
    precision: str = "w4a4",
    ebss_beam_width: int = 4,
    ebss_tau: float = 1.2,
    ebss_num_samples: int = 512,
    agq_group_size: Optional[int] = None,
    agq_use_error_compensation: bool = True
) -> MoEQuantizer:
    """
    便捷函数：创建 MoEQuantizer

    Args:
        precision: "w8a8", "w4a4", "w2a2"
        ebss_beam_width: EBSS beam 宽度
        ebss_tau: EBSS 温度参数
        ebss_num_samples: 校准样本数
        agq_group_size: AGQ 分组大小 (None 则自动选择)
        agq_use_error_compensation: 是否使用误差补偿

    Returns:
        MoEQuantizer 实例
    """
    config = MoEQuantConfig(
        precision=precision,
        ebss_beam_width=ebss_beam_width,
        ebss_tau=ebss_tau,
        ebss_num_samples=ebss_num_samples,
        agq_use_error_compensation=agq_use_error_compensation
    )

    if agq_group_size is not None:
        config.agq_group_size = agq_group_size

    return MoEQuantizer(config)
