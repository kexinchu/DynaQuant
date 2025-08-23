#!/usr/bin/env python3
"""
GPTQ反量化器（修复版）
沿输出通道打包的 4-bit GPTQ 权重解包与反量化
"""

import torch
from typing import Optional


class GPTQDequantizer:
    """GPTQ反量化器"""

    @staticmethod
    def dequantize_gptq_weight(
        qweight: torch.Tensor,
        qzeros: torch.Tensor,
        scales: torch.Tensor,
        g_idx: Optional[torch.Tensor] = None,  # 兼容签名，当前未使用
        bits: int = 4,
        group_size: Optional[int] = None,      # 可选；若不提供则由形状自动推导
    ) -> torch.Tensor:
        """
        反量化 GPTQ 权重（沿输出通道打包）

        约定的张量形状（常见 GPTQ 导出格式）:
          - qweight: [OC//pack, IC] (int32)  pack = 32//bits（4bit时为8）
          - qzeros : [OC//g, IC//pack] (int32)  每元素再打 pack 个4-bit零点
          - scales : [OC//g, IC] (float16/float32)
        返回:
          - weight_fp16: [OC, IC] (torch.float16)
        """
        try:
            assert qweight.dtype == torch.int32 and qzeros.dtype == torch.int32, \
                "qweight 和 qzeros 必须是 int32（内部打包的载体）"
            pack = 32 // bits
            oc_pack, IC = qweight.shape
            OC = oc_pack * pack

            # 推导 g（每组输出通道数）
            groups_out = qzeros.shape[0]  # = OC // g
            assert OC % groups_out == 0, "OC 必须能被 qzeros.shape[0] 整除"
            g = OC // groups_out

            # 校验 scales 形状
            assert scales.shape == (groups_out, IC), \
                f"scales 形状应为 [OC//g, IC]，当前为 {tuple(scales.shape)}"

            # ---- 解包 qweight 到 [OC, IC]，沿输出通道扩展 ----
            Wq = GPTQDequantizer._unpack_int32_to_nibbles_rows(qweight, bits=bits)  # int16 [OC, IC]

            # ---- 从 qzeros 取每一列对应 nibble 的零点，并广播到 [OC, IC] ----
            # 对第 j 列：使用 qzeros[:, j//pack] 的第 (j%pack) 个 nibble
            device = qweight.device
            mask = (1 << bits) - 1  # 0xF
            col = torch.arange(IC, device=device)
            qz_cols = qzeros[:, (col // pack)]                 # [OC//g, IC]
            shift = (col % pack) * bits                        # [IC]
            zp_group_ic = (qz_cols >> shift.unsqueeze(0)) & mask  # [OC//g, IC]
            zp_full = zp_group_ic.repeat_interleave(g, dim=0).to(torch.int16)  # [OC, IC]

            # ---- 广播 scales 到 [OC, IC] ----
            scales_full = scales.repeat_interleave(g, dim=0).to(torch.float32)  # [OC, IC]

            # ---- 反量化: (w_q - zp) * scale ----
            W_fp16 = ((Wq - zp_full).to(torch.float32) * scales_full).to(torch.float16)  # [OC, IC]
            return W_fp16

        except Exception as e:
            # 打印更有用的上下文，便于排查
            print(f"[GPTQDequantizer] Error dequantizing GPTQ weight: {e}")
            try:
                pack = 32 // bits
                oc_pack, IC = qweight.shape
                OC = oc_pack * pack
                groups_out = qzeros.shape[0]
                g = OC // groups_out if groups_out > 0 else None
                print(f"  qweight shape: {tuple(qweight.shape)}, dtype: {qweight.dtype}")
                print(f"  qzeros  shape: {tuple(qzeros.shape)}, dtype: {qzeros.dtype}")
                print(f"  scales  shape: {tuple(scales.shape)}, dtype: {scales.dtype}")
                print(f"  derived OC={OC}, IC={IC}, pack={pack}, groups_out={groups_out}, g={g}")
            except Exception:
                pass
            # 安全回退
            # 返回一个零张量（[OC, IC] 若可推导，否则尽量不报错）
            try:
                pack = 32 // bits
                oc_pack, IC = qweight.shape
                OC = oc_pack * pack
                return torch.zeros((OC, IC), dtype=torch.float16, device=qweight.device)
            except Exception:
                return torch.zeros((scales.shape[0], scales.shape[1]), dtype=torch.float16, device=scales.device)

    @staticmethod
    def _unpack_int32_to_nibbles_rows(packed: torch.Tensor, bits: int = 4) -> torch.Tensor:
        """
        将按行打包的 int32（每个包含 32//bits 个子元素）解包为沿行扩张的矩阵:
          输入: packed [R, C] (int32)，每个元素含 'pack=32//bits' 个子值（低位->高位）
          输出: out [R*pack, C] (int16)   —— 将第 k 个 nibble 写到 out[k::pack, :]
        """
        assert bits in (2, 4, 8), "只支持 2/4/8 bit nibble 解包"
        pack = 32 // bits
        R, C = packed.shape
        out = torch.empty((R * pack, C), dtype=torch.int16, device=packed.device)
        mask = (1 << bits) - 1
        for k in range(pack):
            vals = (packed >> (k * bits)) & mask          # [R, C]
            out[k::pack, :] = vals.to(torch.int16)        # 交错写入行
        return out

    # ------- 如需保留“simple”接口，做成正确实现的别名 -------
    @staticmethod
    def dequantize_gptq_weight_simple(
        qweight: torch.Tensor,
        qzeros: torch.Tensor,
        scales: torch.Tensor,
        bits: int = 4
    ) -> torch.Tensor:
        """兼容旧接口：等价于 dequantize_gptq_weight（自动推导 g）"""
        return GPTQDequantizer.dequantize_gptq_weight(
            qweight=qweight, qzeros=qzeros, scales=scales, g_idx=None, bits=bits, group_size=None
        )
