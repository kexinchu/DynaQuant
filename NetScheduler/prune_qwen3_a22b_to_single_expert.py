#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, os, re, shutil
from pathlib import Path
from typing import Dict, List, Tuple
import torch
from safetensors.torch import load_file as st_load_file, save_file as st_save_file

# ======== Qwen3 典型命名约定（可按需微调）========
PAT_EXPERT = re.compile(
    r"^model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<eid>\d+)\.(?P<w>w1|w2|w3)\.weight$"
)
PAT_GATE_W = re.compile(r"^model\.layers\.(?P<layer>\d+)\.mlp\.gate\.weight$")
PAT_GATE_B = re.compile(r"^model\.layers\.(?P<layer>\d+)\.mlp\.gate\.bias$")

# 其它保留：attention、ln、embedding 等一概原样通过

def load_json(p: Path):
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def dump_json(obj, p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def find_weight_shards(src: Path) -> Tuple[str, List[Path], Path]:
    st_index = src / "model.safetensors.index.json"
    if st_index.exists():
        idx = load_json(st_index)
        files = sorted(set(idx.get("weight_map", {}).values()))
        return "safetensors", [src / f for f in files], st_index
    single_st = sorted(src.glob("*.safetensors"))
    if single_st:
        return "safetensors", single_st, None

    pt_index = src / "pytorch_model.bin.index.json"
    if pt_index.exists():
        idx = load_json(pt_index)
        files = sorted(set(idx.get("weight_map", {}).values()))
        return "bin", [src / f for f in files], pt_index
    single_bin = sorted(src.glob("pytorch_model*.bin"))
    if single_bin:
        return "bin", single_bin, None

    raise FileNotFoundError("No weight shards found.")

def rewrite_qwen_config(cfg: Dict, verbose=True):
    # 常见键名：n_experts=128，n_routed_experts=8（或 moe_top_k），将它们置 1
    candidate_1 = [
        "n_experts", "num_experts", "moe_num_experts",
        "num_local_experts", "expert_num", "experts"
    ]
    candidate_topk = [
        "n_routed_experts", "top_k", "moe_top_k",
        "router_top_k", "num_experts_per_token", "num_experts_per_tok"
    ]
    changed = False

    def set_if_present(k, v=1):
        nonlocal changed
        if k in cfg and cfg[k] != v:
            cfg[k] = v
            changed = True
            if verbose: print(f"[config] Set {k}={v}")

    for k in candidate_1:
        set_if_present(k, 1)
    for k in candidate_topk:
        set_if_present(k, 1)

    # 有些实现把 MoE 配置放子结构里，递归兜底
    for k, v in list(cfg.items()):
        if isinstance(v, dict):
            if rewrite_qwen_config(v, verbose):
                changed = True
    return changed

def estimate_bytes(t: torch.Tensor) -> int:
    try:
        return t.numel() * t.element_size()
    except Exception:
        return 0

def process_state_dict(sd: Dict[str, torch.Tensor], keep_eid: int) -> Dict[str, torch.Tensor]:
    """Qwen 专用裁剪：
       - 删除除 keep_eid 外的所有 experts.{eid}.(w1|w2|w3).weight
       - 将保留 expert 的路径 eid 归零：experts.0.*
       - 裁剪 gate.{weight,bias} 到单一 expert 输出
    """
    out: Dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        m_e = PAT_EXPERT.match(k)
        if m_e:
            eid = int(m_e.group("eid"))
            if eid != keep_eid:
                continue
            # 归一化路径 eid -> 0
            k_new = PAT_EXPERT.sub(
                f"model.layers.{m_e.group('layer')}.mlp.experts.0.{m_e.group('w')}.weight",
                k
            )
            out[k_new] = v
            continue

        if PAT_GATE_W.match(k):
            # 期望形状 [E, H] 或 [H, E]，两种都支持：
            if v.dim() == 2:
                E, H = v.size(0), v.size(1)
                if E >= 2:
                    idx = min(keep_eid, E - 1)
                    v_new = v.narrow(0, idx, 1).clone()  # [1, H]
                    out[k] = v_new
                    continue
                # 若不是 [E, H]，尝试 [H, E]
                H2, E2 = v.size(0), v.size(1)
                if E2 >= 2:
                    idx = min(keep_eid, E2 - 1)
                    v_new = v.narrow(1, idx, 1).clone()  # [H,1]
                    out[k] = v_new
                    continue
            # 兜底：不识别则原样保留
            out[k] = v
            continue

        if PAT_GATE_B.match(k):
            if v.dim() == 1 and v.size(0) >= 2:
                idx = min(keep_eid, v.size(0) - 1)
                out[k] = v.narrow(0, idx, 1).clone()  # [1]
            else:
                out[k] = v
            continue

        # 其它权重原样保留
        out[k] = v
    return out

def main():
    ap = argparse.ArgumentParser("Prune Qwen3 MoE multi-expert to single-expert (per layer).")
    ap.add_argument("--src", required=True, type=str)
    ap.add_argument("--dst", required=True, type=str)
    ap.add_argument("--expert-id", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    src = Path(args.src).resolve()
    dst = Path(args.dst).resolve()
    dst.mkdir(parents=True, exist_ok=True)

    # 复制非权重文件（除 config.json）
    for p in src.iterdir():
        if p.is_dir(): 
            continue
        if p.name.endswith((".bin", ".safetensors", ".index.json")):
            continue
        if p.name == "config.json":
            continue
        shutil.copy2(p, dst / p.name)

    # 改 config.json
    cfg_path = src / "config.json"
    if cfg_path.exists():
        cfg = load_json(cfg_path)
        rewrite_qwen_config(cfg, verbose=not args.dry_run)
        if not args.dry_run:
            dump_json(cfg, dst / "config.json")
    else:
        print("[warn] config.json not found; skip config rewrite.")

    fmt, shards, index_json = find_weight_shards(src)
    print(f"[info] format={fmt}, shards={len(shards)}, has_index={bool(index_json)}")

    new_weight_map: Dict[str, str] = {}
    total_bytes = 0

    def write_shard(rel: Path, new_sd: Dict[str, torch.Tensor]):
        nonlocal total_bytes
        if not args.dry_run:
            (dst / rel).parent.mkdir(parents=True, exist_ok=True)
        if fmt == "safetensors":
            if not args.dry_run:
                st_save_file(new_sd, str(dst / rel))
        else:
            if not args.dry_run:
                torch.save(new_sd, dst / rel)
        for k, t in new_sd.items():
            new_weight_map[k] = str(rel)
            total_bytes += estimate_bytes(t)

    # 逐分片处理
    for shard in shards:
        rel = shard.relative_to(src)
        if fmt == "safetensors":
            sd = st_load_file(str(shard))
        else:
            sd = torch.load(shard, map_location="cpu")
            if not isinstance(sd, dict):
                raise RuntimeError(f"{rel} is not a dict-like state_dict.")
        new_sd = process_state_dict(sd, args.expert_id)

        if args.dry_run:
            drop_cnt = len(sd) - len(new_sd)
            print(f"[dry-run] {rel}: keep={len(new_sd)} drop={drop_cnt}")
        else:
            write_shard(rel, new_sd)

        del sd, new_sd

    # 重建 index.json
    if not args.dry_run:
        idx_obj = {
            "metadata": {"total_size": total_bytes},
            "weight_map": new_weight_map,
        }
        if fmt == "safetensors":
            dump_json(idx_obj, dst / "model.safetensors.index.json")
        else:
            dump_json(idx_obj, dst / "pytorch_model.bin.index.json")

    # ======== 自检：确认没有 experts.>0，且 gate 输出维==1 ========
    print("[self-check] scanning pruned weights...")
    problems = 0
    for shard in (dst.glob("*.safetensors") if fmt=="safetensors" else dst.glob("pytorch_model*.bin")):
        sd = st_load_file(str(shard)) if shard.suffix == ".safetensors" else torch.load(shard, map_location="cpu")
        for k, v in sd.items():
            if PAT_EXPERT.match(k):
                # 剩下的都应是 experts.0.*
                if ".experts.0." not in k:
                    print(f"[ERR] found non-zero expert key after prune: {k}")
                    problems += 1
            if PAT_GATE_W.match(k):
                if v.dim()==2 and min(v.size(0), v.size(1)) != 1:
                    print(f"[ERR] gate.weight is not collapsed to 1 on at least one dim: {k} shape={tuple(v.shape)}")
                    problems += 1
            if PAT_GATE_B.match(k):
                if v.dim()!=1 or v.numel()!=1:
                    print(f"[ERR] gate.bias is not size 1: {k} shape={tuple(v.shape)}")
                    problems += 1
        del sd
    if problems == 0:
        print("[self-check] OK ✅")
    else:
        print(f"[self-check] Found {problems} problem(s). Please adjust patterns or report shapes.")

    print(("(dry-run) done." if args.dry_run else f"[done] written to: {dst}"))

if __name__ == "__main__":
    main()
