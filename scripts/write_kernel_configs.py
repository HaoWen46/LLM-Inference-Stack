#!/usr/bin/env python3
"""
Generate Triton kernel config JSON files for Qwen3-30B-A3B-FP8 on RTX 4090.

This script writes heuristic-based configs (no GPU required) derived from
the H100/H20 patterns shipped with vLLM, adapted for Ada Lovelace (sm_89).
The files suppress vLLM's "sub-optimal" warnings on startup and provide
per-batch-size BLOCK_SIZE_M selection instead of a single fixed fallback.

For production-quality configs, run tune_kernels.py afterwards.

Usage:
    uv run python scripts/write_kernel_configs.py
"""
import json
from pathlib import Path

import vllm
from vllm.platforms import current_platform

VLLM_ROOT      = Path(vllm.__file__).parent
FP8_CFG_DIR    = VLLM_ROOT / "model_executor/layers/quantization/utils/configs"
MOE_CFG_DIR    = VLLM_ROOT / "model_executor/layers/fused_moe/configs"
DEVICE_NAME    = current_platform.get_device_name().replace(" ", "_")

BATCH_SIZES    = [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 256, 512, 1024, 1536, 2048, 3072, 4096]

# ── Per-M heuristic ───────────────────────────────────────────────────────────
# Derived from H100/H20 shipped configs:
#   small M  → small BLOCK_SIZE_M (reduces register pressure)
#   large M  → large BLOCK_SIZE_M (hides memory latency via more work per CTA)
# RTX 4090 (Ada Lovelace / sm_89) has 128 FP8 CUDA cores per SM; num_stages=3
# avoids occupancy issues at BLOCK_SIZE_N=128 vs H100's num_stages=4/5.

def fp8_gemm_config(M: int) -> dict:
    """FP8 block GEMM config for _w8a8_block_fp8_matmul.
    BLOCK_SIZE_N/K must be multiples of block_shape=128."""
    if M <= 4:
        bm, bn, gm, nw, ns = 16, 128, 1, 4, 3
    elif M <= 16:
        bm, bn, gm, nw, ns = 16, 128, 8, 4, 3
    elif M <= 48:
        bm, bn, gm, nw, ns = 32, 128, 8, 4, 3
    elif M <= 128:
        bm, bn, gm, nw, ns = 64, 128, 8, 4, 3
    elif M <= 512:
        bm, bn, gm, nw, ns = 64, 128, 16, 4, 3
    else:
        bm, bn, gm, nw, ns = 64, 128, 16, 8, 3
    return {
        "BLOCK_SIZE_M": bm,
        "BLOCK_SIZE_N": bn,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": gm,
        "num_warps": nw,
        "num_stages": ns,
    }


def moe_config(M: int) -> dict:
    """FP8 MoE fused_moe_kernel config.
    BLOCK_SIZE_N must be a multiple of block_shape[0]=128."""
    if M <= 4:
        bm, bn, gm, nw, ns = 16, 128, 1, 4, 3
    elif M <= 16:
        bm, bn, gm, nw, ns = 16, 128, 16, 4, 3
    elif M <= 64:
        bm, bn, gm, nw, ns = 16, 128, 32, 4, 3
    elif M <= 256:
        bm, bn, gm, nw, ns = 32, 128, 16, 4, 3
    elif M <= 1024:
        bm, bn, gm, nw, ns = 64, 128, 16, 4, 3
    else:
        bm, bn, gm, nw, ns = 64, 128, 64, 4, 3
    return {
        "BLOCK_SIZE_M": bm,
        "BLOCK_SIZE_N": bn,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": gm,
        "num_warps": nw,
        "num_stages": ns,
    }


def build_configs(fn) -> dict:
    return {str(M): fn(M) for M in BATCH_SIZES}


def write(path: Path, name: str, configs: dict):
    out = path / name
    with open(out, "w") as f:
        json.dump(configs, f, indent=4)
    print(f"  written: {out.name}")


if __name__ == "__main__":
    print(f"[config] Device: {DEVICE_NAME}")

    # FP8 GEMM shapes needed by Qwen3-30B-A3B-FP8 on this GPU
    print("\nFP8 block GEMM configs:")
    for N, K in [(2048, 2048), (2560, 2048)]:
        fname = f"N={N},K={K},device_name={DEVICE_NAME},dtype=fp8_w8a8,block_shape=[128,128].json"
        write(FP8_CFG_DIR, fname, build_configs(fp8_gemm_config))

    # MoE shapes for Qwen3-30B-A3B (E=128 experts, N_intermediate=384)
    print("\nFP8 MoE configs:")
    fname = f"E=128,N=384,device_name={DEVICE_NAME},dtype=fp8_w8a8,block_shape=[128,128].json"
    write(MOE_CFG_DIR, fname, build_configs(moe_config))

    print("\n[config] Done. Restart vLLM to pick up the configs.")
    print("         For tuned (benchmarked) configs, run: scripts/tune_kernels.py")
