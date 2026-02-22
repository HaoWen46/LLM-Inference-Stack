#!/usr/bin/env python3
"""
Benchmark-tune Triton kernel configs for FP8 block-GEMM and FP8 MoE on the
current GPU.  Overwrites the heuristic configs written by write_kernel_configs.py
with measured-optimal values.

Strategy
--------
- Benchmark 5 representative batch sizes: [1, 16, 64, 256, 2048]
- 24-config search space per shape (tight, covers key BLOCK_SIZE_M / GROUP choices)
- Extrapolate to all 18 batch sizes by nearest-M lookup
- Total kernel compilations: ~120 (≈5–10 min depending on Triton cache state)

Usage (runs safely on GPU 0 while vLLM is on GPUs 1/2):
    CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_ORDER=PCI_BUS_ID \
      uv run python scripts/tune_kernels.py
"""
import json
import sys
import time
from itertools import product
from pathlib import Path

import torch
import triton
import triton.language as tl

import vllm
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    _w8a8_block_fp8_matmul,
)
from vllm.model_executor.layers.fused_moe.fused_moe import invoke_fused_moe_kernel
from vllm.platforms import current_platform

# ── Paths & constants ─────────────────────────────────────────────────────────
VLLM_ROOT   = Path(vllm.__file__).parent
FP8_CFG_DIR = VLLM_ROOT / "model_executor/layers/quantization/utils/configs"
MOE_CFG_DIR = VLLM_ROOT / "model_executor/layers/fused_moe/configs"
DEVICE_NAME = current_platform.get_device_name().replace(" ", "_")
DEVICE      = "cuda"
FP8_DTYPE   = torch.float8_e4m3fn
OUT_DTYPE   = torch.bfloat16
BLOCK_SHAPE = [128, 128]

# Benchmark only these M values; interpolate the rest by nearest-M
PROBE_M     = [1, 16, 64, 256, 2048]
ALL_M       = [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 256, 512, 1024, 1536, 2048, 3072, 4096]
N_WARMUP    = 5
N_ITER      = 50

print(f"[tune] GPU: {torch.cuda.get_device_name(0)}")
print(f"[tune] device_name key: {DEVICE_NAME}")


# ── Timing helper ─────────────────────────────────────────────────────────────

def gpu_ms(fn, warmup=N_WARMUP, reps=N_ITER) -> float:
    """Time fn() on GPU, return mean ms. Returns inf on any exception."""
    try:
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(reps):
            fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / reps
    except Exception:
        return float("inf")


# ── Search space ──────────────────────────────────────────────────────────────
# BLOCK_SIZE_N/K must be multiples of block_shape (128) for FP8 block quant.
# Keep search space small: 24 configs.
FP8_GEMM_CONFIGS = [
    {"BLOCK_SIZE_M": bm, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128,
     "GROUP_SIZE_M": gm, "num_warps": nw, "num_stages": ns}
    for bm in [16, 32, 64, 128]
    for gm in [1, 8, 32]
    for nw in [4]
    for ns in [3]
]  # 4 * 3 * 1 * 1 = 12 configs

MOE_CONFIGS = [
    {"BLOCK_SIZE_M": bm, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128,
     "GROUP_SIZE_M": gm, "num_warps": nw, "num_stages": ns}
    for bm in [16, 32, 64]
    for gm in [1, 8, 32, 64]
    for nw in [4]
    for ns in [3]
]  # 3 * 4 * 1 * 1 = 12 configs


def nearest_config(probe_results: dict, M: int) -> dict:
    """Return the config for the nearest probed M value."""
    nearest = min(probe_results.keys(), key=lambda x: abs(x - M))
    return probe_results[nearest]


def expand_to_all_m(probe_results: dict) -> dict:
    """Extrapolate probe results to all standard batch sizes."""
    return {str(M): nearest_config(probe_results, M) for M in ALL_M}


# ── FP8 block GEMM ────────────────────────────────────────────────────────────

def bench_fp8_gemm(M: int, N: int, K: int, cfg: dict) -> float:
    bn, bk = BLOCK_SHAPE
    A  = torch.randn(M, K, device=DEVICE, dtype=torch.float16).to(FP8_DTYPE)
    B  = torch.randn(N, K, device=DEVICE, dtype=torch.float16).to(FP8_DTYPE)
    As = torch.ones(M, triton.cdiv(K, bk), device=DEVICE)
    Bs = torch.ones(triton.cdiv(N, bn), triton.cdiv(K, bk), device=DEVICE)
    C  = torch.empty(M, N, dtype=OUT_DTYPE, device=DEVICE)
    grid = (triton.cdiv(M, cfg["BLOCK_SIZE_M"]) * triton.cdiv(N, cfg["BLOCK_SIZE_N"]),)

    def run():
        _w8a8_block_fp8_matmul[grid](
            A, B, C, As, Bs, M, N, K, bn, bk,
            A.stride(0), A.stride(1),
            B.stride(1), B.stride(0),
            C.stride(0), C.stride(1),
            As.stride(0), As.stride(1),
            Bs.stride(1), Bs.stride(0),
            BLOCK_SIZE_M=cfg["BLOCK_SIZE_M"],
            BLOCK_SIZE_N=cfg["BLOCK_SIZE_N"],
            BLOCK_SIZE_K=cfg["BLOCK_SIZE_K"],
            GROUP_SIZE_M=cfg["GROUP_SIZE_M"],
            num_warps=cfg["num_warps"],
            num_stages=cfg["num_stages"],
        )

    return gpu_ms(run)


def tune_fp8_gemm(N: int, K: int) -> dict:
    probe = {}
    for M in PROBE_M:
        best_ms, best_cfg = float("inf"), None
        for cfg in FP8_GEMM_CONFIGS:
            ms = bench_fp8_gemm(M, N, K, cfg)
            if ms < best_ms:
                best_ms, best_cfg = ms, cfg
        print(f"  M={M:5d}: {best_ms:.3f}ms  {best_cfg}", flush=True)
        if best_cfg:
            probe[M] = best_cfg
    return expand_to_all_m(probe)


def write_fp8_config(N: int, K: int, configs: dict):
    if not configs:
        print(f"  [skip] No valid configs for N={N},K={K}")
        return
    fname = f"N={N},K={K},device_name={DEVICE_NAME},dtype=fp8_w8a8,block_shape=[128,128].json"
    out = FP8_CFG_DIR / fname
    with open(out, "w") as f:
        json.dump(configs, f, indent=4)
    print(f"[tune] Written: {out}")


# ── FP8 MoE ──────────────────────────────────────────────────────────────────

def make_moe_tensors(M: int, E: int, N_int: int, K: int, top_k: int):
    bn, bk = BLOCK_SHAPE
    w1       = torch.randn(E, 2*N_int, K, dtype=torch.float16, device=DEVICE).to(FP8_DTYPE)
    w1_scale = torch.ones(E, triton.cdiv(2*N_int, bn), triton.cdiv(K, bk), device=DEVICE)
    A        = torch.randn(M, K, dtype=torch.float16, device=DEVICE).to(FP8_DTYPE)
    A_scale  = torch.ones(M, triton.cdiv(K, bk), device=DEVICE)
    C        = torch.empty(M, top_k, 2*N_int, dtype=OUT_DTYPE, device=DEVICE)
    topk_w   = torch.rand(M, top_k, device=DEVICE).softmax(dim=-1).contiguous()
    raw_eids = torch.randint(0, E, (M*top_k,), device=DEVICE)
    order    = raw_eids.argsort()
    s_tok    = torch.arange(M, device=DEVICE).repeat_interleave(top_k)[order]
    exp_ids  = raw_eids[order]
    n_pad    = torch.tensor([M*top_k], device=DEVICE, dtype=torch.int32)
    return w1, w1_scale, A, A_scale, C, topk_w, s_tok, exp_ids, n_pad


def bench_moe(M: int, E: int, N_int: int, K: int, top_k: int, cfg: dict) -> float:
    w1, w1_scale, A, A_scale, C, topk_w, s_tok, exp_ids, n_pad = \
        make_moe_tensors(M, E, N_int, K, top_k)

    def run():
        invoke_fused_moe_kernel(
            A=A, B=w1, C=C,
            A_scale=A_scale, B_scale=w1_scale, B_zp=None,
            topk_weights=topk_w, sorted_token_ids=s_tok,
            expert_ids=exp_ids, num_tokens_post_padded=n_pad,
            mul_routed_weight=False, top_k=top_k,
            config=cfg, compute_type=tl.bfloat16,
            use_fp8_w8a8=True, use_int8_w8a8=False,
            use_int8_w8a16=False, use_int4_w4a16=False,
            per_channel_quant=False, block_shape=BLOCK_SHAPE,
        )

    return gpu_ms(run)


def tune_moe(E: int, N_int: int, K: int = 2048, top_k: int = 8) -> dict:
    probe = {}
    for M in PROBE_M:
        best_ms, best_cfg = float("inf"), None
        for cfg in MOE_CONFIGS:
            ms = bench_moe(M, E, N_int, K, top_k, cfg)
            if ms < best_ms:
                best_ms, best_cfg = ms, cfg
        print(f"  M={M:5d}: {best_ms:.3f}ms  {best_cfg}", flush=True)
        if best_cfg:
            probe[M] = best_cfg
    return expand_to_all_m(probe)


def write_moe_config(E: int, N: int, configs: dict):
    if not configs:
        print(f"  [skip] No valid configs for E={E},N={N}")
        return
    fname = f"E={E},N={N},device_name={DEVICE_NAME},dtype=fp8_w8a8,block_shape=[128,128].json"
    out = MOE_CFG_DIR / fname
    with open(out, "w") as f:
        json.dump(configs, f, indent=4)
    print(f"[tune] Written: {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n=== FP8 block GEMM: N=2048, K=2048 ===")
    write_fp8_config(2048, 2048, tune_fp8_gemm(2048, 2048))

    print("\n=== FP8 block GEMM: N=2560, K=2048 ===")
    write_fp8_config(2560, 2048, tune_fp8_gemm(2560, 2048))

    print("\n=== FP8 MoE: E=128, N_int=384, K=2048, top_k=8 ===")
    write_moe_config(128, 384, tune_moe(E=128, N_int=384))

    print("\n[tune] Done. Restart vLLM to pick up the tuned configs.")
