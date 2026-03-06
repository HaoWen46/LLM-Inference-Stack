# scripts/

Helper scripts for launching, downloading, and tuning the LLM stack.

## Quick reference

| Script | Invoked by | Purpose |
|--------|-----------|---------|
| `launch_vllm.sh` | `make vllm` | Start vLLM OpenAI-compatible server |
| `launch_llamacpp.sh` | `make llamacpp` | Start llama-cpp-python server (GGUF models) |
| `download_model.sh` | `make download` | Pull model weights from HuggingFace Hub |
| `watchdog.py` | manual | Supervisor: auto-restart crashed processes |
| `write_kernel_configs.py` | manual | Write heuristic Triton kernel configs (no GPU needed) |
| `tune_kernels.py` | manual | Benchmark-tune Triton kernel configs (GPU required) |
| `gpu_exporter.py` | replaced | Legacy Python GPU Prometheus exporter (replaced by Rust) |

---

## `launch_vllm.sh`

Starts `vllm.entrypoints.openai.api_server` with settings from `config/.env`.

```bash
make vllm
# or in the background:
bash scripts/launch_vllm.sh >> .cache/vllm.log 2>&1 &
```

Key behaviors:
- Loads `config/.env` with `set -a` so all vars are exported to vLLM
- Activates `.venv` directly (not via `uv run`) — this is intentional so manual package overrides like the cublas fix are not reverted
- Sets `HF_HOME` to `MODEL_CACHE_DIR` so HF Hub cache is on the right disk
- Redirects vLLM's compile cache to `.cache/vllm/` (avoids home quota pressure)
- Sets `CUDA_DEVICE_ORDER=PCI_BUS_ID` for consistent GPU ordering on mixed fleets
- Disables NVIDIA custom all-reduce (no NVLink on consumer GPUs)
- Passes `--enforce-eager` (required on this machine — torch.compile hangs)
- Passes `--tokenizer-mode slow` (Qwen3.5 tokenizer needs `all_special_tokens_extended`, which the fast tokenizer Rust lib dropped)
- Enables prefix caching and chunked prefill
- Passes `--limit-mm-per-prompt` when `LIMIT_MM_PER_PROMPT` is set (required for VL models running text-only)
- Appends `VLLM_EXTRA_ARGS` verbatim (use for `--reasoning-parser qwen3`, etc.)

**Required env vars** (must be in `config/.env`):
```
MODEL_NAME          Local path to weights, e.g. /home5/user/models/Qwen/Qwen3.5-35B-A3B
                    (use the full local path, not the HF repo ID, to prevent re-downloading)
VLLM_API_KEY        Internal Bearer token (gateway uses this to talk to vLLM)
TP_SIZE             Tensor parallel degree (1 or 2)
```

**Optional env vars:**
```
LIMIT_MM_PER_PROMPT   e.g. '{"image":0,"video":0}' — disable vision for VL models
VLLM_EXTRA_ARGS       e.g. '--reasoning-parser qwen3' — appended as-is to the server args
```

---

## `launch_llamacpp.sh`

Starts `llama_cpp.server` for GGUF models. Requires `LLAMACPP_MODEL` to be set.

```bash
# Set in config/.env first:
# LLAMACPP_MODEL=/home5/user/models/Qwen/Qwen3-1.7B-GGUF/Qwen3-1.7B-Q8_0.gguf
make llamacpp

# Or pass inline:
LLAMACPP_MODEL=/path/to/model.gguf make llamacpp
```

Binds to `127.0.0.1:8001` by default — access goes through the Rust gateway on `:8080`. Warns if `LLAMACPP_API_KEY` is unset (server will accept unauthenticated requests).

**Variables (all optional except `LLAMACPP_MODEL`)**:
```
LLAMACPP_MODEL       Path to .gguf file (required)
LLAMACPP_PORT        Default: 8001
LLAMACPP_HOST        Default: 127.0.0.1
LLAMACPP_API_KEY     Bearer token for auth (recommended)
LLAMACPP_GPU_LAYERS  -1 = all layers on GPU (default)
LLAMACPP_CTX         Context window size (default: 8192)
```

---

## `download_model.sh`

Downloads model weights from HuggingFace Hub to `MODEL_CACHE_DIR`.

```bash
# Download MODEL_NAME from config/.env
make download

# Download a specific model
bash scripts/download_model.sh meta-llama/Llama-3.2-7B-Instruct

# Download a single GGUF file (faster — skips huggingface_hub entirely, uses wget)
bash scripts/download_model.sh Org/Repo-GGUF filename.q8_0.gguf
```

**Two-stage download** (recommended when `MODEL_CACHE_DIR` is on slow NFS):

Set `MODEL_STAGE_DIR` to a fast local path (e.g. `/tmp/$USER/models`). Each file is downloaded to staging and immediately moved to `MODEL_CACHE_DIR` before the next file starts, so staging disk usage stays minimal throughout. If a download is interrupted, re-running skips files that already exist at the final destination. Leave `MODEL_STAGE_DIR` unset to download directly to `MODEL_CACHE_DIR`.

Skips PyTorch/Flax/TF weights and Rust model files during full repo downloads.

---

## `watchdog.py`

Monitors vLLM (and optionally the gateway) and restarts them if they crash or stop responding to health checks.

```bash
uv run python scripts/watchdog.py
```

By default it monitors vLLM at `http://localhost:$VLLM_PORT/health`. Detects two failure modes:
- **Crash**: process exits — restart immediately after `restart_delay` (default 10s)
- **Hang**: health check fails for more than `hang_timeout` (default 120s) — SIGTERM then restart

The script is a starting point — edit the `processes` list in `main()` to add the gateway or change the launch command. The `cmd` placeholder (`--config /dev/null`) must be replaced with the real vLLM launch arguments or pointed at `launch_vllm.sh` via a shell wrapper.

---

## `write_kernel_configs.py`

Writes heuristic Triton kernel config JSON files into the vLLM installation directory. **No GPU required.** Suppresses vLLM's "sub-optimal kernel config" warnings on startup.

```bash
uv run python scripts/write_kernel_configs.py
```

Generates configs for:
- FP8 block GEMM shapes used by Qwen3/Qwen3.5 FP8 variants (`N=2048,K=2048` and `N=2560,K=2048`)
- FP8 MoE fused kernel (E=128 experts, N_intermediate=384)

Configs are derived from H100/H20 patterns, adapted for Ada Lovelace (sm_89 / RTX 4090). They provide per-batch-size `BLOCK_SIZE_M` selection across 18 batch sizes from 1 to 4096.

Run this once after installing vLLM. For measured-optimal configs, run `tune_kernels.py` afterwards (overwrites these files).

---

## `tune_kernels.py`

Benchmark-tunes Triton kernel configs for FP8 GEMM and FP8 MoE on the **current GPU**. Overwrites the heuristic files written by `write_kernel_configs.py`.

```bash
# Run on GPU 0 while vLLM uses GPUs 1 and 2
CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_ORDER=PCI_BUS_ID \
  uv run python scripts/tune_kernels.py
```

Strategy:
- Benchmarks 5 representative M values: `[1, 16, 64, 256, 2048]`
- Searches 12 configs per M (varying `BLOCK_SIZE_M` and `GROUP_SIZE_M`)
- Extrapolates to all 18 standard batch sizes by nearest-M lookup
- ~120 kernel compilations total — takes **5–10 minutes** depending on Triton cache state

Output is written directly into the vLLM package directory (same location as `write_kernel_configs.py`). Restart vLLM after this runs.

---

## `gpu_exporter.py`

Legacy Python GPU Prometheus exporter. **Replaced by the Rust `gpu-exporter` crate** (`rust/gpu-exporter/`), which is faster and has zero Python overhead.

Use `make gpu-exporter` instead. This file is kept for reference only.
