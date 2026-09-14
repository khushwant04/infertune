# GPU validation environment — Azure A10 (NVads A10 v5)

Working configuration for validating InferTune against real vLLM, established 2026-09-14.
Every version below is load-bearing: the obvious choices do not work.

## The constraint chain

```
Azure NVadsA10_v5  ──requires──>  GRID/vGPU driver only (not the datacenter CUDA driver)
                                        │
                   Azure supports exactly vGPU 18.8 = driver 570.237
                   (v17.x/R550 is EOL for this SKU: lands on a v20.x host -> Code 43)
                                        │
                              driver 570 branch = CUDA 12.8
                                        │
                        ┌───────────────┴───────────────┐
                   torch must be cu128            vLLM wheel must link
                   (not the cu13 default)         libcudart.so.12
                        │                               │
                 torch 2.10.0+cu128              vLLM <= 0.19.1
```

## Verified working stack

| component | version | notes |
|---|---|---|
| VM size | `Standard_NV36ads_A10_v5` | full A10; smaller NV*ads sizes are **partitioned** GPUs |
| OS | Ubuntu 22.04.5 LTS | kernel 6.8.0-1064-azure, built with gcc-11 |
| security type | **Standard** | *not* Trusted Launch — Secure Boot would block the unsigned GRID module |
| driver | **570.237** (vGPU 18.8) | Azure-redistributed `.run`, licence included; no licence server needed |
| CUDA (driver) | 12.8 | reported by `nvidia-smi` |
| torch | **2.10.0+cu128** | from `download.pytorch.org/whl/cu128` |
| vLLM | **0.19.1** | newest release whose wheel links `libcudart.so.12` |
| GPU as seen | `NVIDIA A10-24Q`, sm_86 | 24512 MiB total / 23.72 GiB visible to torch, 72 SMs |

Install with `uv pip install "vllm==0.19.1" --torch-backend=cu128`.

## What does NOT work, and why

**vLLM >= 0.20.0 cannot run on this SKU at all.** Verified by inspecting the wheels:

| vLLM | links | needs driver | on this VM |
|---|---|---|---|
| 0.19.1 | `libcudart.so.12` | >= 525 | ✅ works |
| 0.20.0 | `libcudart.so.13` | >= 580 | ❌ |
| 0.24.0 | `libcudart.so.13` | >= 580 | ❌ `ImportError: libcudart.so.13` |
| 0.29.0 | `libcudart.so.13` (torch 2.13) | >= 580 | ❌ |

Three traps worth recording:

1. **The torch pin is misleading.** vLLM 0.20–0.26 all pin `torch==2.11.0`, and a
   `torch 2.11.0+cu128` build exists — so dependency resolution *succeeds* while the runtime
   still fails, because vLLM's own compiled kernels are cu13. Resolution is not validation.
2. **`--torch-backend=cu128` resolves to vLLM 0.24.0**, which then fails to import. The
   resolver only sees Python metadata, not the linked CUDA runtime.
3. **Installing a CUDA 13 runtime does not help.** CUDA 13 kernels require a >= 580 driver,
   and this SKU is capped at 570.

## Not validatable on this hardware

The A10 is sm_86. Confirmed via `torch.cuda.get_device_capability()`:

- **fp8 KV cache / fp8 weights** — need sm_89. Remain **unvalidated**.
- **MLA (DeepSeek)** — 689 GB of weights; not testable at any size here.
- **mxfp4 (gpt-oss)** — dequantises to bf16 on pre-Blackwell, ~40 GB, does not fit.

## The utilisation ceiling is not 1.0

`gpu_memory_utilization` is a fraction of **total** device memory, but only part of that is
allocatable. Measured on this VM:

```
torch total_memory        23.722 GiB
cuda mem_get_info free    21.373 GiB
vGPU/driver reserve        2.349 GiB   <- consumed before any allocation
=> real ceiling            0.901 = 21.373 / 23.722
```

A sweep confirms vLLM sizes its cache as a straight line in the fraction, with slope equal to
`total_memory`:

| gmu | KV cache | boots |
|---|---|---|
| 0.70 | 15.18 GiB | ✅ |
| 0.78 | 17.08 GiB | ✅ |
| 0.85 | 18.74 GiB | ✅ |
| 0.88 | 19.45 GiB | ✅ |
| 0.90 | 19.93 GiB | ❌ OOM in sampler warm-up |

Fitting those points gives `KV = gmu × 23.722 GiB − 1.424 GiB` to three significant figures.
At 0.90 vLLM budgets 21.35 GiB against 21.37 GiB actually free, leaving ~20 MiB — hence the
OOM. On bare metal, where the reserve is a few hundred MiB, the same formula yields ~0.97.

## Traps that cost real time here

1. **`--kv-cache-memory` does not exist in 0.19.1.** Current vLLM docs describe it; this
   version has no such flag. Compiling a plan to it produces an unparseable command line. The
   adapter introspects and falls back to inverting `--gpu-memory-utilization`.
2. **`bfloat16` is listed in `kv_cache_dtype`'s choices but breaks engine init.** Passing
   `--kv-cache-dtype bfloat16` fails; omitting it (letting vLLM derive KV dtype from the
   model) works. Appearing in `choices` is not the same as being supported.
3. **NVML reports the vGPU *profile* name, `NVIDIA A10-24Q`,** not the board name, so a
   spec-database lookup on the raw string misses and silently falls back to coarse FLOPS
   priors (critical batch size 94 instead of 130).

## First validation results

vLLM 0.19.1, `Qwen/Qwen3-0.6B`, `max_model_len=8192`, `max_num_seqs=32`,
`gpu_memory_utilization=0.85`:

| quantity | InferTune predicted | vLLM actual | error |
|---|---|---|---|
| model weights | 1.110 GiB | 1.120 GiB | **0.87%** |
| KV bytes/token | 112.00 KiB | — | — |
| KV capacity | 175,449 tokens | 175,456 tokens | **0.004%** |

## M2 acceptance measurement

`scripts/validate_vllm.py`, `max_num_seqs=32`, `max_model_len=8192`, one engine boot per model
and no load generation. Each configuration is compiled by the adapter, booted, and scored
against vLLM's own `Available KV cache memory` line:

| model | KV error | weight error | bytes/token error |
|---|---|---|---|
| Qwen3-0.6B | 0.51% | 0.87% | 0.00% |
| Qwen3-4B | 1.85% | 0.89% | 0.01% |
| Qwen2.5-7B-Instruct-AWQ | 1.84% | 0.23% | 0.02% |
| Qwen3-8B | 3.61% | 0.09% | 0.04% |

**4/4 booted, worst KV error 3.61% against the ±5% criterion — PASS.**

Two model corrections were needed to get there, both found by measurement rather than
reasoning:

- **CUDA graph pool.** A fixed 600 MiB + 4 MiB/seq prior predicted 728 MiB against a measured
  **102 MiB** — 7× high. Rescaled to `64 MiB + captured_sizes × hidden × 1200 B`, which now
  predicts 101.5 MiB.
- **Prefill activations must include the MLP intermediate,** not just the residual stream.
  Qwen2.5-7B's intermediate is 5.3× its hidden size; a hidden-only form under-estimated
  activations by ~530 MiB and alone pushed KV error to 5.07%.

Note that relative KV error is a misleading headline on weights-dominated configurations. The
absolute overhead error is roughly constant (0.1–0.5 GiB); it simply represents a larger share
of a smaller cache. Qwen3-8B leaves only ~20% of the budget as KV, which is why the same
absolute miss shows up as 3.61% there and 0.51% on Qwen3-0.6B. Fitting these coefficients per
`(gpu, model class, engine version)` is exactly what M3's measurement store is for.

Two M1 corrections confirmed on real hardware:

- **Tied-embedding dedup.** Without it we would report 1.400 GiB against vLLM's 1.120 GiB —
  **25% high**. Qwen3-0.6B ships both `lm_head.weight` and `embed_tokens.weight`.
- **The logits/sampling term is real.** At `gpu_memory_utilization=0.90` vLLM allocated
  19.82 GiB of KV and then died: `CUDA out of memory occurred when warming up sampler with
  32 dummy requests`. That is exactly the `max_num_seqs x vocab_size x 4 B` buffer the
  estimator sizes explicitly and which vLLM's own profiler under-accounts for — its log
  recommends `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1` and bumping utilisation from
  0.9000 to 0.9046 to compensate.

## Reproducing the environment

```bash
RG=rg-infratune; LOC=southeastasia
az group create -n $RG -l $LOC
az network vnet create -g $RG -n vnet-infratune -l $LOC \
  --address-prefixes 10.42.0.0/16 --subnet-name snet-gpu --subnet-prefixes 10.42.1.0/24
# NSG locked to a single source IP; VM created with NO --zone (no infrastructure redundancy)
az vm create -g $RG -n infertune -l $LOC \
  --image Canonical:0001-com-ubuntu-server-jammy:22_04-lts-gen2:latest \
  --size Standard_NV36ads_A10_v5 --security-type Standard \
  --os-disk-size-gb 128 --storage-sku Premium_LRS \
  --vnet-name vnet-infratune --subnet snet-gpu --nsg ""

# On the VM: purge nvidia, blacklist nouveau, install headers, then:
wget https://download.microsoft.com/download/5e213ec5-834f-4b0a-87f7-772751353b06/NVIDIA-Linux-x86_64-570.237-grid-azure.run
sudo sh NVIDIA-Linux-x86_64-570.237-grid-azure.run --silent --dkms
```

`Standard_NV36ads_A10_v5` is marked `NotAvailableForSubscription` in `southeastasia`, but the
restriction is **type `Zone`** — a non-zonal deployment (no `--zone`) succeeds. All regions
where the SKU is unrestricted had an NVadsA10v5 quota limit of 0.
