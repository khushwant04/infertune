# InferTune

**GPU-aware inference configuration profiler.** Given a model, a GPU, a serving framework, and a
workload, it determines a safe and near-optimal deployment configuration — and shows the memory
arithmetic that justifies it.

> **Status: M0–M4 complete.** The analyzer, memory estimator, vLLM and SGLang adapters,
> benchmark harness, calibration and configuration search all work. `infertune plan` needs no
> GPU and no model download; `infertune profile` uses detected hardware and the installed
> engine's own flags. The memory ledger is validated against real vLLM to **3.61%**; search
> matches an exhaustive grid **exactly** on 5 boots instead of 30.
>
> Two gaps stated plainly: M3's throughput-prediction accuracy (±20%) is **not yet measured**
> because it needs sustained GPU load, and **fp8 paths are unvalidated** (they require sm_89;
> the A10 used for validation is sm_86). Architecture and rationale:
> [`docs/plan.md`](docs/plan.md).

```bash
pip install -e .
infertune plan --model deepseek-ai/DeepSeek-V3 --gpu h200-sxm --gpus 8 --tp 8 \
               --max-num-seqs 64 --max-model-len 32768
```

---

## The problem

Deploying an LLM means picking values for a dozen coupled knobs:

```
tensor_parallel_size · gpu_memory_utilization · max_model_len · max_num_seqs
max_num_batched_tokens · kv_cache_dtype · block_size · quantization · graph capture
```

Get them wrong and you either OOM on boot, silently thrash on KV-cache preemption, or leave half
the GPU idle. Today this is solved by guessing, then bisecting by hand across multi-minute engine
restarts. The knowledge required — KV cache scaling, GQA vs MLA, memory bandwidth roofline,
per-framework flag semantics — is real but poorly distributed.

InferTune replaces the guessing with a memory model and a search.

## What it produces

Not a config, but a **feasible envelope plus a ledger**. The recommendation and the reason arrive
together:

```
Ledger (per GPU)                      Llama-3.1-8B · RTX 4090 24GB · vLLM · bf16
  total VRAM                             23.99 GiB  (24564 MiB, as NVML reports it)
  usable after driver/display            -23.60 GiB
  weights (safetensors-measured)          14.96 GiB
  CUDA context + NCCL                      0.55 GiB
  CUDA graph pool (-O2)                    1.20 GiB
  prefill activations @ 8192 tokens         0.85 GiB
  logits/sampling @ 24 seqs, 128256 vocab   0.35 GiB
  fragmentation safety (3%)                 0.71 GiB
  ───────────────────────────────────────────────────
  KV budget                                4.98 GiB = 40,796 tokens @ 128 KiB/token

Workload: input median 1024 / p95 2048, output median 256 / p95 512
  working set @ 24 concurrent    p50 30,092    p95 34,476    p99 36,386 tokens
  → fits at p99 in bf16, with room to raise concurrency to ~28
  BINDING CONSTRAINT: kv_working_set — the cache, not the GPU, caps concurrency
  B* ≈ 51: decode is still bandwidth-bound at this concurrency, so added
           concurrency is nearly free on latency — but KV runs out first.

  To reach the compute knee:
    kv_cache_dtype=fp8  → 81,592 tokens, carries concurrency to ~48 (≈ B*)

  (naive per-request p95 would have read 61,440 tokens — 1.78x overstated)
```

Every number traces to a term in the ledger or a term in the roofline model. The tool always
names the **binding constraint**, so it can answer "why not higher?"

Both figures above are computed by the code in this repo, not by hand — see
`tests/test_workload.py::test_readme_example_fits_in_bf16_with_headroom`. An earlier draft of
this example got them wrong in two compounding ways, which is documented in
[`docs/plan.md §2.6`](docs/plan.md) as a cautionary note.

## Design commitments

Five decisions that shape everything else. Rationale and arithmetic in
[`docs/plan.md §2`](docs/plan.md).

1. **The workload distribution is a required input, not a refinement.** Paged engines allocate KV
   per *token*, not per slot — `max_num_seqs × max_model_len` is a scheduler admission limit, not
   a reservation. The real constraint is the expected working set, which is undefined without a
   workload. Output is probabilistic (p50/p95), with preemption risk as a first-class result.
   Note that the aggregate p95 is *not* the sum of per-request p95 lengths: summing independent
   sequences concentrates the total, so composing percentiles overstates the tail by ~1.8x at
   realistic concurrency. The working set is therefore sampled, not composed.

2. **Carry bytes, never fractions.** vLLM's `--gpu-memory-utilization` covers weights +
   activations + KV; SGLang's `--mem-fraction-static` covers weights + KV only. The same `0.88`
   describes different machines. Both engines expose absolute levers
   (`--kv-cache-memory`, `--max-total-tokens`), which are the correct compilation target.

3. **Introspect the installed engine; never hardcode flag tables.** Engine flags and defaults
   drift fast enough to rot a static table in two releases, and it fails *silently*. Adapters
   hold a semantic role→flag map plus a version capability matrix.

4. **Partition the search by restart cost.** Boots cost minutes; client-side concurrency sweeps
   cost seconds. One boot yields an entire latency-throughput curve. Analytic pruning first,
   Bayesian optimization last and only over survivors.

5. **The core does not import torch.** Hardware facts enter via a `GPUProfile` sourced from NVML
   *or* a spec database, so the analytical engine is CPU-testable — and so the tool can answer
   "what would I need to serve this?" *before* you rent the GPU.

## Planned architecture

Framework-independent resource model, with thin adapters translating to engine flags. Dependency
direction is strictly downward.

```
CLI → Reporter → Search Orchestrator → { Resource Estimator, Framework Adapters }
                                              ↓                    ↓
                                       Core domain models (pure, no torch)
                                              ↓
                            { Hardware Discovery, Model Analyzer } → Measurement Store
```

Acceptance test for the abstraction: adding SGLang support must require **zero** changes outside
`adapters/`.

## Planned CLI

```bash
infertune profile  --model meta-llama/Llama-3.1-8B-Instruct \
                   --framework vllm \
                   --target-concurrency 32 \
                   --context-length 8192

infertune plan     --model Qwen/Qwen3-32B --gpu A100-80GB --count 2   # no GPU required
infertune benchmark --config recommended.yaml
infertune tune      --model ... --sla-ttft-p99 500ms --maximize throughput
```

`plan` works without any GPU present — capacity planning is a first-class mode, not a degraded one.

## Roadmap

Each milestone has a falsifiable acceptance criterion; see
[`docs/plan.md §9`](docs/plan.md).

| | scope | key acceptance criterion | status |
|---|---|---|---|
| **M0** | Skeleton, core models, units, CPU CI | core imports with no torch installed | ✅ done |
| **M1** | Model analyzer + memory estimator + ledger | weight bytes within **±1%** across 20 checkpoints | ✅ **0.0000%, 20/20** |
| **M2** | vLLM adapter + report | predicted vs vLLM-reported KV within **±5%** | ✅ **3.61%, 4/4 booted** |
| **M3** | Benchmark harness + measurement store + calibration | throughput prediction within **±20%** held-out | 🟡 built, needs GPU |
| **M4** | SGLang adapter + search | within **10%** of a grid search using **≤12** boots | ✅ **0.00% gap, 5 boots vs 30** |

Measured on an Azure A10 (`docs/gpu-validation-a10.md`), one engine boot per configuration:

| model | KV error | weight error |
|---|---|---|
| Qwen3-0.6B | 0.51% | 0.87% |
| Qwen3-4B | 1.85% | 0.89% |
| Qwen2.5-7B-Instruct-AWQ | 1.84% | 0.23% |
| Qwen3-8B | 3.61% | 0.09% |

fp8 KV paths are **unvalidated**: they need sm_89 and the A10 is sm_86.

### What M1 gets right that naive sizing does not

Each factor is asserted by a test, against a released checkpoint's own config:

| architecture | example | naive error |
|---|---|---|
| MLA latent cache | DeepSeek-V3 | **56.9× overestimate** |
| Attention/Mamba hybrid | Nemotron-H-8B | **13× overestimate** (4 of 52 layers cache KV) |
| Interleaved local/global | gpt-oss-20b | **~2× at 8K** (12 of 24 layers capped at 128 tokens) |
| Tied embeddings shipped twice | Qwen3-0.6B | **26.1% overestimate** |
| `metadata.total_size` for fp8 | DeepSeek-V3 | **1.99× overestimate** |
| Per-request p95 composed into aggregate | any | **~1.8× overestimate** |

Deliberately deferred: multi-node, disaggregated prefill/decode, speculative decoding, LoRA, MIG,
non-NVIDIA hardware.

## Scope

InferTune **recommends and benchmarks**. It does not own production deployment — it emits configs
for whatever does. Owning deployment would multiply the surface area without strengthening the
core idea.

## Documentation

- [`docs/plan.md`](docs/plan.md) — architecture, memory model derivation, performance model,
  adapter mapping table, milestones, risks, open questions.
