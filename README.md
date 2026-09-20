# InferTune

**GPU-aware inference configuration profiler.** Given a model, a GPU, a serving framework and a
workload, InferTune determines a safe, near-optimal deployment configuration — and shows the memory
arithmetic that justifies it.

It answers three questions that are normally answered by guessing:

- Will this model fit, and at what concurrency?
- What is actually the limit — VRAM, the KV cache, or compute?
- What would I need to rent in order to serve this? (asked *before* renting it)

---

## Install

Requires Python 3.11+. No GPU and no CUDA toolchain needed for planning.

```bash
git clone https://github.com/khushwant04/infertune.git
cd infertune
pip install -e .
```

## Quickstart

`plan` needs no GPU and downloads no weights — model facts come from checkpoint metadata read over
HTTP range requests, hardware facts from a built-in specification database.

```bash
infertune plan --model Qwen/Qwen3-8B --gpu a10 --max-num-seqs 64 --max-model-len 8192
```

```
Qwen/Qwen3-8B on 1x NVIDIA A10  (tp=1)
AttentionSpec, 36/36 layers cache state; weights measured via safetensors-header (sharded)

term                       bytes  formula
total VRAM             23.94 GiB  device capacity (binary units)
usable after driver    21.57 GiB  total - driver/display reservation
driver/vGPU reserve     2.37 GiB  total - torch-allocatable (outside the engine's budget)
model weights          15.26 GiB  measured checkpoint bytes / tp=1
CUDA graph pool       364.00 MiB  64.00 MiB base + 64 captured sizes x hidden/tp x 1200 B
attention workspace   128.00 MiB  attention backend scratch
prefill activations   512.00 MiB  8192 tokens x (hidden+intermediate)/tp x 2 elements
logits + sampling     111.28 MiB  64 seqs x 151936 vocab x 4 B x 3
fragmentation safety  662.56 MiB  3% of usable VRAM
KV cache budget         4.58 GiB  usable - everything above

KV capacity             33,314 tokens (144.00 KiB/token, bf16)
concurrency headroom    23 (requested 64)
critical batch size B*  130
binding constraint      kv_working_set — the KV cache budget, not compute, limits concurrency
verdict                 cache is the wall: a smaller KV dtype or more GPUs would help; a faster
                        GPU would not

! composing per-request p95 lengths would have read 163,840 tokens against an actual aggregate
p95 of 87,343 (1.88x overstated)
```

The output is not a config but a **ledger**: the recommendation and the reason arrive together.
Every number traces to a term in the memory model or a term in the roofline model, and the tool
always names the **binding constraint** — so it can answer "why not higher?"

Run `infertune gpus` to list the 16 GPUs in the specification database.

## The problem

Deploying an LLM means picking values for a dozen coupled knobs:

```
tensor_parallel_size · gpu_memory_utilization · max_model_len · max_num_seqs
max_num_batched_tokens · kv_cache_dtype · block_size · quantization · graph capture
```

Get them wrong and you either OOM on boot, silently thrash on KV-cache preemption, or leave half
the GPU idle. The usual remedy is to bisect by hand across multi-minute engine restarts. The
knowledge required — KV cache scaling, GQA vs MLA, the memory-bandwidth roofline, per-framework
flag semantics — is real, but poorly distributed.

Naive sizing is not off by a few percent. On DeepSeek-V3's MLA cache it overestimates by **56.9x**;
on Nemotron-H's attention/Mamba hybrid by **13x**. InferTune models the architectures individually
and [validates the arithmetic against real engines](docs/status.md).

## CLI

| command | GPU needed | what it does |
|---|---|---|
| `plan` | no | Size a configuration for hardware you do not have in hand. |
| `profile` | yes | Same, on detected hardware, using the installed engine's own flags. |
| `tune` | no | Rank candidates analytically. Dry-run only; `--execute` fails clearly until engine lifecycle support exists. |
| `benchmark` | yes | Sweep concurrency against a running engine and record the curve. |
| `kv` | no | KV cache cost per token for a model. |
| `working-set` | no | Aggregate in-flight KV tokens for a workload. |
| `gpus` | no | List GPUs known to the specification database. |

`tune` enumerates the space freely, prunes it analytically, and only then spends boots:

```console
$ infertune tune --model Qwen/Qwen3-8B --gpu a10 --boots 5
enumerated 15 -> 5 to boot (3x fewer)
  0 infeasible (no KV budget), 10 dominated (another candidate is at least as good on KV
  capacity and predicted throughput)

#  configuration                   KV tokens  pred tok/s  pred TTFT  headroom
1  tp=1 len=4096 seqs=256 kv=bf16     26,361        1895      280ms        18
2  tp=1 len=4096 seqs=128 kv=bf16     32,210        1495      280ms        22
3  tp=1 len=4096 seqs=64 kv=bf16      35,135        1052      280ms        24
4  tp=1 len=4096 seqs=32 kv=bf16      36,597         660      280ms        25
5  tp=1 len=4096 seqs=16 kv=bf16      37,328         378      280ms        26

dry run: would boot 5 configurations and sweep concurrency on each. Engine execution is not
implemented yet; `--execute` exits with a clear error instead of pretending to measure.
```

Against a separate 30-candidate **synthetic** objective with a known ground truth, this strategy
selected the same winner after **5 simulated boots instead of 30**, with a **0.00%** throughput
gap. This validates the pruning algorithm, not real engine execution.

## Design commitments

Five decisions shape everything else. Full rationale and arithmetic in
[`docs/plan.md §2`](docs/plan.md).

1. **The workload distribution is a required input, not a refinement.** Paged engines allocate KV
   per *token*; `max_num_seqs × max_model_len` is a scheduler admission limit, not a reservation.
   The real constraint is the expected working set, which is undefined without a workload. Output
   is probabilistic (p50/p95/p99), with preemption risk as a first-class result. The aggregate p95
   is *sampled*, never composed from per-request p95 lengths — independent sequences concentrate,
   so composing percentiles overstates the tail by ~1.8x at realistic concurrency.

2. **Carry bytes, never fractions.** vLLM's `--gpu-memory-utilization` covers weights +
   activations + KV; SGLang's `--mem-fraction-static` covers weights + KV only. The same `0.88`
   describes different machines. Absolute levers are the correct compilation target.

3. **Introspect the installed engine; never hardcode flag tables.** Flags and defaults drift fast
   enough to rot a static table in two releases, and they fail *silently*. Adapters hold a semantic
   role→flag map plus a version capability matrix. This has already paid off twice: vLLM 0.19.1
   lacks the `--kv-cache-memory` flag the plan assumed, and SGLang relocated its entire argument
   surface between releases.

4. **Partition the search by restart cost.** Boots cost minutes; client-side concurrency sweeps
   cost seconds, and one boot yields an entire latency–throughput curve. Analytic pruning first;
   measurement only over survivors.

5. **The core does not import torch.** Hardware facts enter via a `GPUProfile` sourced from NVML
   *or* a specification database, so the analytical engine is fully CPU-testable — and so the tool
   can size a deployment before the GPU exists. Enforced by an AST test, not by convention.

## Architecture

A framework-independent resource model, with thin adapters translating to engine flags. Dependency
direction is strictly downward.

```
CLI → Reporter → Search Orchestrator → { Resource Estimator, Framework Adapters }
                                              ↓                    ↓
                                       Core domain models (pure, no torch)
                                              ↓
                            { Hardware Discovery, Model Analyzer } → Measurement Store
```

```
src/infertune/
  core/        units, dtypes, cache, model, gpu, plan, workload   (stdlib only)
  models/      safetensors header reader, HF config analyzer
  estimator/   memory ledger, roofline, calibration
  hardware/    NVML discovery, GPU specification database
  adapters/    vllm, sglang, registry
  bench/       load generator, metrics, harness
  search/      space enumeration, pruning engine
  store/       SQLite measurement store
```

The abstraction was tested by adding SGLang: it required **zero** changes outside `adapters/`.

## Scope

InferTune **recommends and benchmarks**. It does not own production deployment — it emits configs
for whatever does. Owning deployment would multiply the surface area without strengthening the core
idea.

Out of scope by choice: multi-node, disaggregated prefill/decode, speculative decoding, LoRA, MIG,
non-NVIDIA hardware.

## Status

Milestones M0–M4 have implementation work in the repository, but their evidence is not all the
same. M2's memory ledger was validated against running vLLM on an A10 to **3.61%** worst case.
M3's throughput criterion remains unmeasured. M4's pruning strategy matched an exhaustive
**synthetic** grid exactly; real SGLang validation and engine-backed `tune --execute` remain
outstanding. Until engine lifecycle support exists, `--execute` fails explicitly.

FP8 paths are also unvalidated: they require sm_89, while the A10 used for validation is sm_86.

Full breakdown, measured numbers and remaining work: [`docs/status.md`](docs/status.md).

## Development

```bash
pip install -e ".[dev]"

ruff check src tests scripts
ruff format --check src tests scripts
mypy src tests scripts
pytest                              # default CPU-only suite
```

The default suite needs no GPU, no model download and no serving engine. Network-dependent tests
are opt-in:

```bash
pytest -m network --run-network
```

## Documentation

- [`docs/plan.md`](docs/plan.md) — architecture, memory model derivation, performance model, adapter
  mapping tables, milestones, risks, open questions.
- [`docs/status.md`](docs/status.md) — what is built, what is measured, what remains.
- [`docs/gpu-validation-a10.md`](docs/gpu-validation-a10.md) — the validation environment, the
  driver/vLLM/torch compatibility chain that constrains it, and how to reproduce it.
