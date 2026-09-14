# InferTune — GPU-aware inference configuration profiler

**Plan and architecture.** This document is the design record: what was decided, why, and which
assumptions measurement later overturned (§2.6). It is not a status report — for what is built and
what is verified, see [`status.md`](status.md).

> One-line scope: given `(model, GPU(s), serving framework, workload, SLA)`, produce a
> *feasible* and *near-optimal* engine configuration, together with the memory ledger and
> assumptions that justify it.

---

## 1. What "correct" means here

The tool is only useful if it is trustworthy, and trust comes from being right about memory.
So the success criteria are ordered:

1. **No OOM.** A recommended config that crashes on boot or under load is a total failure.
   This dominates everything else.
2. **No silent thrashing.** A config that boots but spends its life preempting/recomputing is
   almost as bad, and much harder for a user to notice. Preemption risk is a first-class output.
3. **Explainable.** Every number in the output traces back to a term in a memory ledger or a
   term in a performance model. No unexplained magic constants.
4. **Then** near-optimal throughput under the latency SLA.

Consequence for the design: the primary artifact is not a config, it is a **feasible envelope
plus a ledger**. The single recommended config is a point chosen inside that envelope.

---

## 2. Five design decisions

These are the decisions worth making up front, because they're expensive to retrofit. Two of
them correct assumptions in the original sketch.

### 2.1 `max_num_seqs × max_model_len` is *not* the memory constraint

This is the most important correction, and it's worth walking through the arithmetic.

For Llama-3.1-8B (32 layers, 8 KV heads, head_dim 128, bf16):

```
kv_bytes_per_token = 2 (K,V) × 8 kv_heads × 128 head_dim × 2 bytes × 32 layers
                   = 131,072 bytes = 128 KiB / token
```

The MVP output in the original sketch proposed `max_num_seqs=24`, `max_model_len=8192` with
5.4 GB of KV cache. But:

```
24 seqs × 8192 tokens × 128 KiB = 24 GiB of KV cache
5.4 GiB of KV cache            = 44,236 tokens ≈ 5.4 seqs at full 8K context
```

Those numbers are inconsistent by ~4.4×, *if* you read `max_num_seqs` as a worst-case
reservation. And a naive profiler built on the worst-case reading would have recommended
`max_num_seqs=5` — leaving the GPU badly underutilized for any realistic workload.

The resolution: paged-attention engines allocate KV per *token*, not per *slot*, and
`max_num_seqs` is a scheduler admission limit, not a reservation. vLLM handles KV exhaustion by
preempting and recomputing rather than refusing to boot
([vLLM preemption docs](https://docs.vllm.ai/en/stable/configuration/optimization/)). So
`max_num_seqs=24` is genuinely fine here *for a workload whose in-flight sequences average
~1.15K tokens* (24 × 1150 × 128 KiB ≈ 3.4 GiB, comfortably inside 5.4 GiB) and genuinely bad for
a workload that actually fills 8K contexts.

**So the constraint is on the expected working set, and it is undefined without a workload
distribution.** This makes the workload profile a required input, not an optional refinement,
and it makes the correct output a probabilistic statement:

```
KV working set  p50: 30,092   p95: 34,476   p99: 36,386 tokens
available:      40,796 tokens (4.98 GiB @ 128 KiB/token)
→ fits at p99; headroom to raise concurrency to ~28
→ binding constraint: kv_working_set — the cache caps concurrency, not the GPU
```

That output is honest and actionable. `max_num_seqs: 24` alone is neither.

One important refinement, learned while implementing this (see §2.6): the aggregate p95 must be
**sampled from the joint distribution, not composed from per-request p95 lengths**. Summing many
independent sequences concentrates the total, so composing percentiles overstates the tail — by
1.78× in the example above, and the error grows with concurrency.

### 2.2 Target the engines' *direct* memory levers, not the fractions

Both engines now expose an absolute KV budget, which is a far better compilation target than
the fractional knobs:

- vLLM: `--kv-cache-memory` (exact bytes). Passing it skips vLLM's own profiling pass and the
  CUDA-graph memory estimation pass. vLLM logs the value that reproduces its current
  allocation — which is also our best calibration signal (see §6).
- SGLang: `--max-total-tokens` (size of the token memory pool).

The fractional knobs are the trap, because **their semantics differ and are not
interconvertible**:

| | denominator | what it covers |
|---|---|---|
| vLLM `--gpu-memory-utilization` | fraction of *total* GPU memory | the whole vLLM instance: weights + activations + KV |
| SGLang `--mem-fraction-static` | fraction of *total* GPU memory | *static* only: weights + KV pool — activations live outside it |

So `gpu_memory_utilization: 0.88` and `mem_fraction_static: 0.88` describe different machines.
Any system that treats them as a portable "memory fraction" field will produce configs that are
conservative on one engine and OOM on the other. This single table is the strongest argument for
the adapter layer: the framework-independent model must carry **bytes**, and each adapter solves
for its own knob.

### 2.3 Derive the parameter schema from the installed engine

The #1 long-term maintenance risk is flag drift. vLLM's V1 engine changed defaults and flag
names substantially (prefix caching became default-on, so the operative flag is now
`--no-enable-prefix-caching` to *disable* it; optimization levels `-O0..-O3` and `--kv-cache-memory`
are recent additions). A hardcoded flag table rots within two releases and fails silently.

Instead: **introspect the installed engine at runtime** to build the parameter schema
(`vllm.EngineArgs` dataclass fields / the argparse spec; SGLang's `ServerArgs`). The adapter
then holds only (a) the semantic mapping from our model to a parameter *role*, and (b) a
per-version capability matrix. Unknown-to-the-installed-version parameters are dropped with a
warning rather than emitted into a command line that won't parse. Adapters record the engine
version they resolved against, and every stored benchmark result is keyed by it.

### 2.4 Partition the search space by restart cost

Benchmark cost is dominated by engine startup — weight load, `torch.compile`, CUDA graph
capture — which is 30 s to several minutes, versus seconds for a load-generator sweep. Treating
all knobs as one flat vector for Bayesian optimization wastes almost all of the budget on
process launches.

So split the space:

- **Outer loop (restart required):** `tensor_parallel_size`, `kv_cache_dtype`, quantization,
  `max_model_len`, KV budget, CUDA-graph settings. Expensive. Few candidates, chosen analytically.
- **Inner loop (no restart):** client-side concurrency and request-rate sweeps, and any knob the
  engine can vary per-request. Cheap. Sweep densely to build a throughput/latency curve and find
  the SLA knee for *free*.

One restart can yield an entire latency-throughput curve. This roughly inverts the cost model
that makes brute force look hopeless in the original sketch.

### 2.5 The core must not import torch

The estimator, the model analyzer, and the adapters' config *generation* have zero GPU
dependencies. Hardware facts enter through a `GPUProfile` that can come from NVML *or* from a
spec-sheet database. This means:

- The whole analytical engine is unit-testable in CI on CPU-only runners (relevant: this very
  sandbox has no GPU).
- The tool answers "what would I need to serve Qwen3-32B?" *before* you rent the GPU — which is
  arguably the highest-value use case, and it's lost if hardware detection is mandatory.

`pynvml`, `torch`, `vllm`, `sglang` are all optional extras.

### 2.6 Two errors this plan originally contained

Both were found during M0 implementation, when the code was made to reproduce this document's own
worked example. Both were *conservative* — neither would have caused an OOM — and that is exactly
what makes them instructive: a profiler that is quietly too cautious looks like it is working
while it wastes a third of the GPU.

**Error 1: VRAM was read as decimal GB.** GPU memory is quoted in binary units, because memory is
manufactured in powers of two. A "24 GB" RTX 4090 reports 24564 MiB ≈ 23.99 GiB, and an "80 GB"
H100 offers roughly 85.5 *decimal* GB
([why an H100 80GB offers 85.52 GB](https://thundergolfer.com/blog/nvidia-gpu-memory-capacity);
an [RTX 4090 reporting 24564 MiB](https://github.com/vllm-project/vllm/issues/7553)). Reading the
nameplate as decimal understates capacity by 7.4% — 1.65 GiB on a 24 GB card, which is 32% of that
card's KV budget once weights are subtracted, or about 9,900 tokens for Llama-3.1-8B.

This is the opposite of the disk-capacity convention, which is why it is easy to get backwards.
`units.vram_nameplate()` exists solely to make the choice explicit at every call site.

**Error 1b (found in M1): tied embeddings must be deduplicated.** Qwen3-0.6B declares
``tie_word_embeddings: true`` *and* ships both `lm_head.weight` and
`model.embed_tokens.weight`. Only one copy is resident at inference time, so summing every
tensor overstates resident weights by a whole 151936x1024 matrix — **26.1%** on that model.
Note that Qwen3-0.6B-**Base** omits `lm_head` entirely, so the correction is checkpoint-specific
and cannot be inferred from the config alone. This is the strongest argument for measuring
tensor metadata rather than deriving from parameter counts.

**Error 1c (found in M1): the checkpoint's own `metadata.total_size` cannot be trusted.**
The safetensors shard index publishes a `total_size` field, which is tempting to use as a
one-request byte total. For deepseek-ai/DeepSeek-V3 it claims **1369 GB** while the tensor
headers sum to **689 GB** — a **1.99x** overstatement, because the field is computed as if
every tensor were 16-bit and the checkpoint is fp8. An estimator trusting it would have
declared DeepSeek-V3 infeasible on 8xH200 when it comfortably fits.

The fix is to read every shard header and sum `data_offsets`, parallelised — 163 shards and
91,991 tensors complete in about 8 seconds. The index's claim is still read, and any
disagreement is surfaced as a diagnostic, which makes the tool more accurate than the
checkpoint's own metadata.

**Error 2: aggregate percentiles were composed from per-request percentiles.** The original
example computed a p95 working set as `concurrency × (input_p95 + output_p95)`. That is not a p95
of anything. The sum of many independent sequence lengths concentrates around its mean, so the
aggregate tail is far tighter than the per-request tail — 34,476 tokens rather than 61,440 in the
worked example, a 1.78× overstatement that grows with concurrency.

Together the two errors turned "fits at p99 with room to grow to ~28 concurrent" into "preempts
at p95, switch to fp8 KV" — a recommendation to trade away numerical precision to solve a problem
that did not exist.

The structural lessons, both now enforced by tests:

* Ledgers must **reconcile** against a measured usable-VRAM figure
  (`ResourcePlan.reconciles_with`). A ledger that does not sum correctly has a missing or
  double-counted term.
* Worked examples in documentation must be **generated by the code**, not written by hand. The
  appendix's example is asserted in `tests/test_workload.py`, so it cannot drift from the
  implementation.

---

## 3. Architecture

Dependency direction is strictly downward; nothing below imports anything above.

```
                         CLI  (infertune profile | plan | benchmark | tune)
                                          │
                              ┌───────────┴───────────┐
                              │       Reporter        │  ledger, envelope, risks, YAML
                              └───────────┬───────────┘
                                          │
                    ┌─────────────────────┴─────────────────────┐
                    │            Search Orchestrator            │
                    │  analytic prune → outer loop → inner loop │
                    └──────┬─────────────────────────┬──────────┘
                           │                         │
              ┌────────────▼───────────┐  ┌──────────▼───────────┐
              │   Resource Estimator   │  │  Framework Adapters  │
              │  memory ledger + perf  │  │   vllm | sglang      │
              └────────────┬───────────┘  └──────────┬───────────┘
                           │                         │
                    ┌──────▼─────────────────────────▼──────┐
                    │        Core domain models (pure)       │
                    │  GPUProfile · ModelProfile ·           │
                    │  WorkloadProfile · ResourcePlan ·      │
                    │  UniversalConfig · Measurement         │
                    └──────┬─────────────────┬───────────────┘
                           │                 │
              ┌────────────▼──────┐  ┌───────▼────────────┐
              │ Hardware Discovery│  │  Model Analyzer    │
              │ NVML | spec DB    │  │  HF config | local │
              └───────────────────┘  └────────────────────┘
                           │
                    ┌──────▼──────────────────────────┐
                    │  Measurement store (SQLite)     │
                    │  + calibration coefficients     │
                    └─────────────────────────────────┘
```

Package layout:

```
infertune/
  core/          # dataclasses/pydantic models, units, no I/O, no torch
  hardware/      # nvml.py, specdb.py (YAML: bandwidth, FLOPS, SMs, dtype support), topology.py
  models/        # hf.py (config.json + safetensors header), arch/{dense,gqa,mla,moe,hybrid}.py
  workload/      # profiles, trace ingestion, distributions
  estimator/     # memory.py, roofline.py, calibration.py
  adapters/      # base.py, vllm.py, sglang.py, introspect.py
  search/        # prune.py, outer.py, inner.py, objective.py
  bench/         # loadgen.py, harness.py, metrics.py
  store/         # sqlite measurement store
  report/        # rich console + YAML/JSON emitters
  cli.py
```

---

## 4. Core data models

Framework-independent, and deliberately carrying **bytes and distributions**, not fractions and
point estimates.

```python
@dataclass(frozen=True)
class GPUProfile:
    name: str
    count: int
    vram_bytes: int                 # per device, total
    vram_usable_bytes: int          # minus driver/display/co-tenant reservation
    compute_capability: tuple[int, int]
    sm_count: int
    mem_bandwidth_bytes_s: float
    dense_flops: dict[str, float]   # {"bf16": 9.9e14, "fp8": 1.98e15, ...}
    supports: frozenset[str]        # {"bf16","fp8_e4m3","fp8_kv","nvfp4",...}
    interconnect: Interconnect      # NVLINK | PCIE_GEN4_X16 | ...  + measured/​spec bytes/s
    source: Literal["nvml", "specdb"]   # provenance is part of the data

@dataclass(frozen=True)
class ModelProfile:
    id: str
    n_params_total: int
    n_params_active: int            # differs for MoE; drives compute, not memory
    weight_bytes: int               # measured from safetensors header, not params × dtype
    n_layers: int
    hidden_size: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    vocab_size: int
    max_position_embeddings: int
    attn: AttentionSpec             # MHA | GQA | MLA | sliding-window | hybrid layer pattern
    moe: MoESpec | None
    quant: QuantSpec | None         # method, group size, which tensors are excluded
    tied_embeddings: bool

@dataclass(frozen=True)
class WorkloadProfile:
    input_tokens: Distribution      # not a scalar
    output_tokens: Distribution
    target_concurrency: int | None
    target_rps: float | None
    shared_prefix_tokens: int       # prefix-cache hit potential
    sla: SLA                        # ttft_p99_ms, tpot_p99_ms, or throughput-max

@dataclass(frozen=True)
class ResourcePlan:                 # the framework-independent answer
    weight_bytes_per_gpu: int
    activation_peak_bytes: int
    fixed_overhead_bytes: int       # CUDA ctx, NCCL, cudagraph pool, compile workspace
    safety_bytes: int
    kv_budget_bytes: int            # the number adapters compile down to
    kv_budget_tokens: int
    parallelism: Parallelism        # tp, pp, dp, ep
    dtypes: DtypePlan               # weights, activations, kv
    ledger: list[LedgerEntry]       # every term, with formula + provenance
    binding_constraint: str         # "kv_working_set_p95" | "weights" | "sla_tpot" | ...
```

`Distribution` and `binding_constraint` are the two fields that make the output explainable.
The report can always answer "why not higher?" by naming the binding constraint.

---

## 5. Memory model

### 5.1 Weights: read the safetensors header, don't multiply

`params × bytes_per_param` is wrong for exactly the models people most need help with.
Quantized checkpoints leave embeddings, norms, and often `lm_head` in bf16, and they add
scales/zero-points (AWQ 4-bit at group size 128 costs ≈ 4.15 bits/weight, not 4). MoE
checkpoints need *total* params for memory but *active* params for compute. Tied embeddings
change the count.

The fix is cheap and exact: the safetensors header is a JSON blob at the start of the file
listing every tensor's dtype, shape, and byte offsets, and it is fetchable with an HTTP range
request without downloading any weights
([HF metadata parsing docs](https://huggingface.co/docs/safetensors/en/metadata_parsing)).
Sum `sizeof(dtype) × prod(shape)` over the shard index and you have the real number, for any
model on the Hub, in about a megabyte of traffic.

This is a small implementation detail with a large accuracy payoff, and it should be built in
M1 rather than deferred.

### 5.2 KV cache: per-architecture, not one formula

```
kv_bytes_per_token = 2 × ceil(n_kv_heads / tp) × head_dim × bytes_per_elem × n_kv_layers
```

Note `ceil(n_kv_heads / tp)`: when `n_kv_heads < tp`, KV heads are *replicated* across ranks,
so TP stops reducing per-GPU KV and starts wasting it. A profiler that divides naively will
recommend TP=8 for an 8-KV-head model and mispredict its own memory.

The single formula does not survive contact with modern architectures, so `AttentionSpec`
dispatches:

- **MHA/GQA** — the formula above.
- **MLA** (DeepSeek-family) — compressed latent KV; roughly an order of magnitude smaller. The
  GQA formula overestimates by ~10×, which turns into a wildly conservative config.
- **Sliding-window / hybrid** — only some layers retain full-length KV (e.g. interleaved
  local:global patterns). `n_kv_layers` becomes an effective, context-dependent count.
- **Mamba/SSM hybrids** — fixed-size recurrent state per sequence, not growing with context.
  Different scaling law entirely: memory is per-*sequence*, so `max_num_seqs` really *is* the
  reservation here. (§2.1's argument inverts for these models — the abstraction must allow it.)
- **fp8 KV** — halves it, plus per-block scale overhead, and gate on compute capability.

### 5.3 The terms everyone forgets

Sized explicitly, because these are the actual causes of "it worked at batch 32 and OOMed at 64":

| term | scale | note |
|---|---|---|
| logits + sampling buffers | `max_num_seqs × vocab_size × 4 B` × k | 256 seqs × 256K vocab × 4 B = 256 MiB *per buffer*; several exist. Large-vocab models make this a GiB-scale term. |
| prefill activations | `max_num_batched_tokens × hidden × bytes × k` | scales with the *token budget*, not `max_num_seqs` |
| CUDA graph pool | ~0.5–3 GiB | depends on captured batch sizes; `--enforce-eager` / `-O0` removes it |
| torch.compile / inductor workspace | 0.1–1 GiB | varies by optimization level |
| NCCL / comm buffers | ~100s of MiB per rank | scales with TP degree |
| attention backend workspace | varies | FlashInfer/FA3 scratch |
| allocator fragmentation | 2–5% | why the safety buffer is not superstition |
| multimodal processor cache | 4 GiB default in vLLM | host-side, but a real OOM source on the box |

### 5.4 Inverting the engine's own allocator

The estimator's real job is subtler than "sum the terms": it must **invert the engine's
algorithm**. vLLM measures free memory at init, runs a profiling forward pass, and derives the
KV cache as `gpu_memory_utilization × total − (weights + measured peak activations)`. To
recommend a fraction, we must predict what vLLM will measure. That's fragile.

Recommending `--kv-cache-memory` in bytes sidesteps the inversion entirely — we assert the
budget rather than trying to predict a measurement. The fractional knob is emitted only as a
secondary, clearly-labelled fallback for older versions. This is decision §2.2 paying off.

---

## 6. Performance model: roofline first, calibrate second

Analytical, physically grounded, two coefficients — deliberately *not* a black-box learned model.
A model with two interpretable coefficients can be calibrated from a handful of measurements and
can explain itself; a learned surrogate needs hundreds and can't.

**Decode is bandwidth-bound.** Per step, per GPU:

```
bytes  = weight_bytes/tp + batch × avg_context × kv_bytes_per_token
t_mem  = bytes / (bandwidth × MBU)
```

**Prefill is compute-bound.**

```
flops  = 2 × n_params_active × prompt_tokens        (+ 4 · n_layers · N² · hidden for long N)
t_comp = flops / (dense_flops × MFU)
```

**Per step:** `t = max(t_mem, t_comp) + t_comm + t_fixed`, where `t_comm` needs *both* a
bandwidth term and a per-call latency term — TP does 2 all-reduces per layer, so a 64-layer
model pays ~128 synchronizations per step. On NVLink the bandwidth term dominates and is small;
on PCIe the latency term alone can cost milliseconds. That's the quantitative version of
"topology matters," and it's what lets the tool say *don't use TP on this box* with a reason.

**Critical batch size** — where decode flips from bandwidth- to compute-bound:

```
B* ≈ (dense_flops × MFU) / (bandwidth × MBU)
```

For H100 SXM (≈990 TFLOPS dense bf16, 3.35 TB/s), with MFU 0.5 / MBU 0.8: `B* ≈ 185`. This is
the single most useful number in the whole system, because it tells you whether raising
concurrency buys throughput (below `B*`: nearly free) or only buys latency (above `B*`).
It should be printed in every report.

**Calibration.** `MFU` and `MBU` start as per-architecture priors and are refined per
`(gpu, model_arch_class, framework_version)` from the measurement store, as a small
multiplicative correction on physically-derived features. Predictions are reported as intervals
whose width reflects how much calibration data exists — no data, wide interval, stated as such.
Also worth harvesting: vLLM logs the exact `--kv-cache-memory` value reproducing its own
allocation, which is a direct ground-truth label for the memory model, obtainable from any
single boot with no benchmarking at all.

---

## 7. Adapters

Adapter responsibilities, thin and testable:

```python
class FrameworkAdapter(Protocol):
    version: str
    def schema(self) -> ParamSchema: ...                       # introspected, §2.3
    def capabilities(self) -> Capabilities: ...                # fp8 kv? MLA? chunked prefill?
    def compile(self, plan: ResourcePlan, uc: UniversalConfig) -> LaunchSpec: ...
    def validate(self, spec: LaunchSpec) -> list[Diagnostic]: ...
    def launch(self, spec: LaunchSpec) -> ServerHandle: ...     # optional extra
    def parse_startup_log(self, text: str) -> StartupFacts: ... # calibration harvest
```

Mapping table (roles → flags). Spellings are **validated against the installed version at
runtime**, per §2.3; treat this as the semantic map, not a source of truth for flag strings.

| role | vLLM | SGLang |
|---|---|---|
| absolute KV budget | `--kv-cache-memory` (bytes) | `--max-total-tokens` (tokens) |
| memory fraction | `--gpu-memory-utilization` (whole instance) | `--mem-fraction-static` (weights + KV only) |
| max context | `--max-model-len` | `--context-length` |
| concurrent request cap | `--max-num-seqs` | `--max-running-requests` |
| prefill token budget | `--max-num-batched-tokens` | `--chunked-prefill-size` |
| tensor parallel | `--tensor-parallel-size` | `--tp-size` |
| pipeline parallel | `--pipeline-parallel-size` | `--pp-size` |
| data parallel | `--data-parallel-size` | `--dp-size` |
| expert parallel | `--enable-expert-parallel` | expert-parallel flags — verify per version |
| KV dtype | `--kv-cache-dtype` | `--kv-cache-dtype` |
| prefix caching | default on; `--no-enable-prefix-caching` disables | RadixAttention on; `--disable-radix-cache` |
| graph capture | `--enforce-eager`, `-O0..-O3` | `--disable-cuda-graph`, `--cuda-graph-max-bs` |
| page/block size | `--block-size` | `--page-size` |

Adapters also own **environment sanity checks**, which are cheap and catch real production
misconfiguration:

- CPU cores: vLLM V1 needs ≥ `2 + N` physical cores for N GPUs (API server + engine core +
  one worker per GPU); with data parallelism, `A + DP + N + (1 if DP>1)`. Underprovisioning here
  silently caps throughput and is invisible in GPU metrics — a classic "GPU util is low, why?"
- NUMA: multi-socket boxes want workers pinned near their GPU (`--numa-bind`).
- Topology: TP across a PCIe boundary or an unexpected NUMA hop.

Adding a third engine (TensorRT-LLM, LMDeploy) must require touching only `adapters/` and a
capability matrix. That's the acceptance test for the abstraction.

---

## 8. Search

```
1. Enumerate      parallelism × quant × kv_dtype × context   (small, discrete, structured)
2. Analytic prune Drop infeasible (memory ledger) and dominated (roofline) candidates.
                  Typically 1000s → single digits. This is where the estimator earns its keep.
3. Outer loop     For each surviving candidate: boot once. Harvest startup facts.
4. Inner loop     Sweep client concurrency / rate without restarting.
                  → full throughput-vs-latency curve; find the SLA knee.
5. Select         Pareto front over (throughput, p99 latency); pick by the user's objective.
6. Record         Persist every measurement; update calibration coefficients.
```

Bayesian optimization arrives in M4, and only over the *continuous* knobs that survive step 2,
with the constraint handled properly (constrained EI, or a feasibility classifier — an OOM is a
crashed trial, not a bad score, and treating it as a large penalty teaches the optimizer the
wrong shape). Structure exploitation comes first: throughput is roughly monotone-then-flat in
the token budget and roughly unimodal in concurrency, so coordinate ascent with early stopping
beats generic BO at these sample sizes.

---

## 9. Milestones

Each has a falsifiable acceptance criterion. No milestone is "done" because code exists.

**M0 — Skeleton (0.5 wk). ✅ Done.** Package layout, `core/` models, units handling (bytes vs GiB
bugs are endemic in this domain — and duly caught two of our own, see §2.6), CI on CPU.
*Accept:* `infertune --help` runs ✅; core imports without torch installed ✅ (enforced by
`tests/test_core_purity.py`, which AST-checks every `core/` module against
`sys.stdlib_module_names`). 110 tests, ruff + mypy --strict clean.

Also delivered beyond the original M0 scope, because they turned out to be cheap and
load-bearing: `AttentionSpec.kv_bytes_per_token` with TP head-replication handling, and
`WorkloadProfile.working_set` Monte-Carlo sampling. Both are exposed as working CLI commands
(`infertune kv`, `infertune working-set`) so the arithmetic can be checked by hand.

**M1 — Analyzer + Estimator (1.5 wk). ✅ Done.** HF config parsing; safetensors-header weight
measurement; per-architecture cache models; overhead terms; GPU spec DB (15 GPUs) + NVML path;
the ledger.
*Accept:* weight bytes within ±1% across 20 checkpoints — **achieved 0.0000% (byte-exact) on
20/20**, spanning dense, MoE, MLA, Mamba-hybrid, interleaved-attention, AWQ, mxfp4, fp8
compressed-tensors, tied-embedding and encoder-decoder checkpoints ✅; ledger reconciles exactly
against usable VRAM, asserted for every plan ✅; runs with no GPU present ✅.

Delivered beyond the original scope, because the naive formula is most wrong exactly here:

* **MLA** (`MLASpec`) — DeepSeek-V3 caches one 576-element latent per token per layer. The GQA
  formula overestimates by **56.9x**, matching an independently published ~56x measurement.
  The latent is *not* TP-sharded, and fp8 MLA saves 1.76x rather than 2x (vLLM's `fp8_ds_mla`
  keeps RoPE in bf16 and adds per-token scales: 656 B/layer, not 576).
* **Attention/Mamba hybrids** (`RecurrentSpec`, `LayeredCacheSpec`) — Nemotron-H-8B holds KV on
  only 4 of 52 layers (**13x** correction) plus ~50 MiB of *fixed* state per sequence, for which
  `max_num_seqs` genuinely is a reservation.
* **Interleaved local/global attention** — gpt-oss-20b caps 12 of 24 layers at a 128-token
  window, so cache growth is sub-linear (**~2x** at 8K).

`infertune plan` is also wired up early, since the machinery existed and a milestone that
cannot be run cannot be reviewed.

**M2 — vLLM adapter + report (1 wk). ✅ Done.** Introspected schema, `LaunchSpec` generation,
validation diagnostics, report with ledger, envelope, `B*`, binding constraint, and risks;
`infertune profile` on measured hardware.
*Accept:* **4/4 configurations booted without OOM, worst KV prediction error 3.61%** against the
±5% bar, measured on an Azure A10 (see `docs/gpu-validation-a10.md`) ✅.

The criterion needed restating, and the reason is itself the milestone's main finding.
It originally read "predicted vs vLLM-logged `--kv-cache-memory`". **That flag does not exist in
vLLM 0.19.1**, the newest release that runs on an A10 at all — so scoring uses vLLM's
`Available KV cache memory` and `GPU KV cache size` lines instead, which are equivalent and
report both bytes and tokens.

That absence is precisely what §2.3 predicted: a hardcoded flag table would have emitted an
unparseable command line ten releases out of date. The adapter detects the missing lever and
falls back to inverting `--gpu-memory-utilization`, which reintroduces the inversion §2.2 hoped
to avoid — so the inversion is now explicit, clamped to a measured ceiling, and reported as a
diagnostic rather than hidden.

Three further corrections, all found by measurement rather than reasoning (details in
`docs/gpu-validation-a10.md`):

* **The utilisation ceiling is not 1.0.** A vGPU consumes 2.35 GiB of a 23.72 GiB card before
  any allocation, so the real ceiling is `free/total = 0.901`. An early inversion produced
  0.9613 and OOMed. `vram_usable_bytes` now means *torch-allocatable*, and non-torch overheads
  are no longer double-subtracted.
* **The CUDA graph pool prior was 7× too high** (728 MiB predicted vs 102 MiB measured).
* **Prefill activations must include the MLP intermediate**, which is 5.3× hidden on
  Qwen2.5-7B; omitting it cost ~530 MiB and alone breached the ±5% bar.

Deliberately **not** validated: fp8 KV and fp8 weights need sm_89, and the A10 is sm_86. Those
paths remain unverified rather than assumed working.

**M3 — Benchmark harness + measurement store (1.5 wk). 🟡 Implemented, acceptance pending
hardware.** Load generator with configurable distributions, TTFT/TPOT/throughput/p99, SQLite
store, roofline model, MFU/MBU calibration, inner-loop concurrency sweep, `infertune benchmark`.

*Accept:* one boot produces a full latency-throughput curve ✅ (verified against a mock engine);
post-calibration throughput prediction within **±20%** on held-out configs — **not yet
measured**, because it requires sustained GPU load rather than the boot-only evidence M2 used.

What *is* verified locally, without a GPU:

* **The calibration arithmetic round-trips.** Synthesising latencies from known coefficients
  and fitting them back recovers `mbu` to within 5% and `mfu` to within 10%. If the algebra
  were wrong, this would fail on a laptop rather than after renting a GPU.
* **The load generator measures what it claims.** A stdlib mock engine emits SSE chunks with
  *controlled* delays (80 ms to first token, 20 ms between), and the measured TTFT/TPOT match.
  Timing code that is only ever exercised against a real engine cannot be checked this way.
* **The sweep stops early for the right reasons** — SLA breach, throughput saturation, or
  excessive errors — since latency is monotone in concurrency.
* **A single outlier cannot move a fit.** Coefficients are aggregated by median, so one evicted
  spot instance or cold cache is ignored rather than enshrined.
* **Measurements are keyed by engine version.** vLLM changes memory accounting and scheduling
  defaults between releases, so pooling versions would quietly corrupt the calibration set.

Two modelling notes worth recording, both of which change reported numbers:

* **TPOT excludes the first token.** Including it lets prefill leak into a decode metric, making
  long prompts look like a decode regression.
* **Output throughput excludes prompt tokens.** Counting both inflates the figure by the
  input/output ratio, which is the usual reason published throughput numbers are incomparable.

**M4 — SGLang adapter + search (2 wk). ✅ Done (search validated synthetically).** Second
adapter, engine registry, analytic prune, Pareto selection, outer/inner loops, `infertune tune`.

*Accept:*

* **SGLang support adds zero changes outside `adapters/`** ✅ — verified with `git diff`:
  **0 files changed** in `core/`, `estimator/`, `bench/`, `store/`, `models/` or `hardware/`. The
  only change outside `adapters/` attributable to SGLang is a one-line mypy stub-ignore in
  `pyproject.toml` for an optional import — build configuration, not logic. (`cli.py` also
  changed, but for the `tune` command, which is the search feature rather than SGLang support.)
* **Within 10% of a grid search on ≤12 boots** ✅ — measured **0.00% gap using 5 boots against
  30**, i.e. the pruned search selected the *identical* configuration the exhaustive baseline
  did, 6× cheaper.

Validated against a synthetic objective rather than hardware, deliberately: search quality is a
property of the algorithm, not of any particular GPU, so it needs a known ground truth. The
synthetic objective is shaped to punish naive strategies — throughput saturates in concurrency,
fp8 buys capacity but costs latency, long contexts cost throughput — and a test asserts that
simply maximising `max_num_seqs` does *not* win.

### What the second adapter revealed

Adding SGLang was the real test of whether the core is framework-independent. Three genuine
differences the role indirection had to absorb:

1. **SGLang has an absolute token lever.** `--max-total-tokens` sizes the KV pool directly, so no
   fraction inversion is needed at all. vLLM 0.19.1 has no equivalent — which is why M2's test
   asserts `KV_BUDGET_TOKENS` is *unmapped* for vLLM. The role existed in the enum before there
   was an engine using it, and that turned out to be right.
2. **The memory fraction is scoped differently.** `--mem-fraction-static` covers weights + KV
   pool, with activations **outside** it; vLLM's `--gpu-memory-utilization` covers all three.
   Applying vLLM's formula to SGLang over-reserves; applying SGLang's to vLLM under-reserves and
   OOMs. This is §2.2's argument made executable, and a test asserts the two formulas differ.
3. **Polarity is inverted.** SGLang exposes `--disable-radix-cache` and `--disable-cuda-graph`
   where vLLM exposes `--enable-prefix-caching` and `--enforce-eager`. A role means the same
   thing to a user; the flag expressing it may be negated.

Also worth recording: **SGLang moved its entire argument surface** into
`sglang.srt.arg_groups.fields.*` modules, so `mem_fraction_static` is no longer in
`server_args.py` at all. A hardcoded flag table would be comprehensively broken. That is the
second engine independently confirming §2.3.

**Not done:** Bayesian optimisation over the surviving continuous knobs. Analytic pruning plus
Pareto selection already hits the acceptance target exactly, so BO would add machinery without
evidence that it is needed — better justified once M3's calibration is fitted on real
measurements and the prediction error is known.

**Deferred, deliberately:** multi-node, disaggregated prefill/decode, speculative decoding,
LoRA, MIG, non-NVIDIA. Each is a real dimension; none belongs in an MVP.

---

## 10. Developing this without a GPU

This is normally the blocker for a project like this. The layering makes most of it testable on
CPU, which is what §2.5 buys:

- **Golden fixtures.** Vendored `config.json` + safetensors index headers for ~25 models
  (dense, GQA, MLA, MoE, sliding-window, Mamba-hybrid, AWQ/GPTQ/fp8). Estimator tests are pure
  functions over these — fast, hermetic, no network.
- **Recorded hardware.** NVML/`nvidia-smi` captures and spec-DB entries as fixtures, so
  `GPUProfile` construction is tested both ways.
- **Recorded engine surfaces.** Captured `EngineArgs`/`ServerArgs` introspection dumps and
  startup logs per version, so adapter compilation and log parsing are tested without CUDA.
- **Property tests.** Monotonicity invariants that must hold regardless of coefficients:
  KV budget non-increasing in `max_model_len`; TP never *increasing* per-GPU weight bytes;
  fp8 KV never exceeding bf16 KV; ledger terms summing to the asserted total.
- **Public-number validation.** Cross-check predictions against published benchmark numbers for
  well-documented (model, GPU, engine) triples before we ever touch hardware.

GPU-only work is confined to M2/M3 acceptance and can be batched onto rented time in short,
scripted sessions. Everything up to that point is CPU work.

---

## 11. Risks

| risk | severity | mitigation |
|---|---|---|
| Engine flag/semantic drift | high | runtime introspection (§2.3); version-keyed capability matrix and measurements; never hardcode |
| Architecture zoo outgrows the KV model | high | `AttentionSpec` dispatch from day one; refuse-with-explanation on unknown architectures rather than guessing wrong |
| Overhead terms are empirical, not derivable | medium | measure and store; calibrate per (gpu, engine version); report intervals, and prefer honest conservatism where uncalibrated |
| Predictions look authoritative but aren't | medium | always emit intervals + provenance + binding constraint; never a bare number |
| GPU access limits validation | medium | §10; prioritize the ±5% memory check, which is cheap (one boot, no load) and catches the failure mode that matters most |
| Scope creep into a full deployment tool | medium | the tool recommends and benchmarks; it does not own production deployment |

---

## 12. Open questions

1. **Recommend-only, or deploy?** I'd argue recommend + benchmark, and emit configs for whatever
   owns deployment. Owning deployment triples the surface area and adds nothing to the core idea.
2. **Primary user:** someone with the GPU in hand (detection path) or someone doing capacity
   planning before buying (spec-DB path)? Both work, but it decides which gets the polished UX,
   and it changes the CLI's default mode.
3. **How much does the SLA-constrained mode matter vs. max-throughput?** Constrained mode needs
   the latency model to be good; max-throughput mode barely needs it. This is the biggest lever
   on M3's difficulty.
4. **Is MoE/MLA in the MVP?** Excluding them keeps M1 simple, but they're where naive
   calculators are most wrong, so they're where the tool is most differentiated.
5. Name: `infertune` is used throughout here as a placeholder — easy to change before M0, awkward
   after.

---

## Appendix: worked example, end to end

Llama-3.1-8B-Instruct on a single RTX 4090 (24 GB), vLLM, bf16 — the sketch's own example,
recomputed:

All figures below are produced by the code in this repository and asserted in
`tests/test_workload.py`, so they cannot drift from the implementation.

```
Ledger (per GPU)
  total VRAM (NVML)                                 23.99 GiB  (24564 MiB — binary, not 24 GB)
  usable after driver/display                       23.60 GiB
  weights (safetensors-measured, bf16)              14.96 GiB  (16.06 decimal GB)
  CUDA context + NCCL                                0.55 GiB
  CUDA graph pool (-O2, default capture sizes)       1.20 GiB
  prefill activations @ max_num_batched_tokens=8192  0.85 GiB
  logits/sampling @ max_num_seqs=24, vocab 128256    0.35 GiB
  fragmentation safety (3% of usable)                0.71 GiB
  ─────────────────────────────────────────────────────────
  KV budget                                          4.98 GiB  = 40,796 tokens @ 128 KiB/token

Workload: input median 1024 / p95 2048, output median 256 / p95 512 (log-normal)
  working set @ 24 concurrent   p50 30,092   p95 34,476   p99 36,386 tokens
  → fits at p99 in bf16; no mitigation required
  BINDING CONSTRAINT: kv_working_set — headroom runs out at ~28-32 concurrent
  → B* ≈ 51 for this GPU (82.6 TFLOPS bf16 w/ fp32 accum, 1008 GB/s, MFU 0.5 / MBU 0.8),
    so decode is still bandwidth-bound at 24. Added concurrency is nearly free on
    latency, but the cache runs out before the GPU does — KV is the wall, not compute.
  → to reach the compute knee:
      kv_cache_dtype=fp8   → 64 KiB/token → 81,592 tokens → carries ~48 concurrent (≈ B*)

  Diagnostics:
    naive per-request p95 would read 61,440 tokens (1.78x overstated) — see §2.6
    decimal-GB VRAM would give a 3.77 GiB budget / 30,883 tokens (-32%) — see §2.6
```

Note that the recommendation and the *reason* arrive together, that the binding constraint is
named, and that the diagnostics show what a naive calculation would have concluded instead. That
last part is the product: the tool is most valuable exactly where intuition is wrong.
