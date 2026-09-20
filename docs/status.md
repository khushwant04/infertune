# Project status

Last updated: post-M4 implementation audit.

This file is the authoritative record of what has been built, what has been *measured*, and what
remains. The distinction matters: several things in this repo work locally but have not been held
against the acceptance criterion in [`plan.md §9`](plan.md), and those are marked as such rather
than counted as done.

## Milestones

Each milestone carries a falsifiable acceptance criterion fixed *before* the work started.

| | scope | acceptance criterion | result | state |
|---|---|---|---|---|
| **M0** | Skeleton, core domain models, units, CPU-only CI | core imports with no torch installed | enforced by an AST test over every `core/` module | ✅ met |
| **M1** | Model analyzer, per-architecture cache models, memory estimator, ledger, spec DB | weight bytes within **±1%** across 20 checkpoints | **0.0000%** — byte-exact on 20/20 | ✅ met |
| **M2** | vLLM adapter with runtime introspection, `profile` | predicted vs engine-reported KV within **±5%** | **3.61%** worst case, 4/4 models booted | ✅ met |
| **M3** | Roofline model, benchmark harness, SQLite store, calibration, `benchmark` | throughput prediction within **±20%** on held-out configs | **not measured** — needs sustained GPU load | 🟡 built, unverified |
| **M4** | SGLang adapter, engine registry, configuration search, `tune` dry-run | within **10%** of exhaustive grid using **≤12** boots | **0.00% synthetic gap** using **5 simulated** boots vs the grid's 30 | 🟡 algorithm met; runtime pending |

M4 carried a second, structural criterion: adding a second engine must require **zero** changes
outside `adapters/`. Measured on the merge diff — `core/`, `estimator/`, `bench/`, `store/`,
`models/` and `hardware/` were untouched.

## What has been measured on real hardware

One Azure A10 (`Standard_NV36ads_A10_v5`, vGPU 570.237, vLLM 0.19.1, torch 2.10.0+cu128). Full
environment and the driver-compatibility chain that constrains it: [`gpu-validation-a10.md`](gpu-validation-a10.md).

| model | KV error | weight error |
|---|---|---|
| Qwen3-0.6B | 0.51% | 0.87% |
| Qwen3-4B | 1.85% | 0.89% |
| Qwen2.5-7B-Instruct-AWQ | 1.84% | 0.23% |
| Qwen3-8B | 3.61% | 0.09% |

The A10's actual KV fit, recovered from engine logs, is linear in the utilisation fraction:

```
KV bytes = gpu_memory_utilization x 23.722 GiB - 1.424 GiB
```

with a hard ceiling: 0.88 boots, 0.90 OOMs in sampler warm-up. That ceiling is a vGPU artefact —
the hypervisor holds back 2.349 GiB, so only 0.901 of nominal VRAM is ever torch-allocatable. The
plan had assumed utilisation could approach 1.0.

The resource group has since been deleted; quota is released and billing stopped.
[`../scripts/provision_azure_gpu.sh`](../scripts/provision_azure_gpu.sh) encodes the whole recipe
and takes `--gpus 2` for a `Standard_NV72ads_A10_v5`.

## Correction factors the estimator applies

Naive sizing (`2 x layers x heads x head_dim x len`) is not wrong by a few percent; on several
released architectures it is wrong by more than an order of magnitude. Each factor below is
asserted by a test against the checkpoint's own published config.

| architecture | example | naive error | cause |
|---|---|---|---|
| MLA latent cache | DeepSeek-V3 | **56.9x** over | 576 elements/token/layer, not TP-sharded, fp8 |
| Attention/Mamba hybrid | Nemotron-H-8B | **13x** over | only 4 of 52 layers cache KV |
| Interleaved local/global | gpt-oss-20b | **~2x** at 8K | 12 of 24 layers capped at 128 tokens |
| Tied embeddings shipped twice | Qwen3-0.6B | **26.1%** over | `lm_head` duplicated on disk |
| `metadata.total_size` on fp8 | DeepSeek-V3 | **1.99x** over | header claims 1369 GB, real bytes 689 GB |
| Per-request p95 composed into an aggregate | any | **~1.8x** over | independent sequences concentrate |

Two of these were errors in this project's own plan, caught by measurement rather than review:
binary-vs-decimal VRAM units understated KV by 32%, and composing percentiles overstated the
working-set tail. Both are written up in [`plan.md §2.6`](plan.md).

## Known gaps

**M3's throughput criterion is unmeasured.** The roofline model, load generator, metrics
aggregation, SQLite store and calibration fit all exist and round-trip within 5% (latency) and 10%
(throughput) against synthetic curves. That is a self-consistency check, not a validation. The
±20% held-out claim requires a GPU under sustained load.

**fp8 KV is unvalidated and unvalidatable on the hardware used.** fp8 KV paths need compute
capability sm_89; the A10 is sm_86. The estimator's fp8 arithmetic is exercised by tests but has
never been held against a running engine. Treat fp8 recommendations as modelled, not measured.

**The SGLang adapter has never met a real SGLang install.** Its flag surface is derived from
SGLang's own `srt.arg_groups.fields.*` source rather than from `--help` scraping, which is the
right source of truth, but the round-trip has only been tested against a recorded schema fixture.

**Tensor parallelism has only ever run at tp=1.** The per-call latency term in the roofline model
— the part that predicts when TP stops helping — is entirely untested. NVads A10 v5 has no NVLink,
so a 2-GPU run over PCIe is the sharpest available test of exactly that term.

**`tune --execute` is not implemented.** The dry-run path (enumerate, prune, rank) is exercised,
and the reusable search core accepts an injected boot-and-measure callback. The CLI has no engine
launcher, readiness/termination supervisor, or production callback. Until those exist,
`--execute` exits with code 2 and states that no measurements were taken.

**Stored measurements do not yet feed CLI predictions.** The SQLite store and calibration fitter
work as library components and round-trip against synthetic data, but the CLI does not apply fitted
coefficients to `tune`. Benchmark records also need stronger engine/GPU/TP provenance before they
can safely drive candidate ranking.

## Remaining work

Software prerequisites that can be completed without a GPU:

1. Add engine launch, readiness, log capture, cleanup, and failure handling behind the adapter
   boundary; then connect that lifecycle to the search core and `tune --execute`.
2. Persist complete benchmark provenance and connect version- and TP-matched calibration
   coefficients to candidate scoring.

After those prerequisites, one GPU session on a 2xA10 host can:

3. Measure M3's ±20% throughput criterion against held-out configurations.
4. Run tp=2 over PCIe. Check early whether vGPU has disabled P2P — if it has, the measurement
   still stands but the latency term is being tested under a pessimistic interconnect.
5. Boot the SGLang adapter against a real install and diff its introspected schema against the
   fixture.
6. Run `tune --execute` end to end and confirm the executed winner matches the dry-run ranking.

Deliberately out of scope, and not planned: multi-node, disaggregated prefill/decode, speculative
decoding, LoRA, MIG, non-NVIDIA hardware. Bayesian optimisation over the surviving candidates was
specified in the plan and **not implemented** — analytic pruning plus Pareto filtering already
closed the synthetic test gap to 0.00%, so there is no evidence it would buy anything.

## Engineering gates

Every PR runs four CI jobs: `test` on Python 3.11, 3.12 and 3.13, plus `core-purity`.

```
ruff check src tests scripts
ruff format --check src tests scripts
mypy src tests scripts          # strict
pytest                          # default CPU-only suite
```

The 23 skips are network-dependent tests, opt-in via `pytest -m network --run-network`. Nothing in
the default suite needs a GPU, a model download, or a serving engine.
