# Changelog

All notable changes to InferTune are recorded here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow [Semantic Versioning](https://semver.org/). Before 1.0, minor versions may contain breaking changes.

## [Unreleased]

## [0.1.0] - 2026-10-20

The first release. It is a pre-release: the memory model is validated against real vLLM on an A10, but the throughput model, SGLang, tensor parallelism above 1, and `tune --execute` on a real GPU have not yet been measured. See [`docs/status.md`](docs/status.md).

### Added

**Planning without a GPU**
- `infertune plan`: a full memory ledger for a model, GPU, engine and workload, ending in the **binding constraint** (VRAM, the KV cache, or compute) and a verdict on what would help.
- Model facts come from checkpoint metadata read over HTTP range requests. No weights are downloaded. Weight bytes are byte-exact on 20 of 20 checkpoints tested.
- Per-architecture cache models: GQA/MHA, MLA (DeepSeek-V3), attention/Mamba hybrids (Nemotron-H), interleaved local/global attention (gpt-oss), with tied-embedding and fp8 `total_size` corrections.
- A workload model in which the aggregate p95 working set is sampled, never composed from per-request p95 lengths.
- A GPU specification database of 16 NVIDIA GPUs (`infertune gpus`).
- `infertune kv` and `infertune working-set` expose the core arithmetic directly.

**Engines**
- A vLLM adapter that introspects the installed engine's own argument schema and compiles plans to its flags. It falls back from the absolute KV lever to an inverted `--gpu-memory-utilization` when the installed version lacks it.
- An SGLang adapter behind the same interface, which needed no changes outside `adapters/`.
- Startup-log parsers that recover the KV pool, weight memory and CUDA-graph memory each engine actually allocated.
- `infertune profile`: plans on detected hardware (NVML) and prints a runnable command for the installed vLLM.

**Measurement and search**
- A roofline performance model with MFU/MBU coefficients and a critical batch size B\*.
- `infertune benchmark`: a streaming load generator and concurrency sweep against a running OpenAI-compatible server. It stops early at saturation or an SLA breach.
- A SQLite measurement store keyed by GPU, model, engine, engine version and tensor parallelism, plus a calibration fitter that recovers MBU from decode and MFU from prefill.
- `infertune tune`: enumerates the restart-required configuration space, prunes it analytically to a Pareto frontier, and ranks the survivors. On a synthetic objective it matched an exhaustive 30-boot grid using 5 boots.
- `infertune tune --execute`: boots each surviving candidate on the local GPU, sweeps concurrency on every boot, records the runs, and reports the measured winner with its launch command. The engine lifecycle:
  - starts each engine in its own process group on a fresh port and captures its log to a file;
  - notices an engine that exits during startup (for example an OOM) right away, instead of waiting out the readiness timeout;
  - stops the whole process group, so tensor-parallel workers cannot hold GPU memory into the next boot;
  - records a failed boot with the end of its log, and the search moves on to the next candidate.
- Calibration feeds candidate scoring: `tune` fits roofline coefficients separately for each TP degree from stored runs that match the GPU, model, engine and engine version.
- Runs stored with an `unknown` GPU or engine version are never used for calibration. `benchmark` gained `--engine`, `--tp`, `--gpu` and `--kv-dtype` and resolves the engine version from the installed engine, so its runs carry full provenance.

**Project**
- Apache-2.0 license.
- CPU-only CI on Python 3.11–3.13 with ruff, strict mypy and pytest, plus an AST check that `infertune.core` imports nothing outside the standard library.
- A release workflow that builds the sdist and wheel and creates the GitHub release from this changelog.
- Docs: the design plan, the project status, the A10 validation record, and a runbook for the next GPU session.

### Known limitations

- The ±20% throughput criterion (M3) is unmeasured. Treat predicted tokens/s as ranking signals until the store holds calibrated runs for your GPU.
- `tune --execute` is tested end to end against a fake engine process, not yet on a GPU.
- The SGLang adapter has only been tested against a recorded schema fixture.
- fp8 KV paths need sm_89+ and are modelled, not measured.
- Tensor parallelism has only run at tp=1.
- Pruning is slow when a model has very large KV headroom (a small model on an 80 GB GPU, or a very short workload). It can take a minute or more, because the headroom search samples the working set by Monte Carlo.
- Not on PyPI yet. Install from the repository, or from the wheel attached to the GitHub release.

[Unreleased]: https://github.com/khushwant04/infertune/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/khushwant04/infertune/releases/tag/v0.1.0
