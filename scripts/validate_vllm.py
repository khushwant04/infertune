#!/usr/bin/env python
"""Score InferTune's memory ledger against vLLM's own measurements.

Run this **on a GPU host with vLLM installed**. For each model it:

1. analyses the checkpoint (metadata only, no weight download),
2. builds a :class:`ResourcePlan` for the detected GPU,
3. compiles the plan into engine flags via the vLLM adapter,
4. boots vLLM with exactly those flags,
5. parses vLLM's startup log for its *actual* KV allocation,
6. reports the error.

This is the M2 acceptance measurement. It costs one engine boot per configuration and no
load generation, because vLLM reports its cache size at startup.

Usage::

    python scripts/validate_vllm.py --gpu a10
    python scripts/validate_vllm.py --gpu a10 --models Qwen/Qwen3-0.6B --max-num-seqs 32
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from infertune.adapters import VLLMAdapter
from infertune.core import LogNormal, WorkloadProfile, fmt_bytes, fmt_tokens
from infertune.core.gpu import GPUProfile
from infertune.estimator import InfeasibleConfigurationError, estimate_plan
from infertune.hardware import nvml, specdb
from infertune.models import ModelMetadataError, analyze

DEFAULT_MODELS = [
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen2.5-7B-Instruct-AWQ",
    "Qwen/Qwen3-8B",
]


@dataclass
class Result:
    model: str
    ok: bool
    max_num_seqs: int
    max_model_len: int
    gpu_memory_utilization: float | None = None
    predicted_kv_bytes: int | None = None
    actual_kv_bytes: int | None = None
    predicted_kv_tokens: int | None = None
    actual_kv_tokens: int | None = None
    predicted_weight_bytes: int | None = None
    actual_weight_bytes: int | None = None
    predicted_bytes_per_token: int | None = None
    actual_bytes_per_token: float | None = None
    error: str = ""

    def _pct(self, pred: int | float | None, act: int | float | None) -> float | None:
        if pred is None or act in (None, 0):
            return None
        return abs(pred - act) / act * 100.0  # type: ignore[operator]

    @property
    def kv_bytes_error_pct(self) -> float | None:
        return self._pct(self.predicted_kv_bytes, self.actual_kv_bytes)

    @property
    def weight_error_pct(self) -> float | None:
        return self._pct(self.predicted_weight_bytes, self.actual_weight_bytes)

    @property
    def bytes_per_token_error_pct(self) -> float | None:
        return self._pct(self.predicted_bytes_per_token, self.actual_bytes_per_token)


def resolve_gpu(key: str | None) -> GPUProfile:
    """Prefer a live measurement; fall back to the spec database."""
    if key is None:
        try:
            gpu = nvml.discover()
            print(
                f"GPU (NVML)   : {gpu.name}  {fmt_bytes(gpu.vram_bytes)} total, "
                f"{fmt_bytes(gpu.vram_usable_bytes)} usable"
            )
            return gpu
        except Exception as exc:
            print(f"NVML unavailable ({exc}); falling back to the spec database")
            key = "a10"
    gpu = specdb.load(key)
    print(
        f"GPU (specdb) : {gpu.name}  {fmt_bytes(gpu.vram_bytes)} total, "
        f"{fmt_bytes(gpu.vram_usable_bytes)} usable"
    )
    return gpu


def boot_vllm(model: str, kwargs: dict[str, object], timeout: int) -> str:
    """Boot vLLM once via the offline API and return its combined log output."""
    script = (
        "import sys\n"
        "from vllm import LLM, SamplingParams\n"
        f"llm = LLM(model={model!r}, **{kwargs!r})\n"
        "o = llm.generate(['hello'], SamplingParams(max_tokens=4, temperature=0))\n"
        "print('SMOKE_OK:', o[0].outputs[0].text.strip()[:40])\n"
    )
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write(script)
        path = fh.name
    try:
        proc = subprocess.run(
            [sys.executable, path],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return proc.stdout + proc.stderr
    except subprocess.TimeoutExpired:
        return "TIMEOUT"
    finally:
        Path(path).unlink(missing_ok=True)


def validate_one(
    model: str,
    gpu: GPUProfile,
    adapter: VLLMAdapter,
    workload: WorkloadProfile,
    *,
    max_num_seqs: int,
    max_model_len: int,
    timeout: int,
) -> Result:
    res = Result(model=model, ok=False, max_num_seqs=max_num_seqs, max_model_len=max_model_len)
    try:
        profile = analyze(model)
    except ModelMetadataError as exc:
        res.error = f"analyze failed: {exc}"
        return res

    try:
        plan = estimate_plan(
            profile,
            gpu,
            workload,
            max_num_seqs=max_num_seqs,
            max_model_len=max_model_len,
            samples=2000,
        )
    except (InfeasibleConfigurationError, ValueError) as exc:
        res.error = f"infeasible: {exc}"
        return res

    res.predicted_kv_bytes = plan.kv_budget_bytes
    res.predicted_kv_tokens = plan.kv_budget_tokens
    res.predicted_weight_bytes = plan.weight_bytes_per_gpu
    res.predicted_bytes_per_token = plan.kv_bytes_per_token

    spec = adapter.compile(
        plan,
        model,
        max_num_seqs=max_num_seqs,
        max_model_len=max_model_len,
        total_vram_bytes=gpu.vram_bytes,
    )
    if not spec.is_runnable:
        res.error = "; ".join(d.message for d in spec.errors)
        return res

    kwargs = spec.python_kwargs()
    gmu = kwargs.get("gpu_memory_utilization")
    res.gpu_memory_utilization = float(gmu) if gmu is not None else None
    typed: dict[str, object] = {}
    for k, v in kwargs.items():
        if isinstance(v, bool):
            typed[k] = v
        elif isinstance(v, str) and v.replace(".", "", 1).isdigit():
            typed[k] = float(v) if "." in v else int(v)
        else:
            typed[k] = v

    print(f"  booting with {typed}")
    log = boot_vllm(model, typed, timeout)
    facts = adapter.parse_startup_log(log)

    res.actual_kv_bytes = facts.kv_cache_bytes
    res.actual_kv_tokens = facts.kv_cache_tokens
    res.actual_weight_bytes = facts.weight_bytes
    res.actual_bytes_per_token = facts.bytes_per_token()
    res.ok = "SMOKE_OK:" in log and facts.has_kv_measurement
    if not res.ok:
        tail = [ln for ln in log.splitlines() if "Error" in ln or "error" in ln][-2:]
        res.error = (
            " | ".join(tail) if tail else ("timeout" if log == "TIMEOUT" else "no KV in log")
        )
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    ap.add_argument("--gpu", default=None, help="specdb key; omit to use NVML")
    ap.add_argument("--max-num-seqs", type=int, default=32)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--json", default=None, help="write raw results here")
    args = ap.parse_args()

    gpu = resolve_gpu(args.gpu)
    adapter = VLLMAdapter()
    caps = adapter.capabilities()
    print(f"engine       : vllm {caps.version}  (absolute KV lever: {caps.absolute_kv_budget})")
    workload = WorkloadProfile(
        input_tokens=LogNormal.from_median_p95(1024, 2048),
        output_tokens=LogNormal.from_median_p95(256, 512),
    )

    results: list[Result] = []
    for model in args.models:
        print(f"\n=== {model} ===")
        r = validate_one(
            model,
            gpu,
            adapter,
            workload,
            max_num_seqs=args.max_num_seqs,
            max_model_len=args.max_model_len,
            timeout=args.timeout,
        )
        results.append(r)
        if r.ok:
            print(
                f"  predicted KV : {fmt_bytes(r.predicted_kv_bytes or 0)} "
                f"({fmt_tokens(r.predicted_kv_tokens or 0)} tokens)"
            )
            print(
                f"  actual    KV : {fmt_bytes(r.actual_kv_bytes or 0)} "
                f"({fmt_tokens(r.actual_kv_tokens or 0)} tokens)"
            )
            print(f"  KV error     : {r.kv_bytes_error_pct:.2f}%")
            print(f"  weight error : {r.weight_error_pct:.2f}%")
            print(f"  bytes/token  : {r.bytes_per_token_error_pct:.3f}% error")
        else:
            print(f"  FAILED: {r.error}")

    print("\n" + "=" * 92)
    print(f"{'model':34s} {'gmu':>7s} {'KV err':>8s} {'wt err':>8s} {'B/tok err':>10s}  ok")
    print("-" * 92)
    for r in results:

        def f(v: float | None) -> str:
            return f"{v:.2f}%" if v is not None else "-"

        print(
            f"{r.model:34s} {r.gpu_memory_utilization or 0:7.4f} "
            f"{f(r.kv_bytes_error_pct):>8s} {f(r.weight_error_pct):>8s} "
            f"{f(r.bytes_per_token_error_pct):>10s}  {'yes' if r.ok else 'NO'}"
        )

    scored = [r for r in results if r.ok and r.kv_bytes_error_pct is not None]
    if scored:
        worst = max(r.kv_bytes_error_pct or 0 for r in scored)
        print("-" * 92)
        print(
            f"{len(scored)}/{len(results)} booted; worst KV error {worst:.2f}% "
            f"(M2 criterion: +/-5%)  -> {'PASS' if worst <= 5 else 'FAIL'}"
        )

    if args.json:
        Path(args.json).write_text(json.dumps([asdict(r) for r in results], indent=2))
        print(f"raw results -> {args.json}")
    return 0 if scored and max(r.kv_bytes_error_pct or 0 for r in scored) <= 5 else 1


if __name__ == "__main__":
    raise SystemExit(main())
