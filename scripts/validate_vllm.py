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
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from infertune.adapters import VLLMAdapter
from infertune.core import LogNormal, WorkloadProfile, fmt_bytes, fmt_tokens
from infertune.core.gpu import GPUProfile, Interconnect
from infertune.core.plan import Parallelism
from infertune.estimator import InfeasibleConfigurationError, estimate_plan
from infertune.estimator.memory import (
    estimate_overheads,
    predict_kv_for_utilization,
)
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
    tensor_parallel: int = 1
    total_bytes: int | None = None
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
        if pred is None or act is None or act == 0:
            return None
        return abs(pred - act) / act * 100.0

    @property
    def kv_bytes_error_pct(self) -> float | None:
        return self._pct(self.predicted_kv_bytes, self.actual_kv_bytes)

    @property
    def kv_abs_error_bytes(self) -> int | None:
        """Absolute miss, which is the scale-free measure of the overhead model.

        Relative KV error is misleading on weights-dominated configurations: the same
        absolute overhead error is a small fraction of a large cache and a large fraction
        of a small one.
        """
        if self.predicted_kv_bytes is None or self.actual_kv_bytes is None:
            return None
        return abs(self.predicted_kv_bytes - self.actual_kv_bytes)

    @property
    def kv_share_of_budget(self) -> float | None:
        """Share of the engine's memory budget that ended up as KV cache."""
        if self.actual_kv_bytes is None or not self.gpu_memory_utilization:
            return None
        if self.total_bytes is None:
            return None
        budget = self.gpu_memory_utilization * self.total_bytes
        return self.actual_kv_bytes / budget if budget else None

    @property
    def weight_error_pct(self) -> float | None:
        return self._pct(self.predicted_weight_bytes, self.actual_weight_bytes)

    @property
    def bytes_per_token_error_pct(self) -> float | None:
        return self._pct(self.predicted_bytes_per_token, self.actual_bytes_per_token)


def _pct(value: float | None) -> str:
    return f"{value:.2f}%" if value is not None else "-"


def resolve_gpu(key: str | None, *, count: int = 1) -> GPUProfile:
    """Prefer a live measurement; fall back to the spec database.

    ``count`` must match the tensor-parallel size, or the estimator refuses the plan.
    The spec database describes one device, so a multi-GPU profile assembled from it also
    needs an interconnect; PCIe gen4 is assumed, being the pessimistic realistic case.
    """
    if key is None:
        try:
            gpu = nvml.discover(count=count)
            print(
                f"GPU (NVML)   : {gpu.name} x{gpu.count}  {fmt_bytes(gpu.vram_bytes)} total, "
                f"{fmt_bytes(gpu.vram_usable_bytes)} usable, link {gpu.interconnect}"
            )
            for note in gpu.notes:
                print(f"  note: {note}")
            return gpu
        except Exception as exc:
            print(f"NVML unavailable ({exc}); falling back to the spec database")
            key = "a10"
    gpu = specdb.load(key)
    if count > 1:
        gpu = replace(gpu, count=count, interconnect=Interconnect.PCIE_GEN4_X16)
    print(
        f"GPU (specdb) : {gpu.name} x{gpu.count}  {fmt_bytes(gpu.vram_bytes)} total, "
        f"{fmt_bytes(gpu.vram_usable_bytes)} usable, link {gpu.interconnect}"
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
    tensor_parallel: int = 1,
) -> Result:
    res = Result(
        model=model,
        ok=False,
        max_num_seqs=max_num_seqs,
        max_model_len=max_model_len,
        tensor_parallel=tensor_parallel,
    )
    parallelism = Parallelism(tensor=tensor_parallel)
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
            parallelism=parallelism,
            samples=2000,
        )
    except (InfeasibleConfigurationError, ValueError) as exc:
        res.error = f"infeasible: {exc}"
        return res

    res.predicted_kv_bytes = plan.kv_budget_bytes
    res.predicted_kv_tokens = plan.kv_budget_tokens
    res.predicted_weight_bytes = plan.weight_bytes_per_gpu
    res.predicted_bytes_per_token = plan.kv_bytes_per_token
    res.total_bytes = gpu.vram_bytes

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

    if res.gpu_memory_utilization is not None:
        oh = estimate_overheads(
            profile,
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=min(max_model_len, 8192),
            parallelism=parallelism,
        )
        res.predicted_kv_bytes = predict_kv_for_utilization(
            plan,
            oh,
            res.gpu_memory_utilization,
            gpu.vram_usable_bytes + max(0, gpu.vram_bytes - gpu.vram_usable_bytes),
        )
        res.predicted_kv_tokens = res.predicted_kv_bytes // plan.kv_bytes_per_token
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
    ap.add_argument(
        "--tp",
        type=int,
        default=1,
        help="tensor parallel size; >1 shards weights and tests the TP overhead terms",
    )
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--json", default=None, help="write raw results here")
    args = ap.parse_args()

    gpu = resolve_gpu(args.gpu, count=args.tp)
    adapter = VLLMAdapter()
    caps = adapter.capabilities()
    print(f"engine       : vllm {caps.version}  (absolute KV lever: {caps.absolute_kv_budget})")
    print(f"parallelism  : tp={args.tp}")
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
            tensor_parallel=args.tp,
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
    print(f"{'model':30s} {'KVshare':>8s} {'KV err':>8s} {'KV abs':>10s} {'wt err':>8s}  ok")
    print("-" * 92)
    for r in results:
        share = r.kv_share_of_budget
        abs_mib = (r.kv_abs_error_bytes or 0) / (1024 * 1024)
        print(
            f"{r.model:30s} "
            f"{(f'{share:.0%}' if share else '-'):>8s} "
            f"{_pct(r.kv_bytes_error_pct):>8s} "
            f"{abs_mib:>8.0f}M "
            f"{_pct(r.weight_error_pct):>8s}  "
            f"{'yes' if r.ok else 'NO'}"
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
