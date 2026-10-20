"""Command-line interface.

Unimplemented commands exit non-zero with the milestone that will deliver them, rather
than printing plausible-looking numbers. A profiler that guesses is worse than one that
declines, because the user cannot tell the difference until the deployment OOMs.
"""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import TYPE_CHECKING

import typer
from rich.console import Console
from rich.table import Table

from . import __version__
from .core import (
    AttentionKind,
    AttentionSpec,
    Constant,
    DType,
    LogNormal,
    WorkloadProfile,
    fmt_bytes,
    fmt_tokens,
)
from .core.gpu import GPUProfile
from .core.model import ModelProfile
from .core.plan import ResourcePlan

if TYPE_CHECKING:
    from .adapters.base import FrameworkAdapter
    from .core.workload import SLA
    from .estimator.roofline import RooflineCoefficients
    from .search import PruneReport

app = typer.Typer(
    name="infertune",
    help="GPU-aware inference configuration profiler.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()

_NOT_YET = "Not implemented yet — scheduled for {milestone}. See docs/plan.md §9."


def _unimplemented(command: str, milestone: str) -> None:
    console.print(f"[yellow]{command}[/yellow]: " + _NOT_YET.format(milestone=milestone))
    raise typer.Exit(code=2)


def _render_plan(
    profile_: ModelProfile,
    gpu: GPUProfile,
    plan: ResourcePlan,
    max_num_seqs: int,
    tp: int = 1,
) -> None:
    """Render the memory ledger, envelope, binding constraint and mitigations.

    Shared by ``plan`` (spec-database path) and ``profile`` (measured-hardware path), so
    the two cannot drift apart in what they report.
    """
    from .estimator import with_kv_dtype

    console.print()
    console.print(f"[bold]{profile_.model_id}[/bold] on {gpu.count}x {gpu.name}  (tp={tp})")
    console.print(
        f"[dim]{type(profile_.cache).__name__}, "
        f"{profile_.cache.n_cache_layers}/{profile_.n_layers} layers cache state; "
        f"weights measured via {profile_.weight_bytes_source}[/dim]"
    )
    console.print()

    ledger = Table(box=None, pad_edge=False)
    ledger.add_column("term")
    ledger.add_column("bytes", justify="right")
    ledger.add_column("formula", style="dim")
    for entry in plan.ledger:
        style = "cyan" if entry.is_available else None
        label = entry.label if not entry.is_available else f"[{style}]{entry.label}[/{style}]"
        ledger.add_row(label, fmt_bytes(entry.bytes_), entry.formula)
    console.print(ledger)
    console.print()

    summary = Table(show_header=False, box=None, pad_edge=False)
    summary.add_row(
        "KV capacity",
        f"[bold]{fmt_tokens(plan.kv_budget_tokens)}[/bold] tokens "
        f"({fmt_bytes(plan.kv_bytes_per_token)}/token, {plan.dtypes.kv_cache})",
    )
    if plan.headroom_concurrency is not None:
        summary.add_row(
            "concurrency headroom", f"{plan.headroom_concurrency} (requested {max_num_seqs})"
        )
    if plan.critical_batch_size is not None:
        summary.add_row("critical batch size B*", f"{plan.critical_batch_size:.0f}")
    summary.add_row(
        "binding constraint",
        f"[bold yellow]{plan.binding_constraint.value}[/bold yellow] — "
        f"{plan.binding_constraint.explain()}",
    )
    if plan.cache_limited is not None:
        verdict = (
            "cache is the wall: a smaller KV dtype or more GPUs would help; a faster GPU would not"
            if plan.cache_limited
            else "the GPU saturates before the cache does: fp8 KV would not raise throughput"
        )
        summary.add_row("verdict", verdict)
    console.print(summary)

    if plan.dtypes.kv_cache is not DType.FP8_E4M3 and gpu.supports(DType.FP8_E4M3):
        fp8 = with_kv_dtype(plan, profile_.cache, DType.FP8_E4M3)
        console.print()
        console.print(
            f"[dim]with --kv-cache-dtype fp8: {fmt_tokens(fp8.kv_budget_tokens)} tokens "
            f"({fp8.kv_budget_tokens / max(1, plan.kv_budget_tokens):.2f}x)[/dim]"
        )

    if plan.warnings:
        console.print()
        for warning in plan.warnings:
            console.print(f"[yellow]![/yellow] {warning}")


@app.command()
def version() -> None:
    """Print the InferTune version."""
    console.print(f"infertune {__version__}")


@app.command("kv")
def kv_cache(
    layers: int = typer.Option(..., "--layers", "-l", help="Layers holding a KV cache."),
    kv_heads: int = typer.Option(
        ..., "--kv-heads", "-k", help="Key/value heads (GQA group count)."
    ),
    head_dim: int = typer.Option(..., "--head-dim", "-d", help="Dimension per head."),
    dtype: str = typer.Option("bf16", "--dtype", help="KV cache dtype, e.g. bf16, fp8_e4m3."),
    tp: int = typer.Option(1, "--tp", help="Tensor parallel size."),
    budget: str | None = typer.Option(
        None, "--budget", help="KV budget to convert into tokens, e.g. '3.77GiB'."
    ),
) -> None:
    """Compute KV cache cost per token.

    The load-bearing arithmetic of the whole system, exposed directly so it can be
    sanity-checked by hand.
    """
    from .core.units import parse_size

    try:
        kv_dtype = DType.parse(dtype)
        spec = AttentionSpec(
            kind=AttentionKind.GQA if kv_heads > 0 else AttentionKind.MHA,
            n_kv_heads=kv_heads,
            head_dim=head_dim,
            n_kv_layers=layers,
        )
        per_token = spec.kv_bytes_per_token(kv_dtype, tensor_parallel_size=tp)
    except ValueError as exc:
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=1) from None

    table = Table(show_header=False, box=None)
    table.add_row("KV bytes/token (per GPU)", f"[bold]{fmt_bytes(per_token)}[/bold]")
    table.add_row("KV heads per GPU", str(spec.kv_heads_per_gpu(tp)))

    replication = spec.kv_replication_factor(tp)
    if replication > 1.0:
        table.add_row(
            "[yellow]KV replication[/yellow]",
            f"[yellow]{replication:.2f}x — tp={tp} exceeds {kv_heads} KV heads, "
            f"so heads are replicated and TP no longer reduces per-GPU KV[/yellow]",
        )

    if budget is not None:
        try:
            budget_bytes = parse_size(budget)
        except ValueError as exc:
            console.print(f"[red]error:[/red] {exc}")
            raise typer.Exit(code=1) from None
        table.add_row("KV budget", fmt_bytes(budget_bytes))
        table.add_row("Token capacity", f"[bold]{fmt_tokens(budget_bytes // per_token)}[/bold]")

    console.print(table)


@app.command("working-set")
def working_set(
    concurrency: int = typer.Option(..., "--concurrency", "-c", help="In-flight requests."),
    input_median: float = typer.Option(1024, "--input-median", help="Median prompt tokens."),
    input_p95: float | None = typer.Option(
        None, "--input-p95", help="p95 prompt tokens. Omit for a fixed length."
    ),
    output_median: float = typer.Option(256, "--output-median", help="Median output tokens."),
    output_p95: float | None = typer.Option(
        None, "--output-p95", help="p95 output tokens. Omit for a fixed length."
    ),
    seed: int = typer.Option(0, "--seed", help="Sampling seed; fixed for reproducibility."),
) -> None:
    """Estimate aggregate in-flight KV tokens.

    Shows why per-request p95 lengths must not be composed into an aggregate p95:
    summing independent sequences concentrates the total, so the naive composition
    overstates the tail and drives needlessly conservative configurations.
    """
    try:
        workload = WorkloadProfile(
            input_tokens=(
                Constant(input_median)
                if input_p95 is None
                else LogNormal.from_median_p95(input_median, input_p95)
            ),
            output_tokens=(
                Constant(output_median)
                if output_p95 is None
                else LogNormal.from_median_p95(output_median, output_p95)
            ),
            target_concurrency=concurrency,
        )
        result = workload.working_set(seed=seed)
    except ValueError as exc:
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=1) from None

    table = Table(show_header=False, box=None)
    table.add_row("Concurrency", str(result.concurrency))
    table.add_row("Working set p50", fmt_tokens(result.p50_tokens) + " tokens")
    table.add_row("Working set p95", f"[bold]{fmt_tokens(result.p95_tokens)}[/bold] tokens")
    table.add_row("Working set p99", fmt_tokens(result.p99_tokens) + " tokens")
    table.add_row(
        "[dim]naive per-request p95[/dim]",
        f"[dim]{fmt_tokens(result.naive_p95_tokens)} tokens "
        f"({result.naive_overstatement:.2f}x overstated)[/dim]",
    )
    console.print(table)


@app.command()
def profile(
    model: str = typer.Option(..., "--model", "-m", help="Hugging Face repository id."),
    max_num_seqs: int = typer.Option(32, "--max-num-seqs", help="Concurrent request cap."),
    max_model_len: int = typer.Option(8192, "--max-model-len", help="Context length."),
    tp: int = typer.Option(1, "--tp", help="Tensor parallel size."),
    input_median: float = typer.Option(1024, "--input-median", help="Median prompt tokens."),
    input_p95: float = typer.Option(2048, "--input-p95", help="p95 prompt tokens."),
    output_median: float = typer.Option(256, "--output-median", help="Median output tokens."),
    output_p95: float = typer.Option(512, "--output-p95", help="p95 output tokens."),
    gpu_key: str | None = typer.Option(
        None, "--gpu", help="Spec-DB key, used instead of detecting hardware."
    ),
    show_command: bool = typer.Option(
        True, "--command/--no-command", help="Emit a runnable vLLM command line."
    ),
) -> None:
    """Profile a model on **detected** hardware and recommend a configuration.

    Measures the GPU actually present via NVML, then compiles the plan through the vLLM
    adapter against the *installed* vLLM's own parameter schema — so the emitted command line
    matches the engine on this machine rather than whatever the docs describe.
    """
    from .adapters import VLLMAdapter, VLLMSchemaError
    from .core.plan import Parallelism
    from .estimator import InfeasibleConfigurationError, estimate_plan
    from .estimator.memory import safe_utilization_ceiling
    from .hardware import nvml, specdb
    from .models import ModelMetadataError, analyze

    if gpu_key:
        try:
            gpu = specdb.load(gpu_key)
        except (specdb.UnknownGPUError, RuntimeError) as exc:
            console.print(f"[red]error:[/red] {exc}")
            raise typer.Exit(code=1) from None
    else:
        try:
            gpu = nvml.discover()
        except Exception as exc:
            console.print(f"[red]no GPU detected:[/red] {exc}")
            console.print(
                "[dim]Use 'infertune plan --gpu <key>' for capacity planning without "
                "hardware, or pass --gpu here to override detection.[/dim]"
            )
            raise typer.Exit(code=2) from None

    with console.status(f"reading metadata for {model}..."):
        try:
            profile_ = analyze(model)
        except ModelMetadataError as exc:
            console.print(f"[red]error:[/red] {exc}")
            raise typer.Exit(code=1) from None

    workload = WorkloadProfile(
        input_tokens=LogNormal.from_median_p95(input_median, input_p95),
        output_tokens=LogNormal.from_median_p95(output_median, output_p95),
        target_concurrency=max_num_seqs,
    )
    try:
        plan_ = estimate_plan(
            profile_,
            gpu,
            workload,
            max_num_seqs=max_num_seqs,
            max_model_len=max_model_len,
            parallelism=Parallelism(tensor=tp),
        )
    except InfeasibleConfigurationError as exc:
        console.print(f"[red]infeasible:[/red] {exc}")
        raise typer.Exit(code=1) from None
    except ValueError as exc:
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=1) from None

    _render_plan(profile_, gpu, plan_, max_num_seqs, tp)

    if not show_command:
        return
    try:
        adapter = VLLMAdapter()
    except VLLMSchemaError as exc:
        console.print()
        console.print(f"[yellow]![/yellow] cannot introspect vLLM here: {exc}")
        return

    ceiling = safe_utilization_ceiling(
        gpu.vram_bytes, max(0, gpu.vram_bytes - gpu.vram_usable_bytes)
    )
    spec = adapter.compile(
        plan_,
        model,
        max_num_seqs=max_num_seqs,
        max_model_len=max_model_len,
        total_vram_bytes=gpu.vram_bytes,
        max_utilization=ceiling,
    )
    caps = adapter.capabilities()
    console.print()
    console.print(
        f"[bold]vllm {caps.version}[/bold] [dim](schema {adapter.schema().source}; "
        f"absolute KV lever: {caps.absolute_kv_budget})[/dim]"
    )
    console.print(f"  [green]{spec.command_line('vllm')}[/green]")
    for diag in spec.diagnostics:
        colour = {"info": "dim", "warning": "yellow", "error": "red"}[diag.severity.value]
        console.print(f"  [{colour}]{diag}[/{colour}]")


@app.command("gpus")
def list_gpus() -> None:
    """List GPUs known to the specification database."""
    from .hardware import specdb

    table = Table(box=None)
    table.add_column("key")
    table.add_column("name")
    table.add_column("VRAM", justify="right")
    table.add_column("bandwidth", justify="right")
    table.add_column("B* (bf16)", justify="right")
    for key in specdb.available():
        gpu = specdb.load(key)
        try:
            critical = f"{gpu.critical_batch_size(DType.BF16):.0f}"
        except ValueError:
            critical = "-"
        table.add_row(
            key,
            gpu.name,
            fmt_bytes(gpu.vram_bytes),
            f"{gpu.mem_bandwidth_bytes_s / 1e9:.0f} GB/s",
            critical,
        )
    console.print(table)


@app.command("plan")
def plan_cmd(
    model: str = typer.Option(..., "--model", "-m", help="Hugging Face repository id."),
    gpu_key: str = typer.Option(..., "--gpu", "-g", help="GPU key, e.g. 'h100-sxm'."),
    count: int = typer.Option(1, "--gpus", help="Number of GPUs available."),
    tp: int = typer.Option(1, "--tp", help="Tensor parallel size."),
    max_num_seqs: int = typer.Option(32, "--max-num-seqs", help="Concurrent request cap."),
    max_model_len: int = typer.Option(8192, "--max-model-len", help="Context length."),
    input_median: float = typer.Option(1024, "--input-median", help="Median prompt tokens."),
    input_p95: float = typer.Option(2048, "--input-p95", help="p95 prompt tokens."),
    output_median: float = typer.Option(256, "--output-median", help="Median output tokens."),
    output_p95: float = typer.Option(512, "--output-p95", help="p95 output tokens."),
    kv_dtype: str | None = typer.Option(None, "--kv-dtype", help="KV cache dtype override."),
    eager: bool = typer.Option(False, "--eager", help="Assume --enforce-eager (no graphs)."),
) -> None:
    """Plan a configuration for hardware you do not have in hand.

    Needs no GPU: hardware facts come from the specification database, and model facts from
    checkpoint metadata read over HTTP range requests without downloading weights.
    """
    from .core.plan import Parallelism
    from .estimator import InfeasibleConfigurationError, estimate_plan
    from .hardware import specdb
    from .models import ModelMetadataError, analyze

    try:
        gpu = specdb.load(gpu_key, count=count)
    except (specdb.UnknownGPUError, RuntimeError) as exc:
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=1) from None

    with console.status(f"reading metadata for {model}..."):
        try:
            profile_ = analyze(model)
        except ModelMetadataError as exc:
            console.print(f"[red]error:[/red] {exc}")
            raise typer.Exit(code=1) from None

    workload = WorkloadProfile(
        input_tokens=LogNormal.from_median_p95(input_median, input_p95),
        output_tokens=LogNormal.from_median_p95(output_median, output_p95),
        target_concurrency=max_num_seqs,
    )
    try:
        resolved_kv = DType.parse(kv_dtype) if kv_dtype else None
        plan = estimate_plan(
            profile_,
            gpu,
            workload,
            max_num_seqs=max_num_seqs,
            max_model_len=max_model_len,
            parallelism=Parallelism(tensor=tp),
            kv_dtype=resolved_kv,
            enforce_eager=eager,
        )
    except InfeasibleConfigurationError as exc:
        console.print(f"[red]infeasible:[/red] {exc}")
        raise typer.Exit(code=1) from None
    except ValueError as exc:
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=1) from None

    _render_plan(profile_, gpu, plan, max_num_seqs, tp)


def _gpu_label(gpu_key: str | None) -> str:
    """The GPU's name as ``tune`` keys it, from the spec DB or detection; empty if unknown."""
    from .hardware import nvml, specdb

    try:
        return specdb.load(gpu_key).name if gpu_key else nvml.discover().name
    except Exception:
        return ""


def _engine_version(engine: str) -> str:
    """The installed engine's version, or empty when it cannot be introspected here."""
    from .adapters import UnknownEngineError, get_adapter

    try:
        return get_adapter(engine).capabilities().version
    except (UnknownEngineError, RuntimeError):
        return ""


def _kv_bytes_per_token(model: str, tp: int, kv_dtype: str | None) -> int | None:
    """Per-GPU KV bytes per token, which MBU calibration needs; ``None`` if not derivable.

    Without an explicit dtype this follows the estimator's default — the weight dtype, or bf16
    for sub-byte (quantized) weights, which engines never use as a KV dtype.
    """
    from .models import ModelMetadataError, analyze

    try:
        profile_ = analyze(model)
        dtype = DType.parse(kv_dtype) if kv_dtype else profile_.weight_dtype
        if dtype.is_sub_byte:
            dtype = DType.BF16
        return profile_.cache.marginal_bytes_per_token(dtype, tensor_parallel_size=tp)
    except (ModelMetadataError, ValueError):
        return None


@app.command()
def benchmark(
    model: str = typer.Option(..., "--model", "-m", help="Model name the server serves."),
    base_url: str = typer.Option("http://127.0.0.1:8000", "--url", help="Engine base URL."),
    concurrencies: str = typer.Option(
        "1,2,4,8,16,32", "--concurrencies", help="Comma-separated ladder."
    ),
    requests_per_point: int = typer.Option(32, "--requests", help="Requests per point."),
    input_median: float = typer.Option(1024, "--input-median", help="Median prompt tokens."),
    input_p95: float = typer.Option(2048, "--input-p95", help="p95 prompt tokens."),
    output_median: float = typer.Option(256, "--output-median", help="Median output tokens."),
    output_p95: float = typer.Option(512, "--output-p95", help="p95 output tokens."),
    ttft_p99_ms: float | None = typer.Option(None, "--ttft-p99-ms", help="TTFT SLA."),
    tpot_p99_ms: float | None = typer.Option(None, "--tpot-p99-ms", help="TPOT SLA."),
    store_path: str | None = typer.Option(None, "--store", help="SQLite file to record into."),
    engine: str = typer.Option("vllm", "--engine", help="Engine serving at --url."),
    tp: int = typer.Option(1, "--tp", help="Tensor parallel size the engine runs with."),
    kv_dtype: str | None = typer.Option(
        None, "--kv-dtype", help="KV cache dtype the engine runs with; default follows weights."
    ),
    gpu_key: str | None = typer.Option(
        None, "--gpu", help="Spec-DB key of the serving GPU; omit to detect."
    ),
    gpu_name: str = typer.Option("", "--gpu-name", help="GPU label; overrides --gpu."),
    engine_version: str = typer.Option(
        "", "--engine-version", help="Engine version; omit to introspect the installed engine."
    ),
) -> None:
    """Sweep concurrency against a **running** engine and record the curve.

    Deliberately does not start or stop anything. Engine boots cost minutes while changing
    client-side concurrency costs nothing, so one boot should yield the whole
    latency-throughput curve (see docs/plan.md §2.4).

    With ``--store``, the run is keyed by GPU, model, engine, engine version and TP. A run
    whose GPU or engine version cannot be established is stored as ``unknown`` and is never
    used for calibration.
    """
    from .bench import EndpointConfig, SweepConfig, sweep_concurrency, wait_for_endpoint
    from .core.workload import SLA
    from .runner import stored_points
    from .store import UNKNOWN, MeasurementStore, RunKey, StoredRun

    try:
        ladder = tuple(sorted({int(x) for x in concurrencies.split(",") if x.strip()}))
        if not ladder:
            raise ValueError("empty ladder")
        config = SweepConfig(concurrencies=ladder, requests_per_point=requests_per_point)
        workload = WorkloadProfile(
            input_tokens=LogNormal.from_median_p95(input_median, input_p95),
            output_tokens=LogNormal.from_median_p95(output_median, output_p95),
        )
    except ValueError as exc:
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=1) from None

    gpu_name = gpu_name or _gpu_label(gpu_key)
    engine_version = engine_version or _engine_version(engine)

    endpoint = EndpointConfig(base_url=base_url, model=model)
    console.print(f"waiting for {base_url} ...")
    if not wait_for_endpoint(endpoint, timeout_s=120, interval_s=2):
        console.print(f"[red]error:[/red] no engine responding at {base_url}")
        raise typer.Exit(code=1)

    sla = SLA(ttft_p99_ms=ttft_p99_ms, tpot_p99_ms=tpot_p99_ms)
    table = Table(box=None, pad_edge=False)
    for column, justify in (
        ("conc", "right"),
        ("out tok/s", "right"),
        ("req/s", "right"),
        ("ttft p50", "right"),
        ("ttft p99", "right"),
        ("tpot p50", "right"),
        ("tpot p99", "right"),
        ("err", "right"),
    ):
        table.add_column(column, justify=justify)  # type: ignore[arg-type]

    def show(point: object) -> None:
        p = point  # BenchmarkResult
        table.add_row(
            str(p.concurrency),  # type: ignore[attr-defined]
            f"{p.output_throughput_tokens_s:.0f}",  # type: ignore[attr-defined]
            f"{p.requests_per_s:.2f}",  # type: ignore[attr-defined]
            f"{p.ttft.p50_ms:.0f}ms",  # type: ignore[attr-defined]
            f"{p.ttft.p99_ms:.0f}ms",  # type: ignore[attr-defined]
            f"{p.tpot.p50_ms:.1f}ms",  # type: ignore[attr-defined]
            f"{p.tpot.p99_ms:.1f}ms",  # type: ignore[attr-defined]
            f"{p.error_rate:.0%}",  # type: ignore[attr-defined]
        )

    with console.status("sweeping..."):
        sweep = sweep_concurrency(
            endpoint,
            workload,
            config,
            sla=sla,
            model=model,
            engine=engine,
            engine_version=engine_version,
            gpu=gpu_name,
            on_point=show,
        )
    console.print(table)

    best = sweep.best_throughput
    if best is not None:
        console.print()
        console.print(
            f"best throughput   : [bold]{best.output_throughput_tokens_s:.0f}[/bold] "
            f"output tok/s at concurrency {best.concurrency}"
        )
    if sweep.saturation_concurrency is not None:
        console.print(
            f"saturation point  : concurrency {sweep.saturation_concurrency} "
            "[dim](empirical critical batch size)[/dim]"
        )
    if not sla.is_unconstrained:
        knee = sweep.knee(ttft_p99_ms=ttft_p99_ms, tpot_p99_ms=tpot_p99_ms)
        if knee is None:
            console.print("[yellow]no operating point satisfied the SLA[/yellow]")
        else:
            console.print(
                f"SLA knee          : concurrency {knee.concurrency} at "
                f"{knee.output_throughput_tokens_s:.0f} output tok/s"
            )
    for note in sweep.notes:
        console.print(f"[dim]! {note}[/dim]")

    if store_path:
        key = RunKey(
            gpu=gpu_name or UNKNOWN,
            model=model,
            engine=engine,
            engine_version=engine_version or UNKNOWN,
            tensor_parallel=tp,
        )
        store = MeasurementStore(store_path)
        run_id = store.record(
            StoredRun(
                key=key,
                gpu_count=tp,
                kv_dtype=kv_dtype,
                kv_bytes_per_token=_kv_bytes_per_token(model, tp, kv_dtype),
                points=stored_points(sweep),
                notes=" | ".join(sweep.notes),
                extra={"source": "benchmark", "infertune_version": __version__},
            )
        )
        if not key.is_complete:
            console.print(
                "[yellow]![/yellow] stored without full provenance "
                f"(gpu={key.gpu}, engine_version={key.engine_version}); this run will not be "
                "used for calibration. Pass --gpu/--gpu-name and --engine-version."
            )
        runs, points = store.count()
        console.print()
        console.print(
            f"[dim]recorded run {run_id} in {store_path} ({runs} runs, {points} points)[/dim]"
        )


def _ladder(text: str) -> tuple[int, ...]:
    ladder = tuple(sorted({int(x) for x in text.split(",") if x.strip()}))
    if not ladder:
        raise ValueError("empty concurrency ladder")
    return ladder


@app.command()
def tune(
    model: str = typer.Option(..., "--model", "-m", help="Hugging Face repository id."),
    gpu_key: str | None = typer.Option(None, "--gpu", help="Spec-DB key; omit to detect."),
    gpus: int = typer.Option(1, "--gpus", help="GPUs available."),
    engine: str = typer.Option("vllm", "--engine", help="Serving engine."),
    boot_budget: int = typer.Option(12, "--boots", help="Maximum engine boots to spend."),
    ttft_p99_ms: float | None = typer.Option(None, "--ttft-p99-ms", help="TTFT SLA."),
    tpot_p99_ms: float | None = typer.Option(None, "--tpot-p99-ms", help="TPOT SLA."),
    input_median: float = typer.Option(1024, "--input-median", help="Median prompt tokens."),
    input_p95: float = typer.Option(2048, "--input-p95", help="p95 prompt tokens."),
    output_median: float = typer.Option(256, "--output-median", help="Median output tokens."),
    output_p95: float = typer.Option(512, "--output-p95", help="p95 output tokens."),
    dry_run: bool = typer.Option(
        True,
        "--dry-run/--execute",
        help="Plan the search only, or boot the survivors and measure them.",
    ),
    store_path: str = typer.Option(
        "infertune.db", "--store", help="SQLite file: calibrates predictions, records runs."
    ),
    calibrate_: bool = typer.Option(
        True, "--calibrate/--no-calibrate", help="Fit roofline coefficients from the store."
    ),
    concurrencies: str = typer.Option(
        "1,2,4,8,16,32,64,128,256", "--concurrencies", help="Sweep ladder per boot."
    ),
    requests_per_point: int = typer.Option(32, "--requests", help="Requests per sweep point."),
    boot_timeout_s: float = typer.Option(
        900, "--boot-timeout", help="Seconds to wait for each engine to become ready."
    ),
    log_dir: str = typer.Option("infertune-runs", "--log-dir", help="Engine logs go here."),
    host: str = typer.Option("127.0.0.1", "--host", help="Interface engines bind to."),
    port: int = typer.Option(0, "--port", help="Engine port; 0 picks a free one per boot."),
    executable: str | None = typer.Option(
        None, "--executable", help="Engine launcher, e.g. '/opt/venv/bin/vllm'."
    ),
) -> None:
    """Search the configuration space under an SLA.

    Enumerates the restart-required space, prunes it analytically to the Pareto frontier, and
    ranks the survivors. With ``--execute`` it then boots each survivor on this machine, sweeps
    client-side concurrency on every boot, records the runs, and picks the measured winner.
    """
    from .adapters import UnknownEngineError, get_adapter
    from .core.workload import SLA
    from .hardware import nvml, specdb
    from .models import ModelMetadataError, analyze
    from .search import enumerate_candidates, prune

    executing = not dry_run
    no_boot = "no engine was started and no measurements were taken."

    try:
        ladder = _ladder(concurrencies)
        if boot_budget < 1:
            raise ValueError("--boots must be >= 1")
    except ValueError as exc:
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=1) from None

    try:
        gpu = specdb.load(gpu_key, count=gpus) if gpu_key else nvml.discover(count=gpus)
    except Exception as exc:
        console.print(f"[red]error:[/red] {exc}")
        if executing:
            console.print(f"[red]cannot execute:[/red] {no_boot}")
            raise typer.Exit(code=2) from None
        console.print("[dim]Pass --gpu <key> to select hardware without detection.[/dim]")
        raise typer.Exit(code=1) from None

    # Executing needs the installed engine's own schema; a dry run only reports it.
    adapter: FrameworkAdapter | None = None
    adapter_error = ""
    try:
        adapter = get_adapter(engine)
    except (UnknownEngineError, RuntimeError) as exc:
        if executing:
            console.print(f"[red]cannot execute:[/red] {engine} is not usable here: {exc}")
            console.print(f"[dim]{no_boot}[/dim]")
            raise typer.Exit(code=2) from None
        adapter_error = str(exc)

    with console.status(f"reading metadata for {model}..."):
        try:
            profile_ = analyze(model)
        except ModelMetadataError as exc:
            console.print(f"[red]error:[/red] {exc}")
            raise typer.Exit(code=1) from None

    workload = WorkloadProfile(
        input_tokens=LogNormal.from_median_p95(input_median, input_p95),
        output_tokens=LogNormal.from_median_p95(output_median, output_p95),
    )
    sla = SLA(ttft_p99_ms=ttft_p99_ms, tpot_p99_ms=tpot_p99_ms)
    candidates = enumerate_candidates(gpu)

    console.print()
    console.print(
        f"[bold]{profile_.model_id}[/bold] on {gpu.count}x {gpu.name}  "
        f"[dim](engine: {engine})[/dim]"
    )
    if adapter is not None:
        console.print(
            f"[dim]{engine} {adapter.capabilities().version} "
            f"(schema {adapter.schema().source})[/dim]"
        )
    else:
        console.print(f"[dim]! {engine} not introspectable here: {adapter_error}[/dim]")

    coefficients_by_tp = _calibration(
        store_path if calibrate_ else None,
        adapter,
        profile_,
        gpu,
        model,
        tuple(sorted({c.tensor_parallel for c in candidates})),
    )

    with console.status(f"pruning {len(candidates)} candidates analytically..."):
        report = prune(
            candidates,
            profile_,
            gpu,
            workload,
            coefficients_by_tp=coefficients_by_tp,
            keep=boot_budget,
        )

    console.print(
        f"enumerated {report.enumerated} -> {len(report.survivors)} to boot "
        f"[dim]({report.reduction_factor:.0f}x fewer)[/dim]"
    )
    for reason in report.reasons:
        console.print(f"  [dim]{reason}[/dim]")

    if not report.survivors:
        console.print("[red]nothing feasible on this hardware[/red]")
        raise typer.Exit(code=1)

    table = Table(box=None, pad_edge=False)
    for column in ("#", "configuration", "KV tokens", "pred tok/s", "pred TTFT", "headroom"):
        table.add_column(column, justify="right" if column != "configuration" else "left")
    for index, scored in enumerate(report.survivors, start=1):
        table.add_row(
            str(index),
            scored.candidate.label(),
            fmt_tokens(scored.kv_tokens),
            f"{scored.predicted_throughput:.0f}",
            f"{scored.predicted_ttft_s * 1000:.0f}ms",
            str(scored.headroom_concurrency or "-"),
        )
    console.print()
    console.print(table)

    if not sla.is_unconstrained:
        console.print()
        console.print(
            f"[dim]SLA: ttft p99 <= {ttft_p99_ms or 'any'} ms, "
            f"tpot p99 <= {tpot_p99_ms or 'any'} ms[/dim]"
        )

    if not executing:
        console.print()
        console.print(
            f"[yellow]dry run[/yellow]: would boot {len(report.survivors)} configurations "
            "and sweep concurrency on each. Re-run with --execute on the serving host to "
            "measure them."
        )
        return

    if adapter is None:  # pragma: no cover - guarded above; keeps the type checker honest
        raise typer.Exit(code=2)
    _execute(
        adapter=adapter,
        report=report,
        profile_=profile_,
        gpu=gpu,
        workload=workload,
        sla=sla,
        model=model,
        boot_budget=boot_budget,
        ladder=ladder,
        requests_per_point=requests_per_point,
        boot_timeout_s=boot_timeout_s,
        log_dir=Path(log_dir),
        host=host,
        port=port or None,
        executable=executable,
        store_path=store_path,
        coefficients_by_tp=coefficients_by_tp,
    )


def _calibration(
    store_path: str | None,
    adapter: FrameworkAdapter | None,
    profile_: ModelProfile,
    gpu: GPUProfile,
    model: str,
    tensor_parallel: tuple[int, ...],
) -> dict[int, RooflineCoefficients] | None:
    """Per-TP roofline coefficients fitted from comparable stored runs, when there are any.

    Only runs that match this GPU, model, engine *and* engine version are used, and only when
    the engine version is known: a calibration from another release is not evidence here.
    """
    from .estimator import calibrate_by_tp
    from .store import MeasurementStore, RunKey

    if store_path is None or not Path(store_path).exists():
        return None
    if adapter is None:
        console.print("[dim]calibration skipped: engine version unknown[/dim]")
        return None
    key = RunKey(
        gpu=gpu.name,
        model=model,
        engine=adapter.name,
        engine_version=adapter.capabilities().version,
    )
    runs = MeasurementStore(store_path).runs()
    results = calibrate_by_tp(runs, key, tensor_parallel, profile_, gpu)
    fitted = {tp: r.coefficients for tp, r in results.items() if r.is_calibrated}
    for tp, result in sorted(results.items()):
        if result.is_calibrated:
            console.print(f"[dim]calibrated tp={tp} from {store_path}: {result.summary()}[/dim]")
        elif result.rejected:
            console.print(
                f"[dim]tp={tp}: no usable calibration in {store_path} "
                f"({len(result.rejected)} samples rejected); using prior coefficients[/dim]"
            )
    return fitted or None


def _execute(
    *,
    adapter: FrameworkAdapter,
    report: PruneReport,
    profile_: ModelProfile,
    gpu: GPUProfile,
    workload: WorkloadProfile,
    sla: SLA,
    model: str,
    boot_budget: int,
    ladder: tuple[int, ...],
    requests_per_point: int,
    boot_timeout_s: float,
    log_dir: Path,
    host: str,
    port: int | None,
    executable: str | None,
    store_path: str,
    coefficients_by_tp: dict[int, RooflineCoefficients] | None,
) -> None:
    """Boot the pruned survivors, measure them, and report the measured winner."""
    from .bench import SweepConfig
    from .runner import EngineExecutor, ExecutionConfig
    from .search import search
    from .store import MeasurementStore

    executor = EngineExecutor(
        adapter,
        gpu,
        workload,
        ExecutionConfig(
            model_id=model,
            log_dir=log_dir,
            host=host,
            port=port,
            executable=executable,
            ready_timeout_s=boot_timeout_s,
            sweep=SweepConfig(concurrencies=ladder, requests_per_point=requests_per_point),
        ),
        sla=sla,
        store=MeasurementStore(store_path),
        on_event=lambda line: console.print(f"[dim]{line}[/dim]"),
    )

    console.print()
    console.print(f"[bold]executing[/bold]: up to {len(report.survivors)} boots")
    result = search(
        profile_,
        gpu,
        workload,
        executor,
        sla=sla,
        boot_budget=boot_budget,
        coefficients_by_tp=coefficients_by_tp,
        candidates=tuple(s.candidate for s in report.survivors),
    )

    table = Table(box=None, pad_edge=False)
    for column in (
        "#",
        "configuration",
        "status",
        "tok/s",
        "at conc",
        "pred tok/s",
        "KV pred/engine",
    ):
        left = column in ("configuration", "status")
        table.add_column(column, justify="left" if left else "right")
    for index, (evaluation, record) in enumerate(
        zip(result.evaluations, executor.records, strict=True), start=1
    ):
        measured = evaluation.throughput_within_sla(sla)
        concurrency = evaluation.best_concurrency(sla)
        kv = "-"
        if record.startup and record.startup.kv_cache_bytes and record.predicted_kv_bytes:
            kv = (
                f"{fmt_bytes(record.predicted_kv_bytes)} / "
                f"{fmt_bytes(record.startup.kv_cache_bytes)} ({record.kv_error:.1%})"
            )
        status = {"measured": "[green]measured[/green]", "failed": "[red]failed[/red]"}.get(
            record.status, "[yellow]skipped[/yellow]"
        )
        table.add_row(
            str(index),
            evaluation.scored.candidate.label(),
            status,
            f"{measured:.0f}" if measured else "-",
            str(concurrency or "-"),
            f"{evaluation.scored.predicted_throughput:.0f}",
            kv,
        )
    console.print()
    console.print(table)

    for record in executor.records:
        if record.status != "measured":
            console.print(f"[yellow]![/yellow] {record.candidate.label()}: {record.reason}")
            if record.log_tail:
                console.print(f"[dim]{record.log_tail}[/dim]")

    measured_runs = sum(1 for r in executor.records if r.run_id is not None)
    console.print()
    console.print(
        f"[dim]{result.boots_used} boots; {measured_runs} runs recorded in {store_path}; "
        f"logs in {log_dir}[/dim]"
    )

    if result.best is None:
        console.print(
            f"[red]no configuration produced a usable measurement:[/red] {result.notes[-1]}"
        )
        raise typer.Exit(code=1)

    winner = result.best.scored.candidate
    record = next(r for r in executor.records if r.candidate == winner)
    console.print()
    console.print(
        f"[bold green]winner[/bold green]: {winner.label()} — "
        f"[bold]{result.best_throughput:.0f}[/bold] output tok/s at concurrency "
        f"{result.best.best_concurrency(sla)}"
    )
    if result.prediction_error is not None:
        console.print(
            f"[dim]prediction error on the winner: {result.prediction_error:.0%} "
            "(large gaps mean the roofline needs calibrating; re-running tune will use "
            "these runs)[/dim]"
        )
    if record.spec is not None:
        console.print(f"  [green]{shlex.join(adapter.command(record.spec, executable))}[/green]")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
