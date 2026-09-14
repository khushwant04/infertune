"""Command-line interface.

Unimplemented commands exit non-zero with the milestone that will deliver them, rather
than printing plausible-looking numbers. A profiler that guesses is worse than one that
declines, because the user cannot tell the difference until the deployment OOMs.
"""

from __future__ import annotations

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


@app.command()
def benchmark() -> None:
    """Benchmark a configuration and record the measurements."""
    _unimplemented("benchmark", "M3")


@app.command()
def tune() -> None:
    """Search for the best configuration under an SLA."""
    _unimplemented("tune", "M4")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
