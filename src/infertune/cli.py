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
def profile() -> None:
    """Profile a model on detected hardware and recommend a configuration."""
    _unimplemented("profile", "M2")


@app.command("plan")
def plan_cmd() -> None:
    """Plan a configuration for hardware you do not have yet."""
    _unimplemented("plan", "M2")


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
