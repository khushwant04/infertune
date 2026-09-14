"""CLI smoke tests.

The M0 acceptance criterion is that ``infertune --help`` works. Beyond that, the
important behaviour is that unimplemented commands *fail* rather than emit numbers.
"""

from __future__ import annotations

import inspect

from typer.testing import CliRunner

from infertune.cli import app

runner = CliRunner()


def test_help_works() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "inference configuration profiler" in result.stdout


def test_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "infertune" in result.stdout


def test_kv_command_reports_llama_31_8b() -> None:
    result = runner.invoke(app, ["kv", "--layers", "32", "--kv-heads", "8", "--head-dim", "128"])
    assert result.exit_code == 0
    assert "128.00 KiB" in result.stdout


def test_kv_command_converts_a_budget_to_tokens() -> None:
    result = runner.invoke(
        app,
        ["kv", "-l", "32", "-k", "8", "-d", "128", "--budget", "3.77GiB"],
    )
    assert result.exit_code == 0
    assert "30,883" in result.stdout  # 3.77 GiB / 128 KiB, floor-divided


def test_kv_command_warns_about_head_replication() -> None:
    result = runner.invoke(app, ["kv", "-l", "32", "-k", "8", "-d", "128", "--tp", "16"])
    assert result.exit_code == 0
    assert "replication" in result.stdout.lower()


def test_kv_command_rejects_a_bad_dtype() -> None:
    result = runner.invoke(app, ["kv", "-l", "32", "-k", "8", "-d", "128", "--dtype", "float9"])
    assert result.exit_code == 1
    assert "unknown dtype" in result.stdout


def test_kv_command_rejects_a_bad_budget() -> None:
    result = runner.invoke(app, ["kv", "-l", "32", "-k", "8", "-d", "128", "--budget", "lots"])
    assert result.exit_code == 1


def test_working_set_command_reports_the_naive_overstatement() -> None:
    result = runner.invoke(
        app,
        [
            "working-set",
            "-c",
            "24",
            "--input-median",
            "1024",
            "--input-p95",
            "2048",
            "--output-median",
            "256",
            "--output-p95",
            "512",
        ],
    )
    assert result.exit_code == 0
    assert "overstated" in result.stdout


def test_unimplemented_commands_exit_nonzero_with_a_milestone() -> None:
    """Never print plausible numbers for unbuilt features."""
    for command, milestone in (("tune", "M4"),):
        result = runner.invoke(app, [command])
        assert result.exit_code == 2, command
        assert milestone in result.stdout, command
        assert "docs/plan.md" in result.stdout, command


def test_profile_declares_expected_options() -> None:
    from infertune.cli import profile

    declared = _declared_option_names(profile)
    for option in ("--model", "--max-num-seqs", "--max-model-len", "--tp", "--gpu"):
        assert option in declared, f"{option} not declared; found {sorted(declared)}"


def test_profile_without_a_gpu_fails_clearly_and_points_at_plan() -> None:
    """No GPU here, so `profile` must refuse and name the hardware-free alternative."""
    result = runner.invoke(app, ["profile", "--model", "Qwen/Qwen3-0.6B"])
    assert result.exit_code == 2
    assert "no GPU detected" in result.stdout
    assert "infertune plan" in result.stdout


def test_benchmark_declares_expected_options() -> None:
    from infertune.cli import benchmark

    declared = _declared_option_names(benchmark)
    for option in ("--model", "--url", "--concurrencies", "--store", "--ttft-p99-ms"):
        assert option in declared, f"{option} missing; found {sorted(declared)}"


def test_benchmark_fails_clearly_without_an_engine() -> None:
    result = runner.invoke(app, ["benchmark", "--model", "m", "--url", "http://127.0.0.1:1"])
    assert result.exit_code == 1
    assert "no engine responding" in result.stdout


def test_gpus_command_lists_the_spec_database() -> None:
    result = runner.invoke(app, ["gpus"])
    assert result.exit_code == 0
    assert "h100-sxm" in result.stdout
    assert "rtx-4090" in result.stdout


def test_plan_command_requires_a_known_gpu() -> None:
    result = runner.invoke(app, ["plan", "--model", "x/y", "--gpu", "gtx-750-ti"])
    assert result.exit_code == 1
    assert "unknown GPU" in result.stdout


def _declared_option_names(callback: object) -> set[str]:
    """CLI flag names declared by a command callback.

    Introspects the signature rather than the rendered ``--help`` text, because rich wraps
    and truncates output at the terminal width — which differs between a developer's terminal
    and CI, making any assertion on rendered help environment-dependent.
    """
    names: set[str] = set()
    for parameter in inspect.signature(callback).parameters.values():  # type: ignore[arg-type]
        default = parameter.default
        for attribute in ("param_decls", "_param_decls"):
            decls = getattr(default, attribute, None)
            if decls:
                names.update(str(d) for d in decls if str(d).startswith("-"))
    return names


def test_plan_command_declares_expected_options() -> None:
    """The plan command needs no GPU, so it must be reachable in any environment."""
    from infertune.cli import plan_cmd

    declared = _declared_option_names(plan_cmd)
    for option in ("--model", "--gpu", "--tp", "--max-num-seqs", "--max-model-len", "--kv-dtype"):
        assert option in declared, f"{option} not declared; found {sorted(declared)}"


def test_plan_help_exits_cleanly() -> None:
    """Rendering is width-dependent, so assert only on the exit status here."""
    assert runner.invoke(app, ["plan", "--help"]).exit_code == 0


def test_expected_commands_are_registered() -> None:
    registered = {
        info.name or (info.callback.__name__ if info.callback else "")
        for info in app.registered_commands
    }
    for command in ("kv", "working-set", "plan", "gpus", "profile", "benchmark", "tune"):
        assert command in registered, f"{command} missing; found {sorted(registered)}"
