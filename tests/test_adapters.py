"""Framework adapter tests.

Hermetic: the vLLM parameter schema and a real startup log are vendored under
``tests/fixtures/vllm``, captured from vLLM 0.19.1 running on an Azure A10. No GPU, no vLLM
install, no network.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from infertune.adapters import (
    Capabilities,
    Diagnostic,
    LaunchSpec,
    ParamRole,
    ParamSchema,
    ParamSpec,
    Severity,
    StartupFacts,
    VLLMAdapter,
    VLLMSchemaError,
    load_schema,
)
from infertune.adapters.vllm import MAX_GPU_MEMORY_UTILIZATION, ROLE_DESTS
from infertune.core.dtypes import DType
from infertune.core.plan import (
    BindingConstraint,
    DtypePlan,
    Parallelism,
    ResourcePlan,
)
from infertune.core.units import gib, mib

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "vllm"
SCHEMA_FILE = FIXTURES / "engine_args_0.19.1.json"
LOG_FILE = FIXTURES / "startup_qwen3-0.6b_0.19.1.log"

# Measured on Azure Standard_NV36ads_A10_v5.
A10_TOTAL_BYTES = mib(24512)


@pytest.fixture
def schema() -> ParamSchema:
    if not SCHEMA_FILE.is_file():
        pytest.skip("vLLM schema fixture not vendored")
    return load_schema(SCHEMA_FILE)


@pytest.fixture
def adapter(schema: ParamSchema) -> VLLMAdapter:
    return VLLMAdapter(schema)


def make_plan(**overrides: object) -> ResourcePlan:
    defaults: dict[str, object] = {
        "weight_bytes_per_gpu": gib(1.11),
        "activation_peak_bytes": gib(0.30),
        "fixed_overhead_bytes": gib(1.00),
        "safety_bytes": gib(0.71),
        "kv_budget_bytes": gib(18.74),
        "kv_bytes_per_token": 112 * 1024,
        "parallelism": Parallelism(),
        "dtypes": DtypePlan(weights=DType.BF16, activations=DType.BF16, kv_cache=DType.BF16),
        "binding_constraint": BindingConstraint.KV_WORKING_SET,
    }
    defaults.update(overrides)
    return ResourcePlan(**defaults)  # type: ignore[arg-type]


class TestSchemaIntrospection:
    def test_schema_loads_real_vllm_surface(self, schema: ParamSchema) -> None:
        assert schema.engine == "vllm"
        assert schema.version == "0.19.1"
        assert len(schema.params) > 100, "expected the full engine-args surface"
        assert schema.source.startswith("recorded:")

    def test_every_role_resolves_or_is_knowably_absent(self, schema: ParamSchema) -> None:
        resolved = {r: schema.resolve(d) for r, d in ROLE_DESTS.items()}
        # Of the roles vLLM maps, only the absolute KV lever is missing in 0.19.1.
        missing = {r.value for r, spec in resolved.items() if spec is None}
        assert missing == {"kv_budget_bytes"}, missing

    def test_token_budget_role_is_unmapped_for_vllm(self) -> None:
        """vLLM has no token-denominated KV lever; SGLang's --max-total-tokens does (M4).

        The role exists in the enum so the abstraction spans both engines, but it must not
        be silently mapped onto something vLLM-shaped.
        """
        assert ParamRole.KV_BUDGET_TOKENS not in ROLE_DESTS

    def test_flag_names_come_from_the_engine_not_from_us(self, schema: ParamSchema) -> None:
        spec = schema.get("tensor_parallel_size")
        assert spec is not None
        assert spec.primary_flag == "--tensor-parallel-size"
        assert "-tp" in spec.flags  # the short alias is the engine's, not invented here

    def test_boolean_negation_flag_is_discovered(self, schema: ParamSchema) -> None:
        spec = schema.get("enable_prefix_caching")
        assert spec is not None
        assert spec.negated_flag() == "--no-enable-prefix-caching"

    def test_rejects_a_schema_for_another_engine(self) -> None:
        with pytest.raises(VLLMSchemaError, match="not vllm"):
            VLLMAdapter(
                ParamSchema(
                    engine="sglang", version="0.0.0", params={"x": ParamSpec("x", ("--x",))}
                )
            )

    def test_load_schema_rejects_empty_dump(self, tmp_path: Path) -> None:
        p = tmp_path / "empty.json"
        p.write_text(json.dumps({"vllm_version": "9.9.9", "cli_flags": {}}))
        with pytest.raises(VLLMSchemaError, match="no cli_flags"):
            load_schema(p)


class TestCapabilities:
    def test_0_19_1_has_no_absolute_kv_lever(self, adapter: VLLMAdapter) -> None:
        """The finding that vindicates runtime introspection.

        docs/plan.md §2.2 planned to compile to --kv-cache-memory. That flag does not exist
        in the only vLLM that runs on an A10, so a hardcoded table would emit an
        unparseable command line.
        """
        caps = adapter.capabilities()
        assert caps.absolute_kv_budget is False
        assert not adapter.schema().has("kv_cache_memory")

    def test_kv_dtype_choices_are_read_from_the_engine(self, adapter: VLLMAdapter) -> None:
        caps = adapter.capabilities()
        assert "fp8_e4m3" in caps.kv_dtypes
        assert "auto" in caps.kv_dtypes

    def test_prefix_caching_detected_as_default_on(self, adapter: VLLMAdapter) -> None:
        caps = adapter.capabilities()
        assert caps.supports_prefix_caching
        assert caps.prefix_caching_default_on


class TestCompile:
    def test_falls_back_to_the_fractional_lever(self, adapter: VLLMAdapter) -> None:
        spec = adapter.compile(
            make_plan(),
            "Qwen/Qwen3-0.6B",
            max_num_seqs=32,
            max_model_len=8192,
            total_vram_bytes=A10_TOTAL_BYTES,
        )
        flags = [f for f, _ in spec.args]
        assert "--gpu-memory-utilization" in flags
        assert "--kv-cache-memory" not in flags
        assert spec.is_runnable
        assert any("no --kv-cache-memory flag" in d.message for d in spec.diagnostics)

    def test_generated_command_uses_only_flags_the_engine_accepts(
        self, adapter: VLLMAdapter, schema: ParamSchema
    ) -> None:
        """The M2 acceptance property: the command line must actually parse."""
        spec = adapter.compile(
            make_plan(parallelism=Parallelism(tensor=2)),
            "Qwen/Qwen3-8B",
            max_num_seqs=64,
            max_model_len=8192,
            max_num_batched_tokens=8192,
            total_vram_bytes=A10_TOTAL_BYTES,
        )
        known = {f for p in schema.params.values() for f in p.flags}
        for flag, _ in spec.args:
            assert flag in known, f"{flag} is not a real vLLM {schema.version} flag"
        assert not spec.errors

    def test_command_line_and_python_kwargs_agree(self, adapter: VLLMAdapter) -> None:
        spec = adapter.compile(
            make_plan(),
            "m",
            max_num_seqs=32,
            max_model_len=4096,
            total_vram_bytes=A10_TOTAL_BYTES,
        )
        cli = spec.command_line("vllm")
        assert cli.startswith("vllm serve m ")
        kw = spec.python_kwargs()
        assert kw["max_model_len"] == "4096"
        assert kw["max_num_seqs"] == "32"

    def test_tensor_parallel_omitted_when_one(self, adapter: VLLMAdapter) -> None:
        spec = adapter.compile(make_plan(), "m", total_vram_bytes=A10_TOTAL_BYTES)
        assert "--tensor-parallel-size" not in [f for f, _ in spec.args]

    def test_eager_emits_a_bare_flag(self, adapter: VLLMAdapter) -> None:
        spec = adapter.compile(
            make_plan(), "m", total_vram_bytes=A10_TOTAL_BYTES, enforce_eager=True
        )
        assert ("--enforce-eager", None) in spec.args
        assert spec.python_kwargs()["enforce_eager"] is True

    def test_missing_total_vram_is_an_error_not_a_guess(self, adapter: VLLMAdapter) -> None:
        spec = adapter.compile(make_plan(), "m", max_num_seqs=8)
        assert not spec.is_runnable
        assert any("total_vram_bytes" in d.message for d in spec.errors)

    def test_fp8_kv_maps_to_an_accepted_spelling(self, adapter: VLLMAdapter) -> None:
        plan = make_plan(
            dtypes=DtypePlan(weights=DType.BF16, activations=DType.BF16, kv_cache=DType.FP8_E4M3)
        )
        spec = adapter.compile(plan, "m", total_vram_bytes=A10_TOTAL_BYTES)
        assert ("--kv-cache-dtype", "fp8_e4m3") in spec.args

    def test_16bit_kv_emits_no_dtype_flag(self, adapter: VLLMAdapter) -> None:
        """Regression guard for a bug only a real engine boot revealed.

        vLLM 0.19.1 lists "bfloat16" among kv_cache_dtype's choices, so a choices-driven
        mapping happily emits ``--kv-cache-dtype bfloat16`` — and engine-core initialisation
        then fails. Omitting the flag lets vLLM derive KV dtype from the model, which works.
        Appearing in ``choices`` does not mean being supported.
        """
        for dtype in (DType.BF16, DType.FP16):
            plan = make_plan(dtypes=DtypePlan(weights=dtype, activations=dtype, kv_cache=dtype))
            spec = adapter.compile(plan, "m", total_vram_bytes=A10_TOTAL_BYTES)
            assert "--kv-cache-dtype" not in [f for f, _ in spec.args], dtype


class TestInversion:
    def test_inversion_excludes_the_safety_margin(self, adapter: VLLMAdapter) -> None:
        """Safety is already inside vLLM's (1 - gmu) slice; counting it twice under-sizes KV."""
        plan = make_plan()
        gmu, _ = adapter.invert_memory_fraction(plan, A10_TOTAL_BYTES)
        expected = (
            plan.kv_budget_bytes
            + plan.weight_bytes_per_gpu
            + plan.activation_peak_bytes
            + plan.fixed_overhead_bytes
        ) / A10_TOTAL_BYTES
        assert gmu == pytest.approx(expected, abs=1e-4)
        with_safety = expected + plan.safety_bytes / A10_TOTAL_BYTES
        assert gmu < with_safety

    def test_inversion_reproduces_the_observed_operating_point(self, adapter: VLLMAdapter) -> None:
        """Sanity anchor against the real boot: gmu 0.85 yielded 18.74 GiB of KV.

        Inverting that same KV budget must land near 0.85, not somewhere unrelated.
        """
        gmu, _ = adapter.invert_memory_fraction(make_plan(), A10_TOTAL_BYTES)
        assert 0.82 <= gmu <= 0.95, gmu

    def test_inversion_is_clamped_and_warns(self, adapter: VLLMAdapter) -> None:
        huge = make_plan(kv_budget_bytes=gib(60))
        gmu, diag = adapter.invert_memory_fraction(huge, A10_TOTAL_BYTES)
        assert gmu == MAX_GPU_MEMORY_UTILIZATION
        assert diag.severity is Severity.WARNING
        assert "clamped" in diag.message

    def test_inversion_rejects_nonsense_total(self, adapter: VLLMAdapter) -> None:
        with pytest.raises(ValueError, match="total_vram_bytes"):
            adapter.invert_memory_fraction(make_plan(), 0)


class TestStartupLogParsing:
    @pytest.fixture
    def facts(self, adapter: VLLMAdapter) -> StartupFacts:
        if not LOG_FILE.is_file():
            pytest.skip("startup log fixture not vendored")
        return adapter.parse_startup_log(LOG_FILE.read_text())

    def test_extracts_kv_ground_truth(self, facts: StartupFacts) -> None:
        assert facts.kv_cache_tokens == 175_456
        assert facts.kv_cache_bytes == pytest.approx(gib(18.74), rel=1e-3)
        assert facts.has_kv_measurement

    def test_engine_observed_bytes_per_token_matches_our_cache_model(
        self, facts: StartupFacts
    ) -> None:
        """Independent confirmation of the KV formula, from vLLM's own numbers.

        Qwen3-0.6B: 28 layers x 8 KV heads x 128 head_dim x 2 (K,V) x 2 bytes = 112 KiB.
        """
        assert facts.bytes_per_token() == pytest.approx(112 * 1024, rel=0.001)

    def test_extracts_weights_and_graph_pool(self, facts: StartupFacts) -> None:
        assert facts.weight_bytes == pytest.approx(gib(1.12), rel=1e-3)
        assert facts.cuda_graph_bytes == pytest.approx(gib(0.10), rel=0.05)
        assert facts.cuda_graph_estimated_bytes is not None

    def test_captures_vllms_own_profiler_error(self, facts: StartupFacts) -> None:
        """vLLM's graph-pool estimate differs from actual; the gap is worth surfacing."""
        assert facts.cuda_graph_bytes != facts.cuda_graph_estimated_bytes

    def test_extracts_effective_args(self, facts: StartupFacts) -> None:
        assert facts.effective_args["gpu_memory_utilization"] == 0.85
        assert facts.effective_args["max_num_seqs"] == 32
        assert facts.effective_args["max_model_len"] == 8192

    def test_extracts_concurrency_estimate(self, facts: StartupFacts) -> None:
        assert facts.context_per_request == 8192
        assert facts.max_concurrency == pytest.approx(21.42, abs=0.01)
        # vLLM's own figure is just tokens / context.
        assert facts.kv_cache_tokens is not None
        assert facts.context_per_request is not None
        ratio = facts.kv_cache_tokens / facts.context_per_request
        assert ratio == pytest.approx(facts.max_concurrency, abs=0.05)

    def test_captures_the_under_accounting_advisory(self, facts: StartupFacts) -> None:
        assert any("gpu-memory-utilization from" in a for a in facts.advisories)

    def test_empty_log_yields_no_false_facts(self, adapter: VLLMAdapter) -> None:
        facts = adapter.parse_startup_log("nothing useful here")
        assert not facts.has_kv_measurement
        assert facts.weight_bytes is None
        assert facts.bytes_per_token() is None


class TestVersionDrift:
    """The adapter must adapt to the engine, not merely handle 0.19.1."""

    def synthetic_with_absolute_lever(self, base: ParamSchema) -> ParamSchema:
        params = dict(base.params)
        params["kv_cache_memory"] = ParamSpec(
            dest="kv_cache_memory", flags=("--kv-cache-memory",), default=None
        )
        return ParamSchema(engine="vllm", version="0.29.0-synthetic", params=params)

    def test_uses_the_absolute_lever_when_the_version_has_it(self, schema: ParamSchema) -> None:
        adapter = VLLMAdapter(self.synthetic_with_absolute_lever(schema))
        assert adapter.capabilities().absolute_kv_budget is True
        plan = make_plan()
        spec = adapter.compile(plan, "m", max_num_seqs=32)
        assert ("--kv-cache-memory", str(plan.kv_budget_bytes)) in spec.args
        assert "--gpu-memory-utilization" not in [f for f, _ in spec.args]
        # No inversion needed, so no total_vram_bytes required and no error.
        assert spec.is_runnable
        assert any("no inversion required" in n for n in spec.notes)

    def test_unknown_flags_are_rejected_by_validation(self, schema: ParamSchema) -> None:
        adapter = VLLMAdapter(schema)
        bogus = LaunchSpec(
            engine="vllm",
            version=schema.version,
            model="m",
            args=(("--not-a-real-flag", "1"),),
        )
        diags = adapter.validate(bogus)
        assert diags
        assert diags[0].severity is Severity.ERROR
        assert "not accepted by vllm 0.19.1" in diags[0].message

    def test_absent_role_degrades_with_a_warning_not_a_crash(self, schema: ParamSchema) -> None:
        stripped = ParamSchema(
            engine="vllm",
            version="stripped",
            params={k: v for k, v in schema.params.items() if k != "max_num_seqs"},
        )
        adapter = VLLMAdapter(stripped)
        spec = adapter.compile(make_plan(), "m", max_num_seqs=32, total_vram_bytes=A10_TOTAL_BYTES)
        assert any(
            d.role is ParamRole.MAX_SEQS and d.severity is Severity.WARNING
            for d in spec.diagnostics
        )
        assert "--max-num-seqs" not in [f for f, _ in spec.args]


def test_diagnostic_renders_readably() -> None:
    assert str(Diagnostic(Severity.WARNING, "careful")) == "warning: careful"


def test_capabilities_is_hashable_dataclass() -> None:
    c = Capabilities(engine="vllm", version="1", absolute_kv_budget=False)
    assert c.engine == "vllm"
