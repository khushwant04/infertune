"""SGLang adapter.

This suite is the acceptance test for the abstraction, not just for SGLang. The interesting
assertions are the ones showing that a *second* engine differs from the first in ways the role
indirection absorbs: an absolute token lever, a differently-scoped memory fraction, and
inverted flag polarity.

Hermetic: the schema comes from a vendored dump, so no SGLang install is needed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from infertune.adapters import (
    ParamRole,
    ParamSchema,
    ParamSpec,
    Severity,
    UnknownEngineError,
    available_engines,
    get_adapter,
)
from infertune.adapters.sglang import (
    MAX_MEM_FRACTION_STATIC,
    NEGATED_ROLES,
    ROLE_DESTS,
    SGLangAdapter,
    SGLangSchemaError,
    load_schema,
)
from infertune.adapters.vllm import ROLE_DESTS as VLLM_ROLE_DESTS
from infertune.core.dtypes import DType
from infertune.core.plan import (
    BindingConstraint,
    DtypePlan,
    Parallelism,
    ResourcePlan,
)
from infertune.core.units import gib, mib

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "sglang" / "server_args_doc_derived.json"
A10_TOTAL = mib(24512)


@pytest.fixture
def schema() -> ParamSchema:
    if not FIXTURE.is_file():
        pytest.skip("sglang schema fixture not vendored")
    return load_schema(FIXTURE)


@pytest.fixture
def adapter(schema: ParamSchema) -> SGLangAdapter:
    return SGLangAdapter(schema)


def make_plan(**overrides: object) -> ResourcePlan:
    defaults: dict[str, object] = {
        "weight_bytes_per_gpu": gib(14.96),
        "activation_peak_bytes": gib(0.50),
        "fixed_overhead_bytes": gib(0.35),
        "safety_bytes": gib(0.66),
        "kv_budget_bytes": gib(4.80),
        "kv_bytes_per_token": 144 * 1024,
        "parallelism": Parallelism(),
        "dtypes": DtypePlan(weights=DType.BF16, activations=DType.BF16, kv_cache=DType.BF16),
        "binding_constraint": BindingConstraint.KV_WORKING_SET,
    }
    defaults.update(overrides)
    return ResourcePlan(**defaults)  # type: ignore[arg-type]


class TestRegistry:
    def test_both_engines_are_registered(self) -> None:
        assert available_engines() == ("sglang", "vllm")

    def test_get_adapter_by_name(self, schema: ParamSchema) -> None:
        assert get_adapter("sglang", schema).name == "sglang"
        assert get_adapter("SGLang", schema).name == "sglang"

    def test_unknown_engine_lists_alternatives(self) -> None:
        with pytest.raises(UnknownEngineError, match="available"):
            get_adapter("tensorrt-llm")

    def test_registry_is_what_makes_callers_engine_agnostic(self) -> None:
        """Callers name an engine string; they never import an adapter class.

        Without this, the CLI, the search loop and the validation script would each need
        editing to add an engine.
        """
        from infertune.adapters import registry

        assert set(registry._FACTORIES) == {"vllm", "sglang"}


class TestSchema:
    def test_loads_the_vendored_dump(self, schema: ParamSchema) -> None:
        assert schema.engine == "sglang"
        assert schema.source.startswith("recorded:")
        assert schema.has("mem_fraction_static")
        assert schema.has("max_total_tokens")

    def test_every_role_resolves(self, schema: ParamSchema) -> None:
        missing = {
            role.value for role, dests in ROLE_DESTS.items() if schema.resolve(dests) is None
        }
        assert not missing, missing

    def test_rejects_a_schema_for_another_engine(self) -> None:
        wrong = ParamSchema(engine="vllm", version="0", params={"x": ParamSpec("x", ("--x",))})
        with pytest.raises(SGLangSchemaError, match="not sglang"):
            SGLangAdapter(wrong)

    def test_rejects_an_empty_dump(self, tmp_path: Path) -> None:
        p = tmp_path / "e.json"
        p.write_text(json.dumps({"sglang_version": "x", "cli_flags": {}}))
        with pytest.raises(SGLangSchemaError, match="no cli_flags"):
            load_schema(p)


class TestEngineDifferences:
    """Where SGLang and vLLM genuinely differ. This is what the abstraction has to absorb."""

    def test_sglang_has_an_absolute_token_lever_where_vllm_has_none(
        self, adapter: SGLangAdapter
    ) -> None:
        """The role exists in the enum precisely because engines differ here.

        M2 asserted vLLM leaves KV_BUDGET_TOKENS unmapped; SGLang maps it to
        --max-total-tokens, so no fraction inversion is needed at all.
        """
        assert ParamRole.KV_BUDGET_TOKENS in ROLE_DESTS
        assert ParamRole.KV_BUDGET_TOKENS not in VLLM_ROLE_DESTS
        assert adapter.capabilities().absolute_kv_budget is True

    def test_prefers_the_absolute_lever_and_needs_no_vram_total(
        self, adapter: SGLangAdapter
    ) -> None:
        plan = make_plan()
        spec = adapter.compile(plan, "m", max_num_seqs=32, max_model_len=8192)
        assert ("--max-total-tokens", str(plan.kv_budget_tokens)) in spec.args
        assert "--mem-fraction-static" not in [f for f, _ in spec.args]
        assert spec.is_runnable, "no inversion means no total_vram_bytes requirement"
        assert any("no fraction inversion needed" in n for n in spec.notes)

    def test_memory_fraction_is_scoped_differently_than_vllms(self, adapter: SGLangAdapter) -> None:
        """--mem-fraction-static covers weights + KV only; activations sit outside.

        vLLM's --gpu-memory-utilization covers weights + activations + KV. The same numeric
        value therefore describes different machines, which is exactly why plans carry bytes.
        """
        plan = make_plan()
        fraction, note = adapter.invert_memory_fraction(plan, A10_TOTAL)
        expected = (plan.kv_budget_bytes + plan.weight_bytes_per_gpu) / A10_TOTAL
        assert fraction == pytest.approx(expected, abs=1e-4)
        assert "activations sit outside" in note.message

        # The vLLM formula would also include activations, giving a strictly larger number.
        vllm_style = (
            plan.kv_budget_bytes
            + plan.weight_bytes_per_gpu
            + plan.activation_peak_bytes
            + plan.fixed_overhead_bytes
        ) / A10_TOTAL
        assert vllm_style > fraction

    def test_polarity_is_inverted_for_some_roles(self, adapter: SGLangAdapter) -> None:
        """SGLang says --disable-cuda-graph where vLLM says --enforce-eager."""
        assert ParamRole.EAGER in NEGATED_ROLES
        assert ParamRole.PREFIX_CACHING in NEGATED_ROLES
        spec = adapter.compile(make_plan(), "m", enforce_eager=True)
        assert ("--disable-cuda-graph", None) in spec.args
        assert "--enforce-eager" not in [f for f, _ in spec.args]

    def test_launch_command_shape_differs(self, adapter: SGLangAdapter) -> None:
        """SGLang launches a module with --model-path, not `serve <model>`."""
        spec = adapter.compile(make_plan(), "Qwen/Qwen3-8B", max_num_seqs=16)
        cmd = adapter.command(spec)
        assert cmd[:4] == ["python", "-m", "sglang.launch_server", "--model-path"]
        assert cmd[4] == "Qwen/Qwen3-8B"


class TestCompile:
    def test_uses_only_flags_the_engine_accepts(
        self, adapter: SGLangAdapter, schema: ParamSchema
    ) -> None:
        spec = adapter.compile(
            make_plan(parallelism=Parallelism(tensor=2)),
            "m",
            max_num_seqs=64,
            max_model_len=8192,
            max_num_batched_tokens=8192,
        )
        known = {f for p in schema.params.values() for f in p.flags}
        for flag, _ in spec.args:
            assert flag in known, f"{flag} is not a real SGLang flag"
        assert not spec.errors

    def test_maps_context_and_concurrency_to_sglang_names(self, adapter: SGLangAdapter) -> None:
        spec = adapter.compile(make_plan(), "m", max_num_seqs=48, max_model_len=4096)
        flags = dict(spec.args)
        assert flags["--context-length"] == "4096"
        assert flags["--max-running-requests"] == "48"

    def test_prefill_budget_maps_to_chunked_prefill_size(self, adapter: SGLangAdapter) -> None:
        spec = adapter.compile(make_plan(), "m", max_num_batched_tokens=4096)
        assert ("--chunked-prefill-size", "4096") in spec.args

    def test_fp8_kv_uses_an_accepted_spelling(self, adapter: SGLangAdapter) -> None:
        plan = make_plan(
            dtypes=DtypePlan(weights=DType.BF16, activations=DType.BF16, kv_cache=DType.FP8_E4M3)
        )
        spec = adapter.compile(plan, "m")
        assert ("--kv-cache-dtype", "fp8_e4m3") in spec.args

    def test_16bit_kv_emits_no_dtype_flag(self, adapter: SGLangAdapter) -> None:
        spec = adapter.compile(make_plan(), "m")
        assert "--kv-cache-dtype" not in [f for f, _ in spec.args]

    def test_tensor_parallel_uses_tp_size(self, adapter: SGLangAdapter) -> None:
        spec = adapter.compile(make_plan(parallelism=Parallelism(tensor=4)), "m")
        assert ("--tp-size", "4") in spec.args

    def test_fraction_path_requires_total_vram(self, adapter: SGLangAdapter) -> None:
        spec = adapter.compile(make_plan(), "m", prefer_absolute_kv=False)
        assert not spec.is_runnable
        assert any("total_vram_bytes" in d.message for d in spec.errors)

    def test_fraction_is_clamped_with_sglangs_own_advice(self, adapter: SGLangAdapter) -> None:
        huge = make_plan(kv_budget_bytes=gib(60))
        fraction, note = adapter.invert_memory_fraction(huge, A10_TOTAL)
        assert fraction == MAX_MEM_FRACTION_STATIC
        assert note.severity is Severity.WARNING
        assert "lower this value" in note.message

    def test_absent_role_warns_rather_than_crashing(self, schema: ParamSchema) -> None:
        stripped = ParamSchema(
            engine="sglang",
            version="stripped",
            params={k: v for k, v in schema.params.items() if k != "max_running_requests"},
        )
        spec = SGLangAdapter(stripped).compile(make_plan(), "m", max_num_seqs=32)
        assert any(
            d.role is ParamRole.MAX_SEQS and d.severity is Severity.WARNING
            for d in spec.diagnostics
        )


class TestStartupLogParsing:
    def test_extracts_the_kv_pool_size(self, adapter: SGLangAdapter) -> None:
        """SGLang's max_total_num_tokens is the same quantity vLLM calls GPU KV cache size."""
        log = (
            "[2026-09-14 12:00:00] max_total_num_tokens=175456, "
            "chunked_prefill_size=8192, max_prefill_tokens=32768"
        )
        facts = adapter.parse_startup_log(log)
        assert facts.engine == "sglang"
        assert facts.kv_cache_tokens == 175_456
        assert facts.effective_args["chunked_prefill_size"] == 8192

    def test_empty_log_yields_no_false_facts(self, adapter: SGLangAdapter) -> None:
        facts = adapter.parse_startup_log("nothing here")
        assert not facts.has_kv_measurement
        assert facts.weight_bytes is None
