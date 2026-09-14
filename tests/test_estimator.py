"""Memory estimator: ledger reconciliation, feasibility, and binding constraints."""

from __future__ import annotations

import pytest

from infertune.core.cache import AttentionKind, AttentionSpec, LayeredCacheSpec, RecurrentSpec
from infertune.core.dtypes import DType
from infertune.core.model import ModelProfile
from infertune.core.plan import BindingConstraint, Parallelism
from infertune.core.units import MIB, gib
from infertune.core.workload import Constant, LogNormal, WorkloadProfile
from infertune.estimator.memory import (
    InfeasibleConfigurationError,
    estimate_overheads,
    estimate_plan,
    max_concurrency_for_budget,
    with_kv_dtype,
)
from infertune.hardware import specdb

WORKLOAD = WorkloadProfile(
    input_tokens=LogNormal.from_median_p95(1024, 2048),
    output_tokens=LogNormal.from_median_p95(256, 512),
)

LLAMA_31_8B = ModelProfile(
    model_id="meta-llama/Llama-3.1-8B",
    n_params_total=8_030_261_248,
    n_params_active=8_030_261_248,
    weight_bytes=16_060_522_496,
    n_layers=32,
    hidden_size=4096,
    n_heads=32,
    vocab_size=128_256,
    max_position_embeddings=131_072,
    cache=AttentionSpec(kind=AttentionKind.GQA, n_kv_heads=8, head_dim=128, n_kv_layers=32),
    weight_dtype=DType.BF16,
    weight_bytes_source="test-fixture",
)

NEMOTRON_H = ModelProfile(
    model_id="nvidia/Nemotron-H-8B",
    n_params_total=8_100_852_736,
    n_params_active=8_100_852_736,
    weight_bytes=16_201_705_472,
    n_layers=52,
    hidden_size=4096,
    n_heads=32,
    vocab_size=131_072,
    max_position_embeddings=8192,
    cache=LayeredCacheSpec(
        (
            AttentionSpec(kind=AttentionKind.GQA, n_kv_heads=8, head_dim=128, n_kv_layers=4),
            RecurrentSpec(
                d_inner=8192,
                n_groups=8,
                ssm_state_size=128,
                conv_kernel=4,
                n_recurrent_layers=24,
            ),
        )
    ),
    weight_dtype=DType.BF16,
    weight_bytes_source="test-fixture",
)


class TestLedgerReconciliation:
    def test_ledger_sums_exactly_to_usable_vram(self) -> None:
        """The invariant that catches missing or double-counted terms."""
        gpu = specdb.load("a100-80gb")
        plan = estimate_plan(
            LLAMA_31_8B, gpu, WORKLOAD, max_num_seqs=32, max_model_len=8192, samples=800
        )
        assert plan.reconciles_with(gpu.vram_usable_bytes)
        assert plan.total_allocated_bytes == gpu.vram_usable_bytes

    def test_consumption_entries_sum_to_usable(self) -> None:
        gpu = specdb.load("h100-sxm")
        plan = estimate_plan(
            LLAMA_31_8B, gpu, WORKLOAD, max_num_seqs=64, max_model_len=8192, samples=800
        )
        consumed = sum(e.bytes_ for e in plan.ledger if not e.is_available)
        assert consumed == gpu.vram_usable_bytes

    def test_every_entry_explains_itself(self) -> None:
        plan = estimate_plan(
            LLAMA_31_8B,
            specdb.load("h100-sxm"),
            WORKLOAD,
            max_num_seqs=32,
            max_model_len=8192,
            samples=500,
        )
        for entry in plan.ledger:
            assert entry.formula, f"{entry.label} has no formula"
            assert entry.provenance, f"{entry.label} has no provenance"

    def test_reconciliation_holds_across_gpus_and_concurrency(self) -> None:
        for key in ("rtx-4090", "l40s", "a100-80gb", "h100-sxm", "h200-sxm"):
            gpu = specdb.load(key)
            for seqs in (1, 16, 128):
                plan = estimate_plan(
                    LLAMA_31_8B,
                    gpu,
                    WORKLOAD,
                    max_num_seqs=seqs,
                    max_model_len=4096,
                    samples=300,
                )
                assert plan.reconciles_with(gpu.vram_usable_bytes), f"{key} @ {seqs}"


class TestFeasibility:
    def test_weights_larger_than_vram_is_infeasible(self) -> None:
        with pytest.raises(InfeasibleConfigurationError, match="no KV cache fits"):
            estimate_plan(
                LLAMA_31_8B,
                specdb.load("t4"),
                WORKLOAD,
                max_num_seqs=16,
                max_model_len=4096,
                samples=200,
            )

    def test_infeasibility_message_suggests_remedies(self) -> None:
        with pytest.raises(InfeasibleConfigurationError, match="Shard across GPUs"):
            estimate_plan(
                LLAMA_31_8B,
                specdb.load("t4"),
                WORKLOAD,
                max_num_seqs=16,
                max_model_len=4096,
                samples=200,
            )

    def test_tensor_parallelism_makes_a_tight_fit_feasible(self) -> None:
        """Sharding weights frees room for KV — the standard remedy, verified.

        A 16 GB bf16 checkpoint does not fit one 14.76 GiB T4 at all, but four of them
        shard the weights to 3.7 GiB each and leave ample KV room.
        """
        gpu = specdb.load("t4", count=4)
        with pytest.raises(InfeasibleConfigurationError):
            estimate_plan(
                LLAMA_31_8B, gpu, WORKLOAD, max_num_seqs=16, max_model_len=4096, samples=200
            )
        plan = estimate_plan(
            LLAMA_31_8B,
            gpu,
            WORKLOAD,
            max_num_seqs=16,
            max_model_len=4096,
            parallelism=Parallelism(tensor=4),
            samples=400,
        )
        assert plan.kv_budget_bytes > 0
        assert plan.weight_bytes_per_gpu < LLAMA_31_8B.weight_bytes
        assert plan.reconciles_with(gpu.vram_usable_bytes)

    def test_requesting_more_gpus_than_available_refuses(self) -> None:
        with pytest.raises(InfeasibleConfigurationError, match="needs 4 GPUs"):
            estimate_plan(
                LLAMA_31_8B,
                specdb.load("h100-sxm"),
                WORKLOAD,
                max_num_seqs=16,
                max_model_len=4096,
                parallelism=Parallelism(tensor=4),
                samples=200,
            )

    def test_sub_byte_kv_dtype_rejected(self) -> None:
        with pytest.raises(ValueError, match="not a valid KV cache dtype"):
            estimate_plan(
                LLAMA_31_8B,
                specdb.load("h100-sxm"),
                WORKLOAD,
                max_num_seqs=16,
                max_model_len=4096,
                kv_dtype=DType.INT4,
                samples=200,
            )


class TestOverheadTerms:
    def test_logits_scale_with_concurrency_and_vocabulary(self) -> None:
        """A term that is easy to forget and reaches GiB scale on large vocabularies."""
        small = estimate_overheads(
            LLAMA_31_8B,
            max_num_seqs=8,
            max_num_batched_tokens=2048,
            parallelism=Parallelism(),
        )
        large = estimate_overheads(
            LLAMA_31_8B,
            max_num_seqs=256,
            max_num_batched_tokens=2048,
            parallelism=Parallelism(),
        )
        assert large.logits_bytes == 32 * small.logits_bytes
        assert large.logits_bytes / MIB > 350

    def test_prefill_activations_track_the_token_budget_not_concurrency(self) -> None:
        base = estimate_overheads(
            LLAMA_31_8B,
            max_num_seqs=16,
            max_num_batched_tokens=2048,
            parallelism=Parallelism(),
        )
        more_tokens = estimate_overheads(
            LLAMA_31_8B,
            max_num_seqs=16,
            max_num_batched_tokens=8192,
            parallelism=Parallelism(),
        )
        more_seqs = estimate_overheads(
            LLAMA_31_8B,
            max_num_seqs=64,
            max_num_batched_tokens=2048,
            parallelism=Parallelism(),
        )
        assert more_tokens.prefill_activation_bytes == 4 * base.prefill_activation_bytes
        assert more_seqs.prefill_activation_bytes == base.prefill_activation_bytes

    def test_enforce_eager_reclaims_graph_and_compile_memory(self) -> None:
        graphed = estimate_overheads(
            LLAMA_31_8B,
            max_num_seqs=64,
            max_num_batched_tokens=4096,
            parallelism=Parallelism(),
        )
        eager = estimate_overheads(
            LLAMA_31_8B,
            max_num_seqs=64,
            max_num_batched_tokens=4096,
            parallelism=Parallelism(),
            enforce_eager=True,
        )
        assert eager.cuda_graph_bytes == 0
        assert eager.compile_workspace_bytes == 0
        assert eager.total < graphed.total

    def test_nccl_buffers_only_with_parallelism(self) -> None:
        single = estimate_overheads(
            LLAMA_31_8B,
            max_num_seqs=8,
            max_num_batched_tokens=2048,
            parallelism=Parallelism(),
        )
        sharded = estimate_overheads(
            LLAMA_31_8B,
            max_num_seqs=8,
            max_num_batched_tokens=2048,
            parallelism=Parallelism(tensor=2),
        )
        assert single.nccl_bytes == 0
        assert sharded.nccl_bytes > 0

    def test_eager_mode_yields_a_larger_kv_budget(self) -> None:
        gpu = specdb.load("rtx-4090")
        kwargs = {"max_num_seqs": 24, "max_model_len": 8192, "samples": 400}
        graphed = estimate_plan(LLAMA_31_8B, gpu, WORKLOAD, **kwargs)  # type: ignore[arg-type]
        eager = estimate_plan(
            LLAMA_31_8B,
            gpu,
            WORKLOAD,
            enforce_eager=True,
            **kwargs,  # type: ignore[arg-type]
        )
        assert eager.kv_budget_bytes > graphed.kv_budget_bytes


class TestBindingConstraint:
    def test_kv_bound_when_working_set_exceeds_budget(self) -> None:
        heavy = WorkloadProfile(input_tokens=Constant(8000), output_tokens=Constant(2000))
        plan = estimate_plan(
            LLAMA_31_8B,
            specdb.load("rtx-4090"),
            heavy,
            max_num_seqs=64,
            max_model_len=16384,
            samples=400,
        )
        assert plan.binding_constraint is BindingConstraint.KV_WORKING_SET

    def test_never_uses_the_worst_case_product_reasoning(self) -> None:
        """Regression guard for docs/plan.md 2.1.

        A tiny workload on a large GPU must not be called KV-bound merely because the cache
        cannot hold max_num_seqs x max_model_len — that is the reservation reading this
        project rejects.
        """
        tiny = WorkloadProfile(input_tokens=Constant(64), output_tokens=Constant(16))
        plan = estimate_plan(
            LLAMA_31_8B,
            specdb.load("h200-sxm"),
            tiny,
            max_num_seqs=8,
            max_model_len=131_072,
            samples=400,
        )
        assert plan.kv_budget_tokens < 8 * 131_072, "precondition: worst case does not fit"
        assert plan.binding_constraint is not BindingConstraint.KV_WORKING_SET

    def test_compute_bound_above_the_critical_batch_size(self) -> None:
        tiny = WorkloadProfile(input_tokens=Constant(128), output_tokens=Constant(32))
        gpu = specdb.load("rtx-4090")
        plan = estimate_plan(
            LLAMA_31_8B, gpu, tiny, max_num_seqs=256, max_model_len=2048, samples=400
        )
        assert plan.critical_batch_size is not None
        assert plan.critical_batch_size <= 256
        assert plan.binding_constraint is BindingConstraint.COMPUTE_BOUND

    def test_hybrid_models_warn_that_part_of_memory_is_a_reservation(self) -> None:
        """Nemotron-H's Mamba state scales with max_num_seqs directly, not with a paged cache.

        The model still has 4 attention layers, so context *does* scale — the report must
        distinguish the reserved component from the paged one rather than treating the whole
        cache as one or the other.
        """
        plan = estimate_plan(
            NEMOTRON_H,
            specdb.load("h100-sxm"),
            WORKLOAD,
            max_num_seqs=16,
            max_model_len=8192,
            samples=400,
        )
        assert any("genuine reservation" in w for w in plan.warnings)
        assert NEMOTRON_H.cache.scales_with_context, "4 attention layers still scale"
        assert any(e.label == "recurrent state" for e in plan.ledger)

    def test_purely_recurrent_models_have_no_context_cost(self) -> None:
        pure = ModelProfile(
            model_id="test/pure-mamba",
            n_params_total=1_000_000_000,
            n_params_active=1_000_000_000,
            weight_bytes=2_000_000_000,
            n_layers=24,
            hidden_size=2048,
            n_heads=16,
            vocab_size=32_000,
            max_position_embeddings=8192,
            cache=RecurrentSpec(
                d_inner=4096,
                n_groups=8,
                ssm_state_size=128,
                conv_kernel=4,
                n_recurrent_layers=24,
            ),
            weight_dtype=DType.BF16,
            weight_bytes_source="test-fixture",
        )
        plan = estimate_plan(
            pure,
            specdb.load("a100-80gb"),
            WORKLOAD,
            max_num_seqs=16,
            max_model_len=8192,
            samples=200,
        )
        assert plan.binding_constraint is BindingConstraint.USER_LIMIT
        assert any("no memory cost" in w for w in plan.warnings)

    def test_naive_overstatement_is_surfaced(self) -> None:
        plan = estimate_plan(
            LLAMA_31_8B,
            specdb.load("h100-sxm"),
            WORKLOAD,
            max_num_seqs=32,
            max_model_len=8192,
            samples=1500,
        )
        assert any("overstated" in w for w in plan.warnings)


class TestHeadroom:
    def test_headroom_is_reported_and_positive(self) -> None:
        plan = estimate_plan(
            LLAMA_31_8B,
            specdb.load("a100-80gb"),
            WORKLOAD,
            max_num_seqs=32,
            max_model_len=8192,
            samples=800,
        )
        assert plan.headroom_concurrency is not None
        assert plan.headroom_concurrency >= 32

    def test_cache_limited_flag_compares_headroom_to_the_compute_knee(self) -> None:
        plan = estimate_plan(
            LLAMA_31_8B,
            specdb.load("rtx-4090"),
            WORKLOAD,
            max_num_seqs=24,
            max_model_len=8192,
            samples=800,
        )
        assert plan.headroom_concurrency is not None
        assert plan.critical_batch_size is not None
        assert plan.cache_limited == (plan.headroom_concurrency < plan.critical_batch_size)

    def test_fp8_kv_doubles_token_capacity_without_changing_the_budget(self) -> None:
        gpu = specdb.load("rtx-4090")
        plan = estimate_plan(
            LLAMA_31_8B, gpu, WORKLOAD, max_num_seqs=24, max_model_len=8192, samples=400
        )
        fp8 = with_kv_dtype(plan, LLAMA_31_8B.cache, DType.FP8_E4M3)
        assert fp8.kv_budget_bytes == plan.kv_budget_bytes
        assert fp8.kv_budget_tokens == pytest.approx(2 * plan.kv_budget_tokens, abs=2)

    def test_headroom_accounts_for_fixed_recurrent_state(self) -> None:
        """The bug this test locks down.

        Nemotron-H holds ~50 MiB of Mamba state per sequence. A KV-only headroom search
        ignores that and returns a concurrency whose state alone would not fit.
        """
        budget = gib(6)
        cache = NEMOTRON_H.cache
        with_state = max_concurrency_for_budget(
            cache, WORKLOAD, budget, kv_dtype=DType.BF16, samples=600
        )
        per_seq = cache.fixed_bytes_per_sequence(DType.BF16)
        assert per_seq > 40 * MIB, "precondition: state per sequence is large"
        assert with_state * per_seq <= budget, "headroom must not exceed the state budget"

        kv_only = budget // max(1, cache.marginal_bytes_per_token(DType.BF16))
        assert with_state < kv_only, "ignoring fixed state would overstate headroom"

    def test_headroom_is_monotone_in_budget(self) -> None:
        cache = LLAMA_31_8B.cache
        small = max_concurrency_for_budget(
            cache, WORKLOAD, gib(2), kv_dtype=DType.BF16, samples=400
        )
        large = max_concurrency_for_budget(
            cache, WORKLOAD, gib(16), kv_dtype=DType.BF16, samples=400
        )
        assert large > small

    def test_zero_budget_gives_zero_headroom(self) -> None:
        assert (
            max_concurrency_for_budget(
                LLAMA_31_8B.cache, WORKLOAD, 0, kv_dtype=DType.BF16, samples=100
            )
            == 0
        )
