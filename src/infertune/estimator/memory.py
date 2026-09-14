"""Memory estimation: overhead terms, KV budget, and a ledger that reconciles.

The estimator's job is to partition usable VRAM into named terms that sum **exactly** to
the total, leaving whatever remains as the KV budget. A ledger that does not reconcile has
a missing or double-counted term, which is the bug class that produces boot-time OOM — so
reconciliation is asserted, not assumed.

The terms people forget are sized explicitly here, because they are what turn "it worked at
batch 32" into "it OOMs at batch 64":

* **Logits and sampling buffers** scale as ``max_num_seqs x vocab_size``. With a 201k-token
  vocabulary (gpt-oss) and 256 sequences that is 206 MiB *per buffer*, and several exist.
* **Prefill activations** scale with the *token budget*, not ``max_num_seqs``.
* **CUDA graph pool** is 0.5-3 GiB depending on captured batch sizes, and is reclaimed
  entirely by ``--enforce-eager`` / ``-O0``.
* **Fragmentation** is why a safety margin is not superstition.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from ..core.cache import CacheSpec, RecurrentSpec
from ..core.dtypes import DType, bytes_for
from ..core.gpu import GPUProfile
from ..core.model import ModelProfile
from ..core.plan import (
    BindingConstraint,
    DtypePlan,
    LedgerEntry,
    Parallelism,
    ResourcePlan,
)
from ..core.units import GIB, MIB, fmt_bytes
from ..core.workload import WorkloadProfile

CUDA_CONTEXT_BYTES = 300 * MIB
"""CUDA context, driver structures, and cuBLAS/cuDNN workspaces.

**Non-torch**: allocated outside PyTorch's allocator, so it is *not* counted inside an
engine's memory-utilisation budget. On bare metal this is a few hundred MiB; on a vGPU it is
far larger (2.35 GiB measured on an Azure A10-24Q), which is why the real figure should come
from ``torch.cuda.mem_get_info`` when a device is present rather than from this prior.
"""

NCCL_BYTES_PER_RANK = 220 * MIB
"""Communication buffers, per rank once tensor parallelism is in use. Non-torch."""

COMPILE_WORKSPACE_BYTES = 400 * MIB
"""torch.compile / inductor scratch at the default optimization level. Non-torch."""

CUDA_GRAPH_BASE_BYTES = 64 * MIB
CUDA_GRAPH_BYTES_PER_SEQ_PER_HIDDEN = 1200
"""CUDA graph pool, as bytes per captured batch size per unit of hidden dimension.

Calibrated against a measured boot: vLLM 0.19.1 with Qwen3-0.6B (hidden 1024) at
``max_num_seqs=32`` captured a **102 MiB** pool, and reported its own estimate as 0.11 GiB.
An earlier fixed prior of 600 MiB + 4 MiB/seq predicted 728 MiB — 7x too high, which fed
straight into an over-aggressive utilisation and an OOM. Scaling with hidden size is the
physically sensible form, since captured graphs hold activation buffers.

Single-point calibration; M3 refits it from the measurement store.
"""

ACTIVATION_ELEMENTS = 2
"""Live copies of the (residual + MLP-intermediate) stream during a prefill chunk.

Fitted against three measured vLLM boots on an A10 (Qwen3-0.6B, Qwen3-4B,
Qwen2.5-7B-AWQ); worst-case KV prediction error falls from 5.07% to ~1%. M3 refits this
from the measurement store.
"""

ATTENTION_WORKSPACE_BYTES = 128 * MIB
"""FlashAttention/FlashInfer scratch. Torch-side."""

FRAGMENTATION_FRACTION = 0.03
"""Allocator fragmentation and rounding, as a share of usable VRAM."""

LOGITS_BUFFER_COPIES = 3
"""Logits, a softmax/probability buffer, and sorting scratch for top-k/top-p."""


class InfeasibleConfigurationError(ValueError):
    """Raised when no KV cache can be allocated at all."""


@dataclass(frozen=True, slots=True)
class EngineOverheads:
    """Per-GPU overhead terms other than weights and KV cache.

    Split by **which allocator owns them**, because engines size their cache as a fraction of
    total device memory minus their own *PyTorch* usage. Non-torch memory sits outside that
    budget, so conflating the two makes the engine appear to have more room than it does.
    Getting this wrong is not academic: it produced a `gpu_memory_utilization` of 0.9613 on a
    card whose real ceiling is 0.901, and vLLM OOMed during sampler warm-up.
    """

    cuda_context_bytes: int
    nccl_bytes: int
    compile_workspace_bytes: int
    cuda_graph_bytes: int
    attention_workspace_bytes: int
    prefill_activation_bytes: int
    logits_bytes: int

    @property
    def torch_side(self) -> int:
        """Overheads allocated through PyTorch, i.e. inside the engine's own budget."""
        return (
            self.cuda_graph_bytes
            + self.attention_workspace_bytes
            + self.prefill_activation_bytes
            + self.logits_bytes
        )

    @property
    def non_torch(self) -> int:
        """Overheads outside PyTorch's allocator, and outside the engine's budget."""
        return self.cuda_context_bytes + self.nccl_bytes + self.compile_workspace_bytes

    @property
    def total(self) -> int:
        return self.torch_side + self.non_torch


def estimate_overheads(
    model: ModelProfile,
    *,
    max_num_seqs: int,
    max_num_batched_tokens: int,
    parallelism: Parallelism,
    activation_dtype: DType = DType.BF16,
    enforce_eager: bool = False,
) -> EngineOverheads:
    """Size the non-weight, non-KV memory terms."""
    if max_num_seqs < 1:
        raise ValueError(f"max_num_seqs must be >= 1, got {max_num_seqs}")
    if max_num_batched_tokens < 1:
        raise ValueError(f"max_num_batched_tokens must be >= 1, got {max_num_batched_tokens}")

    tp = parallelism.tensor
    hidden_per_gpu = max(1, model.hidden_size // tp)

    # Prefill activations track the token budget, and are dominated by the MLP
    # intermediate rather than the residual stream. Measured against vLLM on an A10, a
    # hidden-only form under-estimated this by ~530 MiB on Qwen2.5-7B (intermediate 5.3x
    # hidden), which alone pushed KV prediction error past 5%.
    inner_per_gpu = max(1, model.effective_intermediate_size // tp)
    prefill = bytes_for(
        max_num_batched_tokens * (hidden_per_gpu + inner_per_gpu) * ACTIVATION_ELEMENTS,
        activation_dtype,
    )

    # Logits are computed in fp32 and are not sharded across TP ranks.
    logits = max_num_seqs * model.vocab_size * 4 * LOGITS_BUFFER_COPIES

    graph = (
        0
        if enforce_eager
        else CUDA_GRAPH_BASE_BYTES
        + max_num_seqs * hidden_per_gpu * CUDA_GRAPH_BYTES_PER_SEQ_PER_HIDDEN
    )

    return EngineOverheads(
        cuda_context_bytes=CUDA_CONTEXT_BYTES,
        nccl_bytes=NCCL_BYTES_PER_RANK if tp > 1 or parallelism.pipeline > 1 else 0,
        compile_workspace_bytes=0 if enforce_eager else COMPILE_WORKSPACE_BYTES,
        cuda_graph_bytes=graph,
        attention_workspace_bytes=ATTENTION_WORKSPACE_BYTES,
        prefill_activation_bytes=prefill,
        logits_bytes=logits,
    )


def _cache_bytes_per_token(cache: CacheSpec, kv_dtype: DType, tp: int) -> int:
    """Marginal cache bytes per token, with a floor of 1 to keep division safe."""
    return max(1, cache.marginal_bytes_per_token(kv_dtype, tp))


def estimate_plan(
    model: ModelProfile,
    gpu: GPUProfile,
    workload: WorkloadProfile,
    *,
    max_num_seqs: int,
    max_model_len: int,
    max_num_batched_tokens: int | None = None,
    parallelism: Parallelism | None = None,
    kv_dtype: DType | None = None,
    activation_dtype: DType = DType.BF16,
    enforce_eager: bool = False,
    samples: int = 8000,
    seed: int = 0,
) -> ResourcePlan:
    """Produce a reconciling :class:`ResourcePlan` for one GPU.

    Raises:
        InfeasibleConfigurationError: when weights plus overheads leave no room for KV.
    """
    parallelism = parallelism or Parallelism()
    kv_dtype = kv_dtype or model.weight_dtype
    if kv_dtype.is_sub_byte:
        raise ValueError(f"{kv_dtype} is not a valid KV cache dtype")
    if max_model_len < 1:
        raise ValueError(f"max_model_len must be >= 1, got {max_model_len}")
    token_budget = max_num_batched_tokens or min(max_model_len, 8192)

    tp = parallelism.tensor
    if parallelism.gpus_required > gpu.count:
        raise InfeasibleConfigurationError(
            f"parallelism needs {parallelism.gpus_required} GPUs but the profile has {gpu.count}"
        )

    usable = gpu.vram_usable_bytes
    weights = model.weight_bytes_per_gpu(tp)
    overheads = estimate_overheads(
        model,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=token_budget,
        parallelism=parallelism,
        activation_dtype=activation_dtype,
        enforce_eager=enforce_eager,
    )
    safety = round(usable * FRAGMENTATION_FRACTION)

    # Recurrent state is a per-sequence reservation rather than a paged allocation, so it
    # comes out of the budget before any KV sizing.
    recurrent_state = model.cache.fixed_bytes_per_sequence(kv_dtype, tp) * max_num_seqs

    # Only torch-side overheads are subtracted here. `vram_usable_bytes` already means
    # "what PyTorch can allocate", i.e. total minus the driver/vGPU reserve, so counting the
    # non-torch terms again would double-subtract them. Measured on an A10-24Q, that reserve
    # is 2.349 GiB of a 23.722 GiB card -- far too large to handle loosely.
    committed = weights + overheads.torch_side + safety + recurrent_state
    kv_budget = usable - committed

    warnings: list[str] = list(model.warnings)

    if kv_budget <= 0:
        detail = (
            f"weights {fmt_bytes(weights)} + torch overheads "
            f"{fmt_bytes(overheads.torch_side)} + "
            f"safety {fmt_bytes(safety)}"
        )
        if recurrent_state:
            detail += f" + recurrent state {fmt_bytes(recurrent_state)}"
        raise InfeasibleConfigurationError(
            f"no KV cache fits on {gpu.name}: {detail} already exceeds usable "
            f"{fmt_bytes(usable)}. Shard across GPUs, quantize weights, or lower "
            f"max_num_seqs/max_num_batched_tokens."
        )

    per_token = _cache_bytes_per_token(model.cache, kv_dtype, tp)
    kv_tokens = kv_budget // per_token

    # The budget available for *all* cache state is the KV budget plus the recurrent state
    # already committed for max_num_seqs, since raising concurrency re-spends both.
    headroom = max_concurrency_for_budget(
        model.cache,
        workload,
        kv_budget + recurrent_state,
        kv_dtype=kv_dtype,
        tensor_parallel_size=tp,
        samples=min(samples, 4000),
        seed=seed,
    )
    try:
        critical: float | None = gpu.critical_batch_size(model.weight_dtype)
    except ValueError:
        critical = None

    binding, extra_warnings = _classify_binding_constraint(
        model=model,
        workload=workload,
        kv_tokens=kv_tokens,
        cache_wall=headroom,
        critical_batch_size=critical,
        max_num_seqs=max_num_seqs,
        max_model_len=max_model_len,
        kv_dtype=kv_dtype,
        weights=weights,
        usable_bytes=usable,
        samples=samples,
        seed=seed,
    )
    warnings.extend(extra_warnings)

    ledger = _build_ledger(
        gpu=gpu,
        model=model,
        weights=weights,
        overheads=overheads,
        safety=safety,
        recurrent_state=recurrent_state,
        kv_budget=kv_budget,
        max_num_seqs=max_num_seqs,
        token_budget=token_budget,
        tp=tp,
    )

    plan = ResourcePlan(
        weight_bytes_per_gpu=weights,
        activation_peak_bytes=overheads.prefill_activation_bytes + overheads.logits_bytes,
        fixed_overhead_bytes=(
            overheads.cuda_graph_bytes + overheads.attention_workspace_bytes + recurrent_state
        ),
        safety_bytes=safety,
        kv_budget_bytes=kv_budget,
        kv_bytes_per_token=per_token,
        parallelism=parallelism,
        dtypes=DtypePlan(
            weights=model.weight_dtype, activations=activation_dtype, kv_cache=kv_dtype
        ),
        binding_constraint=binding,
        ledger=tuple(ledger),
        warnings=tuple(warnings),
        headroom_concurrency=headroom,
        critical_batch_size=critical,
    )

    if not plan.reconciles_with(usable):
        raise AssertionError(  # pragma: no cover - guards an internal invariant
            f"ledger does not reconcile: terms sum to {plan.total_allocated_bytes} but "
            f"usable VRAM is {usable}"
        )
    return plan


def _classify_binding_constraint(
    *,
    model: ModelProfile,
    workload: WorkloadProfile,
    kv_tokens: int,
    cache_wall: int,
    critical_batch_size: float | None,
    max_num_seqs: int,
    max_model_len: int,
    kv_dtype: DType,
    weights: int,
    usable_bytes: int,
    samples: int,
    seed: int,
) -> tuple[BindingConstraint, list[str]]:
    """Decide what is actually limiting the configuration, and say why."""
    warnings: list[str] = []

    if weights >= usable_bytes:
        return BindingConstraint.WEIGHTS_DO_NOT_FIT, warnings

    if model.cache.fixed_bytes_per_sequence(kv_dtype) > 0:
        warnings.append(
            "this model holds fixed state per sequence (Mamba/SSM), so that share of memory "
            "is a genuine reservation scaling directly with max_num_seqs rather than an "
            "admission limit over a paged cache"
        )

    if not model.cache.scales_with_context:
        warnings.append(
            "cache does not grow with context length at all, so max_model_len has no memory "
            "cost and max_num_seqs is the only memory-relevant concurrency control"
        )
        return BindingConstraint.USER_LIMIT, warnings

    working = workload.working_set(max_num_seqs, samples=samples, seed=seed)
    if working.naive_overstatement > 1.2:
        warnings.append(
            f"composing per-request p95 lengths would have read "
            f"{working.naive_p95_tokens:,} tokens against an actual aggregate p95 of "
            f"{working.p95_tokens:,} ({working.naive_overstatement:.2f}x overstated)"
        )

    if working.p95_tokens > kv_tokens:
        return BindingConstraint.KV_WORKING_SET, warnings

    # The requested concurrency fits, so the binding constraint is the *nearest wall above*
    # it: the cache (how far concurrency can rise before the working set overflows) or the
    # GPU (the critical batch size, past which concurrency buys latency, not throughput).
    #
    # Deliberately NOT `kv_tokens < max_num_seqs * max_model_len`. That is the worst-case
    # reservation reading this project exists to reject (docs/plan.md 2.1): it treats
    # max_num_seqs as if every slot held a full-length context, which essentially never
    # happens, and it would label almost every healthy configuration KV-bound.
    critical = critical_batch_size if critical_batch_size is not None else float("inf")

    if cache_wall <= max_num_seqs:
        return BindingConstraint.KV_WORKING_SET, warnings

    if max_num_seqs >= critical:
        warnings.append(
            f"concurrency {max_num_seqs} is at or above the critical batch size "
            f"({critical:.0f}); further concurrency adds latency rather than throughput"
        )
        return BindingConstraint.COMPUTE_BOUND, warnings

    if max_model_len < model.max_position_embeddings and kv_tokens >= max_num_seqs * max_model_len:
        warnings.append(
            f"max_model_len {max_model_len} is below the model's trained context "
            f"{model.max_position_embeddings}, and the cache could hold every sequence at "
            "full length; raising it costs nothing until the working set grows"
        )
        return BindingConstraint.MAX_MODEL_LEN, warnings

    # Neither wall has been reached. Report whichever will be hit first, since that is what
    # actually caps this deployment's throughput.
    if cache_wall < critical:
        return BindingConstraint.KV_WORKING_SET, warnings
    return BindingConstraint.COMPUTE_BOUND, warnings


def _build_ledger(
    *,
    gpu: GPUProfile,
    model: ModelProfile,
    weights: int,
    overheads: EngineOverheads,
    safety: int,
    recurrent_state: int,
    kv_budget: int,
    max_num_seqs: int,
    token_budget: int,
    tp: int,
) -> list[LedgerEntry]:
    entries = [
        LedgerEntry(
            label="total VRAM",
            bytes_=gpu.vram_bytes,
            formula="device capacity (binary units)",
            provenance=gpu.source,
            is_available=True,
        ),
        LedgerEntry(
            label="usable after driver",
            bytes_=gpu.vram_usable_bytes,
            formula="total - driver/display reservation",
            provenance=gpu.source,
            is_available=True,
        ),
        LedgerEntry(
            label="driver/vGPU reserve",
            bytes_=max(0, gpu.vram_bytes - gpu.vram_usable_bytes),
            formula="total - torch-allocatable (outside the engine's budget)",
            provenance=gpu.source,
            is_available=True,
        ),
        LedgerEntry(
            label="model weights",
            bytes_=weights,
            formula=f"measured checkpoint bytes / tp={tp}",
            provenance=model.weight_bytes_source,
        ),
    ]
    if overheads.cuda_graph_bytes:
        entries.append(
            LedgerEntry(
                label="CUDA graph pool",
                bytes_=overheads.cuda_graph_bytes,
                formula=f"{fmt_bytes(CUDA_GRAPH_BASE_BYTES)} base + {max_num_seqs} captured "
                f"sizes x hidden/tp x {CUDA_GRAPH_BYTES_PER_SEQ_PER_HIDDEN} B",
                provenance="prior",
            )
        )
    entries.extend(
        [
            LedgerEntry(
                label="attention workspace",
                bytes_=overheads.attention_workspace_bytes,
                formula="attention backend scratch",
                provenance="prior",
            ),
            LedgerEntry(
                label="prefill activations",
                bytes_=overheads.prefill_activation_bytes,
                formula=f"{token_budget} tokens x (hidden+intermediate)/tp x "
                f"{ACTIVATION_ELEMENTS} elements",
                provenance="derived",
            ),
            LedgerEntry(
                label="logits + sampling",
                bytes_=overheads.logits_bytes,
                formula=f"{max_num_seqs} seqs x {model.vocab_size} vocab x 4 B x "
                f"{LOGITS_BUFFER_COPIES}",
                provenance="derived",
            ),
        ]
    )
    if recurrent_state:
        entries.append(
            LedgerEntry(
                label="recurrent state",
                bytes_=recurrent_state,
                formula=f"{max_num_seqs} seqs x fixed Mamba state per sequence",
                provenance="derived",
            )
        )
    entries.extend(
        [
            LedgerEntry(
                label="fragmentation safety",
                bytes_=safety,
                formula=f"{FRAGMENTATION_FRACTION:.0%} of usable VRAM",
                provenance="prior",
            ),
            LedgerEntry(
                label="KV cache budget",
                bytes_=kv_budget,
                formula="usable - everything above",
                provenance="derived",
            ),
        ]
    )
    return entries


def with_kv_dtype(plan: ResourcePlan, cache: CacheSpec, kv_dtype: DType) -> ResourcePlan:
    """Re-derive token capacity for a different KV dtype, keeping the byte budget fixed.

    Useful for presenting mitigations: the budget does not change, only how many tokens fit
    inside it.
    """
    per_token = _cache_bytes_per_token(cache, kv_dtype, plan.parallelism.tensor)
    return replace(
        plan,
        kv_bytes_per_token=per_token,
        dtypes=DtypePlan(
            weights=plan.dtypes.weights,
            activations=plan.dtypes.activations,
            kv_cache=kv_dtype,
        ),
    )


def max_concurrency_for_budget(
    cache: CacheSpec,
    workload: WorkloadProfile,
    cache_budget_bytes: int,
    *,
    kv_dtype: DType,
    tensor_parallel_size: int = 1,
    quantile: float = 0.95,
    ceiling: int = 4096,
    samples: int = 4000,
    seed: int = 0,
) -> int:
    """Largest concurrency whose total cache footprint fits ``cache_budget_bytes``.

    ``cache_budget_bytes`` must cover **both** components of cache cost:

    * the paged, per-token KV cache, and
    * any fixed per-sequence state (Mamba/SSM), which scales with concurrency directly.

    Omitting the second term badly overestimates headroom for hybrid models: Nemotron-H-8B
    holds ~50 MiB of recurrent state *per sequence*, so 236 concurrent requests would need
    11.5 GiB of state on top of their KV — memory that a KV-only calculation never asks for.

    Binary search over a monotone predicate, so this costs O(log n) Monte-Carlo evaluations
    rather than a linear scan.
    """
    if cache_budget_bytes < 0:
        raise ValueError("cache_budget_bytes must be >= 0")
    per_token = _cache_bytes_per_token(cache, kv_dtype, tensor_parallel_size)
    per_sequence = cache.fixed_bytes_per_sequence(kv_dtype, tensor_parallel_size)

    def fits(concurrency: int) -> bool:
        working = workload.working_set(concurrency, samples=samples, seed=seed)
        tokens = (
            working.p99_tokens
            if quantile >= 0.99
            else working.p95_tokens
            if quantile >= 0.95
            else working.p50_tokens
        )
        needed = concurrency * per_sequence + tokens * per_token
        return needed <= cache_budget_bytes

    if cache_budget_bytes <= 0 or not fits(1):
        return 0
    low, high = 1, 2
    while high <= ceiling and fits(high):
        low, high = high, high * 2
    high = min(high, ceiling)
    while low + 1 < high:
        mid = (low + high) // 2
        if fits(mid):
            low = mid
        else:
            high = mid
    return low


def recurrent_reservation_bytes(cache: CacheSpec, max_num_seqs: int, kv_dtype: DType) -> int:
    """Total fixed recurrent state for ``max_num_seqs`` sequences.

    Exposed separately because for Mamba-hybrid models this term, not the KV cache,
    dominates and behaves completely differently: it is a hard reservation.
    """
    if isinstance(cache, RecurrentSpec) or cache.fixed_bytes_per_sequence(kv_dtype) > 0:
        return cache.fixed_bytes_per_sequence(kv_dtype) * max_num_seqs
    return 0


__all__ = [
    "CUDA_GRAPH_BASE_BYTES",
    "FRAGMENTATION_FRACTION",
    "GIB",
    "EngineOverheads",
    "InfeasibleConfigurationError",
    "estimate_overheads",
    "estimate_plan",
    "max_concurrency_for_budget",
    "recurrent_reservation_bytes",
    "with_kv_dtype",
]


def safe_utilization_ceiling(
    total_bytes: int,
    non_torch_bytes: int,
    *,
    safety_fraction: float = FRAGMENTATION_FRACTION,
) -> float:
    """Highest memory-utilisation fraction an engine can be given safely.

    Engines express the knob as a fraction of **total** device memory, but only
    ``total - non_torch`` is actually allocatable. So the ceiling is not 1.0:

        ceiling = (total - non_torch) / total - safety

    Measured on an Azure A10-24Q, where 2.35 GiB is consumed by the vGPU stack before any
    allocation: ``(23.722 - 2.349) / 23.722 = 0.901``. Asking for 0.90 leaves ~20 MiB and
    OOMs during sampler warm-up; the observed safe range was <= 0.85. On bare metal, where
    non-torch overhead is a few hundred MiB, the same formula yields ~0.97.

    Prefer a measured ``non_torch_bytes`` from ``torch.cuda.mem_get_info`` over a prior.
    """
    if total_bytes <= 0:
        raise ValueError("total_bytes must be > 0")
    if not 0 <= non_torch_bytes < total_bytes:
        raise ValueError("non_torch_bytes must be in [0, total_bytes)")
    if not 0 <= safety_fraction < 1:
        raise ValueError("safety_fraction must be in [0, 1)")
    return max(0.05, (total_bytes - non_torch_bytes) / total_bytes - safety_fraction)


def predict_kv_for_utilization(
    plan: ResourcePlan,
    overheads: EngineOverheads,
    utilization: float,
    total_bytes: int,
) -> int:
    """Predict the KV cache an engine will allocate at a given utilisation fraction.

    Models the engine's actual arithmetic rather than our own ledger split:

        kv = utilization * total_device_memory - (weights + torch-side overheads)

    Verified against vLLM 0.19.1 on an A10 across four utilisation values, which fit
    ``kv = gmu * 23.722 GiB - 1.424 GiB`` with a slope equal to ``torch.total_memory`` to
    three significant figures. This is the quantity the M2 criterion scores, and it is
    obtainable from a single boot with no load generation.
    """
    if total_bytes <= 0:
        raise ValueError("total_bytes must be > 0")
    if not 0 < utilization <= 1:
        raise ValueError(f"utilization must be in (0, 1], got {utilization}")
    budget = round(utilization * total_bytes)
    return max(0, budget - plan.weight_bytes_per_gpu - overheads.torch_side)
