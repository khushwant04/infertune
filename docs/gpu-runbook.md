# GPU runbook: the next validation session

Everything that can be built without a GPU is built. What remains in [`status.md`](status.md) is measurement, and this runbook does all of it in one session on a 2×A10 host. Follow the steps in order: each one tells you whether the next is worth paying for.

Budget about three hours on `Standard_NV72ads_A10_v5`. `--spot` makes it roughly 5× cheaper, but an eviction mid-sweep costs a boot.

## 0. Provision

```bash
./scripts/provision_azure_gpu.sh --gpus 2          # add --spot to save money
```

The script installs the only driver this SKU supports (570.237). That caps the stack at `torch 2.10.0+cu128` and `vLLM 0.19.1`. [`gpu-validation-a10.md`](gpu-validation-a10.md) explains why each pin is load-bearing.

On the VM:

```bash
git clone https://github.com/khushwant04/infertune.git && cd infertune
uv venv && source .venv/bin/activate
uv pip install "vllm==0.19.1" --torch-backend=cu128
uv pip install -e ".[nvml,dev]"
nvidia-smi && infertune version
```

> [!IMPORTANT]
> Use **one GPU identity** for the whole session. Measurements are keyed by the GPU's name, and NVML reports the vGPU profile (`NVIDIA A10-24Q`), not the board name `NVIDIA A10` that `--gpu a10` gives. Calibration only pools runs with identical keys. Either always pass `--gpu a10`, or never pass it.

## 1. Re-check the memory ledger (10 min)

This confirms the box matches the one M2 was validated on before any new claim rests on it.

```bash
python scripts/validate_vllm.py --gpu a10 --models Qwen/Qwen3-0.6B Qwen/Qwen3-8B
```

**Pass:** every KV error within ±5%, as in [the M2 table](gpu-validation-a10.md#m2-acceptance-measurement). If this fails, stop: the driver or vLLM is not the validated stack.

## 2. `tune --execute` end to end at tp=1 (30 min)

The first real run of the engine lifecycle.

```bash
infertune tune --model Qwen/Qwen3-8B --gpu a10 --gpus 1 --boots 5 --execute \
  --store session.db --log-dir runs/tp1
```

Check:

- every boot ends as `measured` or `failed`, and each failure names a real cause (read `runs/tp1/*/NN-*.log`);
- `nvidia-smi` shows no leftover processes after the command exits;
- the **KV pred/engine** column agrees within ±5%;
- the measured winner matches the dry-run ranking's top candidate, or the gap is explained by the prediction error printed under the winner.

That last check is item 6 of the remaining work.

## 3. Held-out throughput: the M3 criterion (60 min)

The criterion: predicted throughput within **±20%** on configurations that were **not** used for calibration.

```bash
# Calibration set: one model, several boots.
infertune tune --model Qwen/Qwen3-4B --gpu a10 --gpus 1 --boots 6 --execute \
  --store session.db --log-dir runs/calib

# Held-out set: a different model, scored with the coefficients fitted above.
infertune tune --model Qwen/Qwen3-8B --gpu a10 --gpus 1 --boots 4 --execute \
  --store session.db --log-dir runs/heldout
```

The held-out run prints `calibrated tp=1 from session.db` and then, per candidate, `pred tok/s` against measured `tok/s`. Record the worst relative error.

**Pass:** at most 20% on every held-out candidate.

> [!NOTE]
> Calibration is fitted per (GPU, model, engine version, TP). A model with no stored runs uses the prior coefficients, so the held-out check above measures the prior plus model-independent overheads. To test transfer across models, run step 3 twice and compare the fitted `mbu`/`mfu` in the summary line.

## 4. Tensor parallelism over PCIe (40 min)

NVads A10 v5 has no NVLink, which makes this the sharpest available test of the roofline's per-call communication term.

```bash
nvidia-smi topo -m                                   # first: is P2P disabled under vGPU?
infertune tune --model Qwen/Qwen3-8B --gpu a10 --gpus 2 --boots 6 --execute \
  --store session.db --log-dir runs/tp2
```

Record the topology output alongside the results. If vGPU has disabled P2P, the measurement still stands, but it tests the latency term under a pessimistic interconnect.

**Check:** tp=2 candidates boot, and their KV error stays within ±5%. Compare the measured tp=1 and tp=2 throughput with the prediction of when TP stops helping.

## 5. SGLang against a real install (30 min)

Use a separate virtualenv: SGLang and vLLM pin different torch builds.

```bash
uv venv .venv-sglang && source .venv-sglang/bin/activate
uv pip install "sglang[all]" --torch-backend=cu128   # pick a release whose wheels link libcudart.so.12
uv pip install -e ".[nvml]"
infertune tune --model Qwen/Qwen3-0.6B --gpu a10 --gpus 1 --engine sglang --boots 3 --execute \
  --store session.db --log-dir runs/sglang
```

Then diff the introspected schema against `tests/fixtures/sglang/server_args_doc_derived.json`. Any flag the adapter emits that the installed SGLang lacks shows up as an `error:` diagnostic before boot.

## 6. Bring the evidence home

```bash
python -c "from infertune.store import MeasurementStore; print(MeasurementStore('session.db').to_json())" > session.json
tar czf session-$(date +%F).tgz session.db session.json runs/
```

Copy the archive off the VM, then **delete the resource group**. Quota and billing continue until you do.

Update [`status.md`](status.md): the M3 row, the SGLang and tensor-parallelism paragraphs under "Known gaps", and the `tune --execute` line. Add a dated results section to [`gpu-validation-a10.md`](gpu-validation-a10.md).

## Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| Every boot fails in seconds with `CUDA out of memory` | Leftover engine from an interrupted run | `nvidia-smi`, kill the PID, rerun |
| `engine not ready after 900s` | Cold download of a large checkpoint | Pre-download with `huggingface-cli download`, or pass `--boot-timeout 1800` |
| `--kv-cache-dtype` rejected | fp8 on sm_86 | Expected on A10: fp8 candidates are not enumerated on sm_86; check `--gpu` |
| `cannot execute: vllm is not usable here` | The Python running `infertune` cannot import vLLM | Install InferTune into the engine's virtualenv: it reads the flag schema from the engine it can import, and launches that same engine |
| Calibration line never appears | GPU name or engine version differs between runs | Keep `--gpu` consistent (see step 0) |
