# IEEE TC Formal Experiment Snapshot

This worktree is the clean, auditable source for IEEE Transactions on
Computers experiments. Do not run formal claims from the exploratory
`/home/kec23008/DynaQuant` worktree.

## Model contract

- Qwen3-30B: FP16 high tier and INT4 low tier.
- Phi-3.5-MoE: FP16 high tier and INT4 low tier.
- Qwen3-Next-80B: INT4 high tier and INT2 low tier.
- Every reported inference run uses one physical RTX A6000. A second GPU may
  run an independent job, but one model is never tensor-parallelized across
  both devices.

The DynaExQ dynamic Qwen3-Next source is
`/dev/shm/dynaexq-models/qwen3-next-80b-bf16`, pinned by
`results/model_manifests/qwen3_next_80b_a3b_bf16.json`. The `/dev/shm`
snapshot is volatile and must be recreated from the manifest's immutable
revision after a reboot. Dynamic expert packing requires the original
three-dimensional expert tensors and therefore does not load the static
AutoRound QuantLinear checkpoint.

After any download or resume, verify all snapshot bytes before a formal run:

```bash
python scripts/verify_model_manifest.py \
  --manifest results/model_manifests/qwen3_next_80b_a3b_bf16.json \
  --output results/model_manifests/qwen3_next_80b_a3b_bf16.verification.json
```

## Static Qwen3-Next checkpoints

Fetch and content-register Intel's official mixed-AutoRound INT4 checkpoint,
then derive INT2 from that exact parent:

```bash
bash scripts/quantize_qwen3_next_static.sh int4
bash scripts/quantize_qwen3_next_static.sh int2
```

The parent is
`Intel/Qwen3-Next-80B-A3B-Instruct-int4-mixed-AutoRound`: W4/group-128
experts, W8 non-expert projections, and FP16 gates, exported as
`auto_round:auto_gptq` by AutoRound 0.12.0 using zero-iteration RTN. ModelScope
exposes a moving `master` reference, so registration hashes its complete
remote path/size/SHA-256 catalog into a stable content-set revision and then
verifies every local file.

The INT2 converter reconstructs the parent's W4 values, applies symmetric
W2/group-64 RTN, and repacks them in the same AutoGPTQ layout. W8 and FP16
override tensors are preserved. It is a calibration-free requantization from
INT4, not a direct BF16-to-INT2 AutoRound run; the provenance and paper must
disclose the possibility of compounded quantization error.

Static paper-protocol evaluation must pass `--autoround-backend triton`.
Remote model code and automatic kernel selection are disabled.

## Shared-GPU acceptance

Idle foreign HBM residency is allowed and recorded separately. A measured
performance sample is invalid if the 2 ms NVML monitor observes any nonzero
foreign-process SM or memory utilization. Re-run invalid samples; never kill
another user's process.

## Clean-tree sequence

1. Commit code, manuscript, manifests, and calibration inputs.
2. Confirm `git status --porcelain` is empty.
3. Run `python -m pytest -q`.
4. Build a calibration ranking for each dynamic model from the clean commit.
5. Run quality, performance, ablation, sensitivity, and mechanism artifacts.
6. Register every claim in `results/paper/manifest.json`.
7. Render figures and run both audits:

```bash
bash scripts/reproduce_paper.sh audit
bash scripts/reproduce_paper.sh paper
bash scripts/reproduce_paper.sh submission-audit
```

The manuscript remains an internal draft until both audits pass.
