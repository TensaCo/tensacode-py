# Compact diffusion comparison

This is a small **from-scratch conditional DDPM baseline**, not a pretrained VLA,
π0, GR00T, or a claim about state-of-the-art diffusion performance. It predicts
16 commands from eight sensor observations and executes four before replanning.
Its 20 output coordinates and the sensor allowlist are the recurrent BC contract.
`DiffusionController` inherits the BC controller and changes only its policy runtime;
the walker, arm bounds, servo, dispatch guard and physical success evaluation stay
shared. There is no teacher fallback.

`protocol.json` freezes architecture, optimization, action decoding and checkpoint
selection before access to the recovery expert dataset. The manifest SHA is an
explicit required training argument. Use the **same qualified manifest as BC**;
continuous normalization uses only training frames; explicit mask, probability and
measurement-age channels preserve their units. Windows never cross physical episodes,
and validation selects the lowest fixed-noise denoising loss. Neither final episode
bytes nor final outcomes are used by training. All collected expert episodes,
including failures, belong to their already declared splits.

```bash
python -m examples.robotics.warehouse.diffusion_policy.train \
  --manifest /path/to/qualified/manifest.json \
  --manifest-sha256 EXPECTED_SHA256 \
  --output /path/to/new/diffusion-policy --device cpu --threads 2
```

CUDA training requires the campaign's coordinated exclusive GPU window and resource
guards. This entry point does not reserve hardware or enforce system headroom.
Do not start a competing CUDA job just by changing `--device`.

The checkpoint contains `weights.pt`, `policy.json`, the immutable dataset/protocol
binding, archived source and training history. Load `model.PolicyRuntime(directory)`
for `step(sensor_observation) -> (action_dict, raw_20_values)` and `reset()`.
The deterministic CPU sampler uses one Torch thread and resets its random stream
per physical episode. Fresh sensor frames continue to enter history while a chunk
is executing. Flags are sampled continuous coordinates thresholded at 0.5; grip
uses the shared binary 0/0.028 m decoding. Diffusion time is an internal noise level,
not an observation of simulator time or teacher phase.

Validation epsilon MSE is an offline denoising diagnostic, **not task success**.
Report the same 85-second physical tests and strict current one-second settling
criterion as BC before making any closed-loop comparison. Short observation
history, small MLP capacity, limited expert coverage and command latency are explicit
limitations. A failed rollout stays in the report. No trained policy or physical
success is implied by passing the synthetic unit tests.

The preregistered success-only demonstration ablation uses the shared
`learned_policy/ablation-protocol.json` and curator. Only training episodes are
filtered by independent strict physical success; the original full validation set
still selects the checkpoint. Training archives and hashes the ablation protocol,
retains the primary manifest hash and exclusions in its binding, and reports any
successful-validation subset diagnostic only after checkpoint selection.

Use `python -m examples.robotics.warehouse.learned_policy.latency --kind diffusion`
with the checkpoint, manifest and output arguments for the shared CPU benchmark.
It measures 512 validation steps after 64 warmup steps, separating 32-step diffusion
planning calls from cached command execution. Run both policies with identical
host affinity; observed timings exclude sensing, walker and simulation costs and
are not a worst-case execution guarantee.

## Separately frozen width correction

The original width 256 denoiser first projects all 320 noisy action coordinates into
256 dimensions. That loses at least 64 independent Gaussian noise directions and
imposes an expected epsilon-MSE floor of 0.20 on full unmasked chunks. The recorded
rank/null-space check is `bottleneck-audit.json`. This is an architectural weakness
of this implementation; it does **not** prove the cause of every physical failure
or support a conclusion about diffusion policies generally.

`protocol-wide.json` preregisters a separate width 512 correction after development
diagnostics and before its training or any final evaluation. It changes only the
hidden width, preserves all original negative results/checkpoints, and uses the
same TRAIN/full-validation split and checkpoint-selection rule. Select it explicitly
with `--protocol examples/robotics/warehouse/diffusion_policy/protocol-wide.json`.
Both widths remain loadable through the same recorded architecture configuration.
