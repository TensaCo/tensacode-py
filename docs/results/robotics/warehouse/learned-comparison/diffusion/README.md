# Compact diffusion evidence

These are small from-scratch diffusion policies trained on the same sensor/action
contract and declared expert splits as the recurrent BC comparison. They are not
pretrained VLAs or a test of state-of-the-art diffusion policies.

`model-index.json` identifies each checkpoint, protocol, immutable dataset manifest,
and selected validation epoch. Original primary and success-only models remain
preserved, including their failed development rollouts. The original width 256
projection loses 64 of 320 Gaussian action-noise directions; `audits/` records the
rank check and exact action replay. This limitation does not establish the cause
of every failure. The width 512 correction has a separate preregistered protocol.

`physical-diagnostics-index.json` links independent scores, checkpoint provenance
and raw trace hashes. These nominal/validation development tests are **not final
held-out results**. There is no cross-task comparison with GR00T or other pretrained
models. Original physical diagnostics precede the later shared arm-slew cap; use
matching control revisions for any comparison.

Portable CPU inference (requires compatible PyTorch and NumPy):

```bash
python verify.py --variant original-primary --inference
```

This checks every packaged hash, extracts the archived model source, loads the
weights and configuration, and verifies seeded local reset behavior on the included
sensor observation. No raw training dataset or simulator is needed for this check.
`training.json` records the tested Torch version. Timing JSON separates full
32-step denoising plans from cached commands; it excludes sensing and simulation
and is not a worst-case latency guarantee.

**Retraining and physical replay are not self-contained from Git alone.** The raw
expert episodes and full physical traces remain in immutable local storage.
`dataset/local-data-index.json` and `physical-diagnostics-index.json` give their
original absolute paths and SHA256 hashes. Copied manifests retain their original
bytes; resolve their relative episode paths against the original local directory,
not this documentation folder. Audit scripts require those local traces and the
matching extracted evaluation source. All negative results remain in the indexes.
