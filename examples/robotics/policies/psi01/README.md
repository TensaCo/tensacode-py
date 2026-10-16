# Ψ₀.1 recorded policy evaluation

This adapter validates and indexes existing native G1/Dex3 policy rollouts from
`~/Documents/psi-0.1`. It does not substitute scripted actions for learned output
or rerun the recorded experiments. The original project prohibits unnecessary
reproduction of its own recorded runs.

```bash
python examples/robotics/policies/psi01/audit.py
```

The evidence bundle is `docs/results/robotics/baselines/psi01/`. The audit checks
all 80 episode identities, the original protocol hash, checkpoint prefixes,
video/outcome pairing, and every bundled file hash. `--write` regenerates the
index after an intentional evidence update; never use it to conceal a mismatch.

This is a recorded-evidence integration, not a new TensorCode policy runtime.
Use the original project's `python -m psi01.evaluate` and resource broker for any
justified new native SIMPLE evaluation. No weights or private executable code
are copied into TensorCode.
