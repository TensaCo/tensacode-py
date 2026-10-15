# Algorithm substrates and robotics — 2026-10-15

Owner requested the framework and a simulated robotics policy before 20:00 UTC,
prioritizing ALG.0 -> ALG.2 and conversions between authored representations.
General algorithm discovery from arbitrary pretrained transformers was explicitly
deprioritized. Work began at 19:13 UTC.

## Implemented

- Versioned, data-only, statically shaped tensor algorithm IR, sparse operand-role
  and next-state topology, explicit parameters and transactional simultaneous state.
- AST source frontend and explicit builder; composition and standalone NumPy export.
- ALG.0 NumPy reference; ALG.1 native PyTorch circuit; ALG.2 actual scaled-dot-product
  attention operand reads with supplied local operations and execution schedule.
- Hard-routing conversions export current trained weights. Finite soft routing is
  explicitly approximate and cannot masquerade as exact symbolic conversion.
- Contextual structural-bias compilation: role-specific query/key bindings, null
  slot, Pq A Pk^T, zero-initialized head/role strengths, complete data-only persistence.
- Native pretrained MHA and standard Q/K/V/O projection factories, preserving
  copied weights and explicit masks. No automatic model patching or hidden ontology.
- Exact structural import of supported native feed-forward and MHA computation;
  shared feed-forward parameters retained. This is not semantic algorithm discovery.
- Explicit recurrent unrolling and gradients, empirical output/state replacement
  validation, shape/finite/budget checks, and failure counterexamples.
- TypeScript artifact execution, native-autograd circuits, hard/soft attention,
  trained parameter export, and Python/TypeScript numerical interoperability.

## Evidence

- Python: 88 tests passed across algorithm, existing program and training suites.
- TypeScript: final full suite passed 1,564 tests across 113 files, including
  dtype conversion and persistence; typecheck and build pass.
- Source formatting and lint on the new Python implementation and examples pass.
- Site documentation build and internal link check pass: 1,401 internal links
  across 30 pages. All five Python/JavaScript site demo comparisons pass.
- `examples/robotics_policy.py`: completed 20-joint MuJoCo rollouts in all three
  substrates, synthetic actuator-residual training through ALG.2, export/reload,
  frozen-controller evaluation on a different trajectory and 25% stronger disturbance,
  rendered replay and source-hashed provenance. Committed artifacts live in
  `docs/results/algorithm-robotics/`.
- Main joint RMS tracking error: baseline 0.01039305595 rad, corrected
  0.00057636997 rad; ALG.0/1/2 agree. Shifted case: baseline 0.01192848846 rad,
  corrected 0.00251780939 rad. Supervision is an authored synthetic calibration
  law; no online system identification or RL is claimed.
- Pinned cached pretrained CLIP vision attention: 50 image tokens, zero-bias
  max output error 0.0, nonzero relation-strength gradient, nonzero bias changes
  output. This establishes integration and gradient plumbing, not semantic binding
  or execution of the robotics algorithm inside CLIP.

## Boundaries

This is a general tensor algorithm framework with robotics in an example, not in
core schemas. ALG.2 hard mode uses supplied local arithmetic and scheduling; it
is not a vanilla transformer synthesized for arbitrary Python. ALG.2 bias mode
only supplies an inductive prior; its binding networks are initially untrained.
There is no demonstrated learned problem identification, universal neural
compilation, arbitrary-pretrained-model decompilation, continuous-domain proof,
unbounded execution, general diffusion substrate or learned locomotion.

The robot is a fixed-base four-limb, twenty-hinge mechanism. The controller is
an authored impedance law using model-based bias-force compensation. Learned
coefficients correct the supplied synthetic actuator disturbance. Some observed
ALG.2 iterations exceed the 5 ms simulation step: the example advances physics
offline and makes no hard-real-time or hardware-safety claim.

Next research: train/validate bindings and relation use on a task with held-out
identities and compositions, comparing equal-information ordinary attention and
bias removal/permutation controls. Do not infer semantic grounding from the
zero-gate or gradient tests.

## Annotated results presentation

`docs/results/algorithm-robotics/index.html` is a portable report with two embedded
side-by-side MP4 comparisons, error traces, controller and substrate diagrams,
original metrics and explicit baseline limitations. The generator
`examples/build_robotics_report.py` replays the committed frozen policy without
retraining and rejects mismatches in RMS, worst joint error or peak torque against
the original results. Replay trajectories and metrics are retained in
`presentation-replay.json`; reported compute timings remain the original unrendered
measurements. The report distinguishes residual-learning gains from substrate
fidelity and pretrained-adapter integration, for which no task gain was measured.

Presentation validation: Chrome loaded both embedded 1280×800, six-second videos;
playback, quarter-speed control and JSON download passed. A 390-pixel mobile
viewport had no document overflow, and no JavaScript errors were observed.
The report generator passes scoped Ruff and Python compilation checks.
