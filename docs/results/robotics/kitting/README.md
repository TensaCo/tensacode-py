# Compositional kitting evidence

[Open the annotated report](evidence.html) · [Runnable example](../../../../examples/robotics/kitting/README.md)

The workcell has five actuators (Cartesian XYZ plus two jaws), four free parts,
three order slots and a quarantine area. There are no weld constraints, mocap
attachments or object pose changes during execution. MuJoCo integrates contact
and friction; native position servos and authored waypoint stages move the robot.

The compiled TensorCode policy handles prerequisite readiness and inventory
matching. Dispatch, retry limits, quality inputs and completion observations are
explicit host logic. This is not a learned VLA, diffusion policy, inferred quality
model, production robot model or proof of task novelty.

## Measurements

Six seeds (100–105), seven scenarios and six controllers produce 252 episodes.
ALG.0/1/2 and native symbolic feedback each complete 42/42 orders. Fixed recipe
completes 6/42; a no-retry ablation completes 18/42. These ablations deliberately
remove rules; they do not represent state-of-the-art learned robotics systems.
All 2,100 recorded decision evaluations match the reference circuit exactly.
The nominal and six exception variants share the same motion stages. Initial
part XY positions vary ±12 mm and masses vary uniformly from 65 to 95 g.

Seed 0 was used for development. The scenarios and acceptance gate were written
before seeds 100–105 ran. An audit then tightened the placement criterion to
require parts on the work surface (z > 0.40 m), rather than merely below the
placement ceiling. The initial protocol/results/source remain in
`acceptance-audit.zip`; the same workloads were rerun to verify that correction.
This is a bounded evaluation with an acceptance audit, not an untouched final
customer qualification. No controller or motion parameters were tuned on these
seeds. Physical states and success outcomes remained unchanged.

Quality and identity flags are supplied, and observations use simulator poses.
The order revision occurs after the first inspection, before physical transfers.
Fault injection offsets the first kit grasp by 8 cm. A failed grasp is detected
from object lift; a placement requires XY proximity and height on the table.
Retries are bounded at two per episode. A whole-order audit independently checks
matching part identity, quality, slot occupancy, quarantine and logical release.
“Seal” is logical release of the order, not physical sealing of a carton.

## Reproduce

From the Python repository root, install `.[robotics]` and use CPU-only execution:

```sh
python examples/robotics/kitting/evaluate.py \
  --seeds 100 101 102 103 104 105 --output /tmp/kitting-evaluation
python examples/robotics/kitting/validate.py \
  --results docs/results/robotics/kitting
```

Use the existing budget runner to cap resources and monitor global headroom:

```sh
python examples/robotics/humanoid/humanoid_budget.py \
  --output /tmp/kitting-budget -- \
  "$(pwd)/.venv/bin/python" examples/robotics/kitting/evaluate.py \
  --output /tmp/kitting-evaluation
```

Rendering on this host used Xvfb and software GL because the first EGL attempt
failed in the graphics driver. It did not affect evaluation. Xvfb, an available
Mesa GL driver, DejaVu fonts and ImageIO's ffmpeg backend are rendering dependencies.

```sh
python examples/robotics/humanoid/humanoid_budget.py \
  --output /tmp/kitting-render-budget -- \
  xvfb-run -a env MUJOCO_GL=glfw __GLX_VENDOR_LIBRARY_NAME=mesa \
  "$(pwd)/.venv/bin/python" examples/robotics/kitting/render.py \
  --results /tmp/kitting-evaluation
python examples/robotics/kitting/build_report.py
```

The HTML builder reads this committed evidence directory. The video replays the
first evaluation seed's combined scenario, selected in advance. Both sides share
simulated time; the shorter baseline is held at its recorded endpoint.

## Files

- `protocol.json`: seeds, scenarios, acceptance, scope and frozen source hashes.
- `report.json`: all episode receipts, final poses, events, timings and summaries.
- `policy.json` / `policy.py`: portable circuit and standalone NumPy code export.
- `alg2-*` / `fixed_recipe-*`: decision traces for the preselected comparison.
- `alg2.npz` / `fixed_recipe.npz`: 25 Hz physical state recordings, times and labels.
- `alg2.xml` / `fixed_recipe.xml`: complete reproducible scenes, with no external assets.
- `execution-source.zip`: exact source corresponding to the final protocol hashes.
- `evaluation-resources.json` / `render-resources.json`: resource limits and global samples.
- `comparison.mp4`, `poster.png`, `render.json`: annotated replay and metadata.
- `validation.json`: independent receipt, physical state, source and artifact checks.
- `bundle-sha256.json`: file integrity manifest, excluding itself.

The value hypothesis is lower work-order changeover effort and fewer manual
interventions. No labour savings, throughput improvement, customer deployment,
hardware reliability or financial return has been measured. Actual VLA/diffusion
baselines, noisy perception and real inventory/order distributions are required
before making competitive or commercial claims.

## Report relocation

The original `index.html` is preserved byte-for-byte as `evidence.html`; its
SHA-256 remains unchanged in `bundle-sha256.json`. The former URL now redirects
to the [humanoid showcase](../index.html). This navigation change does not alter
the recorded measurements, archived execution source or acceptance audit.
