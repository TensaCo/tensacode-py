# Unitree G1 walking and physical grasping

The [HTML report](index.html) shows a free-base G1 walking to a pedestal,
physically pinching an 80 g cube with an authored parallel-jaw attachment, lifting
it about 15 cm, holding and releasing it. The walking network and joint servo run
through TensorCode ALG.0, ALG.1 or hard ALG.2. The native ONNX reference also
succeeds. This demonstrates composition and conversion fidelity, not a newly
learned gait or a better walking policy.

## Reproduce

From the Python checkout, use Python 3.13 and the recorded package versions in
`report.json` for the closest reproduction. A CPU-only PyTorch environment is
sufficient. The resource wrapper requires Linux, user systemd and cgroup v2.
Rendering uses Mesa EGL/llvmpipe, DejaVu Sans and imageio-ffmpeg.

```bash
uv venv .venv-humanoid --python 3.13
uv pip install --python .venv-humanoid/bin/python \
  --index-url https://download.pytorch.org/whl/cpu torch==2.14.0
uv pip install --python .venv-humanoid/bin/python -e '.[robotics]' \
  mujoco==3.16.0 numpy==2.5.3 onnx==1.23.2 onnxruntime==1.31.0 \
  scipy==1.18.1 pyyaml imageio imageio-ffmpeg pillow

.venv-humanoid/bin/python examples/robotics/humanoid/fetch_humanoid_assets.py \
  --output /tmp/tensorcode-g1-assets

# Native reference and observations for fidelity checks.
.venv-humanoid/bin/python examples/robotics/humanoid/humanoid_budget.py \
  --output /tmp/g1-results/native-budget -- \
  "$(pwd)/.venv-humanoid/bin/python" examples/robotics/humanoid/humanoid_policy.py \
  --assets /tmp/tensorcode-g1-assets --task --seconds 30 \
  --output /tmp/g1-results/native

# All substrates, open-finger control, and two placement/mass variations.
.venv-humanoid/bin/python examples/robotics/humanoid/humanoid_budget.py \
  --output /tmp/g1-results/evaluation-budget -- \
  "$(pwd)/.venv-humanoid/bin/python" examples/robotics/humanoid/humanoid_evaluate.py \
  --assets /tmp/tensorcode-g1-assets --reference /tmp/g1-results/native \
  --output /tmp/g1-results/evaluation

.venv-humanoid/bin/python examples/robotics/humanoid/humanoid_budget.py \
  --output /tmp/g1-results/render-budget -- \
  "$(pwd)/.venv-humanoid/bin/python" examples/robotics/humanoid/render_humanoid.py \
  --results /tmp/g1-results/evaluation --output /tmp/g1-results/media
```

For a single ALG.2 rollout, add `--substrate alg2` to `humanoid_policy.py`.
The initial native reference is not required for that command. The two movies
are physics-state replays at 25 fps; the main movie uses ALG.2. The side-by-side
open/closed-finger control uses native ONNX on **both** sides to isolate closure.

The HTML builder is `examples/robotics/humanoid/build_humanoid_report.py`; it reads committed
metrics and media from this folder. The earlier actuator-calibration report is
preserved at `../calibration/index.html`; its reported percentage improvements are not
claims about this humanoid experiment.

## What is supplied and what executes

- Unitree supplies the robot geometry, model, actuator settings and pretrained
  98–512–256–128–29 ELU walking network with observation normalization.
- `compile_walker` is an explicit lowering of that pinned architecture, not a
  general ONNX importer or semantic extraction algorithm. No weights are trained.
- Both the network (50 simulated Hz) and 31-joint PD/torque-limit servo
  (500 simulated Hz) execute in the selected TensorCode substrate. Hard ALG.2
  supplies arithmetic and scheduling; this is not execution inside a pretrained
  transformer or `alg2_bias`.
- The task state machine, world-to-body commands and Cartesian Jacobian IK remain
  authored host code. During manipulation, right-arm position/velocity observation
  channels are zeroed for the walking network, whose base and leg feedback remain
  measured. Previous policy actions remain in the observation.
- Object pose and full robot state are supplied by simulation. The reach path is
  clearance-aware but scripted. The shifted cases provide their changed pose to
  the same task planner.
- The 29 body joints remain articulated. Two simulated sliding fingers are added
  to the right wrist; this is not the stock Unitree dexterous hand. Robot
  self-collisions are disabled. Environment, foot, finger and object contacts are
  active. The floating base and object have no equality constraints or welds.
- Physics advances through `mj_step`; the object and base are not repositioned
  during rollouts. IK operates on a separate kinematic data copy. The renderer
  loads recorded poses only after the physical rollout is complete.

## Measurements and artifacts

`report.json` contains fidelity checks on 67 recorded observations and seven
rollouts: the native reference, three TensorCode substrates, an open-finger
control, and two modest pose/mass variations. These are deterministic checks,
not independent statistical trials. Success requires walking >0.7 m, pelvis
height >0.65 m throughout, at least one second of bilateral contact while the
cube is >0.10 m above its resting height, and reaching the release stage.

The nominal controller walks 0.949 m, lifts 0.152 m and records 2.94 seconds of
bilateral contact above 0.10 m. ALG.0/1/2 and native all succeed. The open-finger
control has no bilateral contact and no lift during the manipulation window;
its 3 mm maximum in the report is the initial object placement before settling.
Both additional known-pose scenes succeed (60 g and 100 g).

- `*-state.npz`: complete generalized positions at 50 Hz (`qpos`).
- `*-trace.json.gz`: time, stage, base/hand/object positions, contacts, pad forces,
  foot-contact flags, policy observation and action at 50 Hz.
- `walker.json.gz`: frozen TensorCode algorithm artifact, preserving the supplied
  pretrained weights. Load using `Algorithm.from_json(gzip.decompress(...).decode())`.
- `scene.xml`: replay model. Its mesh directory is relative to an `assets/`
  checkout; fetch pinned assets into this folder's `assets/` to load it directly.
  Do not commit the fetched checkout. Normal rollout generation constructs its
  own scene from the explicit `--assets` path.
- `artifact-sha256.json`: hashes of the original numerical evidence bundle.
- `*-resources.json`: CPU and available-memory samples and enforced job limits.
- `renderer.json`: actual OpenGL renderer/vendor reported by the video renderer.
- `execution/humanoid_evaluate.py`: exact evaluation driver used for the recorded
  run, before import/formatting cleanup. Its hash matches `report.json`.

The main controller source hash in `report.json` matches
`examples/robotics/humanoid/humanoid_policy.py`. The report was computed on the GB10 peer with
CPU-only PyTorch. No GPU training was needed. The host was released when unrelated
work exhausted its CPU headroom; other workloads were not modified.

## Asset provenance and license

Upstream: <https://github.com/unitreerobotics/unitree_rl_mjlab>, revision
`1425b15f73bd4095f0df53709d7c389c3eb9e790`. Model:
`src/assets/robots/unitree_g1/xmls/g1.xml`. Policy:
`deploy/robots/g1/config/policy/velocity/v0/exported/policy.onnx`.

The upstream Apache 2.0 license is retained as `UNITREE-LICENCE`, including the
upstream Isaac Lab attribution. The compiled walking artifact is derived from
those supplied weights; the scene adds a gripper, pedestal and cube and changes
contact filtering. Geometry is fetched from the pinned upstream checkout rather
than copied into this repository. TensorCode's new integration and presentation
code is covered by this repository's license. No physical-hardware capability,
perception performance, broad robustness or real-time guarantee is claimed.
