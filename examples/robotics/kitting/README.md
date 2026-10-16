# Order-driven kitting

A five-actuator Cartesian robot composes inspection, inventory matching,
quarantine, contact-based pick-and-place and bounded retries. Work orders are
JSON. Supplied simulator poses, identities and quality flags stand in for a real
perception and inspection system; no policy training takes place.

[Measured report](../../../docs/results/robotics/kitting/evidence.html) ·
[Reproduction and limitations](../../../docs/results/robotics/kitting/README.md)

From the Python checkout:

```sh
python -m pip install -e '.[robotics]'
python examples/robotics/kitting/run_recipe.py \
  --recipe examples/robotics/kitting/recipes/change_order.json \
  --substrate alg2 --output /tmp/my-kit
```

The output directory must be new. A failed order exits with code 2 and retains
its receipt and evidence. `alg0`, `alg1`, `alg2` and `native_feedback` are available.
Orders have 1–3 A/B slots; inventory is three A parts and one B part. An optional
revision arrives after the first inspection. Reject indices identify supplied
quality failures; the optional missed-grasp injection offsets the first kit pick.
This is deliberately bounded: it does not handle arbitrary mid-transfer revisions.

- `cell.py`: authored MuJoCo mechanics and native motion stages.
- `policy.py`: generic prerequisite circuit composed with inventory selection.
- `run_recipe.py`: strict recipe parsing and one-order execution.
- `evaluate.py`: paired protocol, baselines, receipts and selected trajectories.
- `render.py`: annotated replay of recorded states.
- `build_report.py` and `report.html`: standalone evidence presentation.

The compiled circuit performs task readiness and part matching. Motion-stage
interpolation, dispatch, retries and observation extraction remain host code;
MuJoCo position actuators provide low-level control. “Seal” is logical order
release, not physical packaging. All objects remain free bodies without welds or
pose overwrites during the simulation.
