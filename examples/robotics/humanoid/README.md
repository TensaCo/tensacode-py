# G1 humanoid walking and gripping

| Script | Purpose |
|---|---|
| `fetch_humanoid_assets.py` | Fetch pinned, licensed model and policy assets |
| `humanoid_policy.py` | Compile the supplied walker and execute the authored manipulation sequence |
| `humanoid_evaluate.py` | Compare substrates, perturbations and the open-gripper control |
| `humanoid_budget.py` | Bound job resources and monitor machine headroom |
| `render_humanoid.py` | Render annotated physics replay videos |
| `validate_humanoid_evidence.py` | Validate traces, metrics, hashes and exported computation |
| `build_humanoid_report.py` | Embed committed media and metrics into the HTML report |

[Reproduction commands and dependencies](../../../docs/results/robotics/humanoid/README.md)
cover the complete workflow. Run commands from the Python repository root.
[View the report](../../../docs/results/robotics/humanoid/index.html).

The supplied walking network and authored controller are distinct from learned
TensorCode improvements. Reorganizing files does not change the recorded evidence.
