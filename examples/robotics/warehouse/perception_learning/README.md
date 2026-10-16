# Multi-view warehouse perception experiment

This directory contains an offline replay-view experiment and an observation adapter. It does not contain a trained vision model. RGB-D pixels and calibrated camera poses enter geometric inference; simulator geometry enters a separate evaluator only after inference returns.

`protocol.json` declares the development/evaluation views and matched conditions. `capture.py` renders actual saved G1 trajectories into RGB-D frames and writes separate evaluation labels. `evaluate.py` compares the frozen inverse estimator, a known-height corrected centroid, arithmetic multi-view averages, geometric-consensus fusion, and an experimental temporal tracker. All methods use the same calibrated cell boundary.

`robust_geometry.py` fits an initial known-size box, rejects RGB/depth silhouette pixels inconsistent with its surface, and refits only when at least 70% of measured pixels support the hypothesis. Rejection counts and original-frame hashes remain in the evidence. It assumes the same known colors, metric dimensions and world-axis alignment as the frozen estimator.

`scene_observer.observe_views` provides a sensor-only simultaneous-view adapter for scene graphs and QA. Supply at least two distinct calibrated camera positions and a fixed workcell boundary. Returned IDs belong to the observation revision; callers must establish semantic identity separately. Missing axes and ambiguous identity retain abstention. Geometric proximity never asserts physical contact.

The temporal tracker is experimental: on development moving-parcel captures it has lower usable coverage than simultaneous fusion. Extra camera views are additional sensing; arithmetic multi-view baselines separate that advantage from fusion quality. Existing final top views were inspected before protocol declaration; evaluation uses held-out viewpoints/intermediate frames of existing episodes, not newly held-out physical rollouts.

Commands from the Python repository root (run rendering/evaluation under the campaign resource guard):

```bash
.venv/bin/python examples/robotics/warehouse/perception_learning/capture.py \
  docs/results/robotics/warehouse/runs/two-parallel-v1 \
  /tmp/perception/development --split development
.venv/bin/python examples/robotics/warehouse/perception_learning/evaluate.py \
  /tmp/perception/development --output /tmp/perception/development-report.json
```
