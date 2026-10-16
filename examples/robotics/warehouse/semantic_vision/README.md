# Language-grounded scene proposals

An optional pretrained Grounding DINO adapter maps **RGB + language query → 2D boxes**. A separate numerical adapter samples measured axial depth and returns a visible-surface point or abstains. The detector receives no object colors, simulator poses, evaluation labels or calibrated workcell target coordinates.

This is an **offline perception study**, not a robot policy. Images come from calibrated external cameras replaying the initial/final states of three-second simulation smoke runs, not completed warehouse task episodes. A detected label does not establish persistent entity identity, support, contact, or a safe grasp. A box's central depth can belong to background; its point is explicitly not an object center. Grounding DINO scores are not calibrated probabilities.

The frozen [protocol](protocol.json) retains separate-query, fixed-crop, and joint-prompt development results. Fresh-view evaluation compares separate generic queries with one frozen joint prompt; the ineffective crop candidate stops after development. All misses, false elephant detections and depth abstentions remain in the artifacts. Independent projected labels cover only the two task parcels, including occluded portions. They support amodal parcel recall at IoU ≥ 0.5, not whole-image precision. No training occurs.

## Run

Use the optional PyTorch/Transformers environment; the recorded study used Torch 2.14.0 and Transformers 5.17.0. Model weights stay outside Git.

```bash
hf download IDEA-Research/grounding-dino-tiny \
  --revision a2bb814dd30d776dcf7e30523b00659f4f141c71 \
  --exclude pytorch_model.bin --max-workers 2

python -m examples.robotics.warehouse.semantic_vision.study \
  CAPTURE_DIRECTORY OUTPUT_DIRECTORY \
  --model-path CACHED_SNAPSHOT_DIRECTORY --split evaluation
```

Run under the project's resource guard. Explicitly place `taskset` **inside** the guarded systemd command; record actual child affinity. Default inference is CPU-only, four threads. Never silently select CUDA. A shared machine still requires coordinated CPU lanes and RAM headroom.

The study copies exact RGB, depth and calibration inputs, renders all query outputs, records source/model/input hashes and runtime, and only then opens independent evaluation annotations. Outputs are immutable: existing output directories are rejected.

## Model sources

- [Official IDEA Research model card](https://huggingface.co/IDEA-Research/grounding-dino-tiny), Apache 2.0; approximately 689 MB safetensors.
- [Transformers implementation documentation](https://huggingface.co/docs/transformers/model_doc/grounding-dino).

The model consumes language-conditioned visual features; the depth stage is a geometric computation, not a learned pose estimator. This combination offers semantic proposals beyond predefined color masks, while exposing why unvalidated neural proposals cannot directly authorize robot motion.
