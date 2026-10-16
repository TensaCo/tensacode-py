# Language-grounded vision: measured progress and remaining failures

[Browse every prediction](index.html) · [Comparison data](comparison.json) · [Artifact validation](validation.json)

Real **Grounding DINO Tiny** inference consumes RGB and language. A separate adapter uses measured depth and camera calibration to sample visible surfaces or abstain. No simulator object pose, color mask, or evaluation label enters the neural model or depth adapter. These are external calibrated RGB-D cameras, not an onboard camera demonstration.

| Six matched v2 captures | Separate generic queries | Frozen joint prompt |
| --- | ---: | ---: |
| Task-parcel amodal boxes matched at IoU ≥ 0.5 | 0 / 12 | 4 / 12 |
| False elephant boxes, when no elephant exists | 6 | 2 |
| Neural forward passes | 24 | 6 |
| CPU wall time, including loading | 149.2 s | 39.3 s |
| Depth samples / abstentions across proposals | 23 / 26 | 7 / 9 |

The joint prompt names a **small cube** instead of a generic box and supplies robot, shelf and elephant alternatives together. Both wording and context changed; this experiment does not isolate their effects. It improves these two narrow measures, but still misses eight parcel labels and produces two false elephant detections. Robot/shelf proposal counts are not independently annotated accuracy measures.

This is **offline perception**, not robot control. Stable depth can belong to background. Returned points are visible-surface samples, never object centers, grasp targets or proof of entity identity. Scores are not calibrated probabilities. The current controller should not trust these proposals without additional geometric and task checks.

## Protocol and evidence

Development used the six `smoke-v1` captures. Separate generic queries missed all 12 parcel labels. A fixed 2×2 crop pilot also missed both parcels and produced three false elephant boxes at four times the forward-pass cost; it was not promoted. One development-only joint-prompt pilot preceded the frozen `smoke-v2` comparison. Every pilot and failure is retained in the browser.

The evaluation contains initial and approximately three-second states at the **same three camera angles** as development. They are new captures of a related simulation revision, not unseen camera angles, independent warehouse episodes, or completed task demonstrations. Tiny solid neon cubes also differ from common natural cardboard-box imagery.

Independent labels are offline amodal projections of the two task parcels, including occluded parts. They are read only after predictions finish. These partial labels support parcel recall, not whole-image precision or visible-only recall. In the label overlays, cyan denotes neural proposals and amber denotes evaluation labels.

Each evaluation directory includes exact RGB, depth and calibration inputs; all predictions; an immutable source ZIP; source/input/model hashes; protocol; and runtime/resource records. Development's earliest pilots retain source hashes without a source ZIP. The initial pilot had a four-core quota but its actual child affinity was not recorded; all later runs explicitly verified CPUs 12–15.

Model: [IDEA-Research/grounding-dino-tiny](https://huggingface.co/IDEA-Research/grounding-dino-tiny), Apache 2.0, revision `a2bb814dd30d776dcf7e30523b00659f4f141c71`. Weight SHA-256: `1a2412ef99bd74bcd3c2a246fa1e48581f8889a1300c9051974741314fc042f3`. Weights remain outside Git. No training or paid service was used.

The evaluation ran CPU-only with four threads, a 12 GiB hard memory cap and global headroom guard. Peak process RSS was approximately 2.61 GiB; available host RAM stayed above 81% during both final runs. The generic guard metadata's Mesa renderer label does not describe model inference: recorded model runtime explicitly says `cpu`, with actual affinity `[12,13,14,15]` and a four-core quota.

Implementation: `examples/robotics/warehouse/semantic_vision/`. Eleven tests cover depth calibration, invalid/mixed-depth abstention, one-to-one partial-label scoring, and truth-independent crop remapping. `validate.py` recomputes artifact consistency and recall; it does not prove semantic correctness.
