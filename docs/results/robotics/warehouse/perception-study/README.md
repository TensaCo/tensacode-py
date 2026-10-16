# Multi-view scene evidence

Frozen geometric consensus and simultaneous view fusion were evaluated on **273 actual RGB-D replay captures**. The development split uses one existing two-G1 episode. Evaluation uses 65 object states per kind from two other existing episodes, with new camera viewpoints and intermediate frames. Existing final top views had already been inspected. This is a held-out-view replay study, not a new controller trial or independent hardware validation.

| Evaluation condition | Kind | Single-view inverse: median error / localized | Three-view consensus fusion: median error / localized |
| --- | --- | --- | --- |
| Clean | Parcel | 1.74 mm / 39 of 65 | **1.26 mm / 62 of 65** |
| Clean | Receiver | 0.80 mm / 49 of 65 | **0.66 mm / 57 of 65** |
| 1 mm depth noise + 20% pixel occlusion | Parcel | 6.19 mm / 0 of 65 | **1.25 mm / 62 of 65** |
| 1 mm depth noise + 20% pixel occlusion | Receiver | 21.00 mm / 0 of 65 | **4.82 mm / 53 of 65** |

Median error includes detected fits, including fits that abstain. Localization coverage is reported separately. Across the four matched conditions, fusion localizes **468 of 520 state observations**, with **zero localized errors above 10 mm**; the largest is 8.26 mm. These observations are correlated replay samples and perturbations, not 520 independent physical episodes.

Extra views supply extra information. The same three-view arithmetic-mean inverse baseline reaches 10.38 mm median receiver error under noise plus occlusion; 58 of its 62 localized outputs exceed 10 mm. Clean receiver centroid estimation has a better median than fusion (0.40 mm), but six errors exceed 10 mm. All methods and failures appear in `results.json`. The temporal variant is retained experimentally because its moving-parcel coverage is lower than simultaneous fusion.

The new estimator rejects RGB/depth silhouette mismatches using explicit box-surface consensus, retaining inlier counts and original-frame hashes. It never updates an unobserved coordinate simply to increase confidence. It uses known color, metric size, axis alignment, camera calibration and a fixed workcell region. The confidence score is heuristic; it is not a calibrated probability. The occlusion intervention removes measured pixels in software; it is not a new simulated physical occluder.

Final three-G1 parcel observations are all localized, with 1.23–1.28 mm evaluation error. **640×480 single-view rendering also resolves the prior 320×240 final-parcel abstention**; the entire final QA improvement cannot be attributed to fusion. The fresh observation adapter consumes pixels/calibration only and does not command robots or assert physical contact.

`results.json` contains every metric, per-state outcome, input hash and frozen-source hash. `source.zip` contains the final implementation. `captures/` includes all final views (21 captures); the remaining captures can be recreated from the existing archived physics configurations using `capture.py`. `resources/` records guarded hardware rendering and CPU fitting. The full temporary dataset remains at `/tmp/tensorcode-perception-campaign`. `provenance.json` records the development-source retention limit.
