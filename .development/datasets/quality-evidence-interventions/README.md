# Training-only evidence interventions

Prepared from the training partition of response-quality-v2. The first 64 distinct
questions with all three original labels positive retain their question and
candidate, with either no evidence or the next selected question's evidence.
Sources are disjoint between those questions. No calibration/development questions
were selected and no final questions were read.

Four assistant review shards cover all 128 interventions. Support is false when
the supplied evidence does not license the candidate; this is not a claim that
the candidate is false in the world. Completeness and constraints are masked to
isolate evidence-support supervision. These are authored interventions, not
natural model failures or human ground truth. References/rationales are metadata,
never model inputs.

To reproduce, run `.development/experiments/prepare_quality_interventions.py
prepare --data <v2-prepared> --output <new-dir> --count 64`. Combine its candidates
with the v2 candidates, supply both sets of review labels to the quality training
preparer, and retain the v2 selection manifest. The combined partitions contain
495 training, 92 calibration and 91 development candidates. Five epochs now use
620 updates rather than 460; the intervention comparison is not equal-compute.

The trained model failed promotion: 49 accepted development candidates included
30 known-good, 12 known-bad and 7 unresolved. See the comparison result in
`docs/results/response-quality-comparisons.json`. No production integration.
