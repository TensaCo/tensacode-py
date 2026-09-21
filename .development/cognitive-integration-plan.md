# Cognitive integration and evaluation

This milestone turns source evidence into revisable interpretations before language
or explicitly authorized actions. Work stays on main. Substantial training and
real-model evaluation run on `gb10-direct`; the development host runs small tests.
The owner confirmed `jacob-valdez` as the checkpoint namespace.

## Implementation

1. Immutable evidence, hypotheses, assessments, goals, plans and observations;
   preserve source revisions and invalidate stale interpretations.
2. Owned hypothesis generation and source-wise NLI verification, explicit label
   mappings, held-out temperature calibration, complete offline checkpoints.
3. Cognitive sessions with an explicitly authored support/contradiction policy,
   evidence revision and learned-encoder episodic retrieval. Generated outputs
   never become observations automatically.
4. Chatbot owns the above components and realizes selected interpretations through
   its own decoder. Questions alone are not factual evidence. Session state and
   weights have separate persistence boundaries.
5. Planner generates proposals; validated action registries execute explicitly
   structured plans. Actual outcomes label only executed choices; execution is
   bounded and replans after observations.
6. Scene can own an image-to-text model for relational descriptions, preserving
   original image source anchors and uncertainty. Symbolic graph operations stay
   stubs; no image-to-claim interface or invented bounding boxes.

## Evidence required

- Small deterministic behavior tests: source preservation, revision, abstention,
  no supervision leakage, model/session roundtrip, stale memory detection,
  outcome-only learning and action validation before effects.
- GB10 verifier evaluation: pinned public data/models; distinct fine-tune,
  calibration and test examples; pre/post accuracy, NLL, Brier and ECE. Disclose
  foundation pretraining overlap rather than asserting globally unseen examples.
- Action-outcome experiment: actual simulator transitions and disjoint scenarios;
  report it as a simulation, not evidence of production operational competence.
- Real-image language evaluation with blank/shuffled image controls. Supplied
  pretrained VLM ability is separate from any TensorCode-trained improvement.
- Independent review, full lightweight tests, build and documentation checks;
  coherent commits/pushes with measured limitations.

## Boundaries

Schemas, attention slots and authored policies are mechanisms, not learned
cognition. NLI probabilities do not establish truth or source trust. Temperature
calibration is sample-dependent and must be repeated after weight changes.
Natural-language proposals are never executable actions. Successful serialization
does not prove competence. Existing negative workspace/vision results remain in
the developer validation record.
