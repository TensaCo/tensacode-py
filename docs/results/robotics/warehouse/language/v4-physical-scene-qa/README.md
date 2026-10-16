# Actual RGB-D scene question answering

Three newly generated, pinned 4B model query intents were certified and answered from the final RGB-D scene graph of the 160-second physical single-twojobs-v2 trial:

- `Where is parcel_0_a?` → observed parcel position, with its camera-track source.
- `Where is the green parcel?` → observed parcel_0_b position, with its camera-track source.
- `Where is parcel_404?` → unknown; no invented object or location.

All three produced their predeclared answer statuses. The model only parses the query; deterministic grounding reads the archived scene graph at simulation time 159.98 s. Answers include observation confidence, estimated uncertainty, and source identifiers. The reader verifies the frozen composition-file hash and never reads evaluator-only object poses or success data.

This is offline historical perception QA, not a successful robot-control trial: the source trial failed its first physical task. The shared scene estimator uses declared color/shape and calibration priors. Raw model outputs, native model/prompt hashes, typed certificates, actual answers, source archive, and per-unit 32/48-SM CUDA preflight are retained.
