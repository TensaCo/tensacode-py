# Six-job environment development sweep

Six development conditions use one shared three-robot/six-job episode each.
The frozen v5 controller and an existing certified mission are reused unchanged;
**no fresh language inference** occurs. Robots/jobs within an episode are correlated.

| Condition | Strict final placements |
|---|---|
| Host nominal | 6/6 |
| Pickup x +20 mm | 6/6 |
| Pickup x −20 mm | 3/6 |
| Pickup y +20 mm | 3/6 |
| Payload 200 g | 2/6 |
| Receiver +10 mm | 6/6 |

Total: 26/36 placements and 3/6 successful whole episodes. Controller failure
can coexist with a physically settled parcel; retained outcomes distinguish both.
Failures and never-reached second jobs are not omitted. This is a development
sweep, not an independently replicated or held-out benchmark.

Original reports, protocol, preflight and harness bytes are preserved. The audit
checks actual masses/inertias, pickup/receiver geometry, source bindings, initial
state hashes and strict current settling. `backout-diagnosis.json` uses evaluator
truth only for offline diagnosis; it is not controller input.

`evidence-index.json` maps historical run paths to local files. Raw traces and full
scene histories are losslessly compressed; poses and per-job sensors are retained.
Runtime source archives are deduplicated through relative symlinks under `shared/`;
use a normal Git checkout with symlink support. The protocol's full frozen source
is `frozen-source.zip`. Each run retains its exact reused mission and environment
case/proof/harness/protocol sidecars.

`admission/` preserves initial prelaunch path-policy rejection logs and corrected
admission records. Rejections occurred before physics and are not policy failures.
Original absolute paths in execution receipts describe the recording environment.
