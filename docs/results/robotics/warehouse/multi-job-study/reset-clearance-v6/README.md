# V6 reset-clearance development experiment

Four three-robot/six-job episodes test an explicitly versioned extension to the
frozen v5 controller. Exact mission input is reused; there is no fresh language
inference. Original results, proposals, amendments and execution failures remain.

| Condition | V5 placements | V6 placements |
|---|---|---|
| Nominal | 6/6 | 6/6 |
| Pickup x −20 mm | 3/6 | 6/6 |
| Pickup x +20 mm | 6/6 | 5/6 |
| Pickup y −20 mm | Not paired | 6/6 |

On the three matched conditions, placements change from **15/18 to 17/18**, while
whole-episode success remains **2/3 for both versions**. The targeted x− case
improves, but x+ regresses. The additional y− case is separate. This is not
universal superiority or a held-out generalization result: there is one episode
per condition, with correlated robots/jobs. All four v6 cases yield 23/24 placements
and 3/4 complete episodes.

`comparison.json` derives the table from independent physical audits. Reports
retain controller failures separately from physical settling. Full clocks,
records, qpos, per-job sensor evidence, extension/base source bindings and exact
mission artifacts are retained. Compression is lossless and hashed.

`preflight-smoke/` contains a loader smoke explicitly excluded from the four cases.
`admission/` preserves loader import failures before physics plus corrected run
receipts. `development-history/` retains unexecuted proposals and amendments;
these are not additional task trials. Runtime archives and the frozen base source
use hash-verified relative symlinks; use a Git checkout with symlink support.
