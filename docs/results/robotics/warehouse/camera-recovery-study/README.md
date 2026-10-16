# Camera × tactile recovery development comparison

Twelve full 85-second episodes cover three **previously observed development
conditions**: nominal placement and pickup offsets of +10 mm along x or y.
Two nominal episodes are reused from the camera study and linked by exact hashes;
the other ten episodes are newly recorded. Two robots sharing one scene are not
independent episode replications.

| Camera samples | Tactile recovery | Strict final placements |
|---|---|---|
| 4 | Off | 4/6 |
| 4 | On | 6/6 |
| 0 | Off | 6/6 |
| 0 | On | 6/6 |

Recovery repairs two failures with four-sample rendering. Zero-sample rendering
already succeeds in all six placements; adding recovery yields no measured gain
there. This is a small development comparison, not held-out generalization.
Failures, full clocks and original source snapshots remain in the evidence.

`protocol.json`, `results.json`, `mechanisms.json` and `summary.json` preserve the
original reports byte-for-byte. `evidence-index.json` maps their recorded run names
to local evidence paths, including the two reused episodes without duplicate raw
files. Each run manifest hashes source, sensor, command, evaluation and replay
artifacts. The frozen `execution-source.zip` matches the protocol hash, and every
run snapshot file matches its corresponding robotics source inside that archive.
`records.json.gz` decompresses to the exact original trace bytes.
Independent audits check sampled strict final settling and scheduler ordering;
retained evaluator geometry is not an independent dynamics reconstruction.
