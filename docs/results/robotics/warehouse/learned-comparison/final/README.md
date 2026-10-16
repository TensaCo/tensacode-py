# Original final matrix — incomplete execution evidence

The original 40-cell matrix contains **33 audited runs and seven runtime-incomplete
runs**. The latter terminated with a shared scene timestamp ordering exception;
they are not physical-policy failures or successes. All runs, logs and receipts
are preserved. `complete_and_matched` remains **false**: this is not a complete
five-condition policy comparison. The one extra audited long-outage run is retained.

`original-final-audit.json` is the exact original independent audit.
`audit.json` independently checks the packaged artifacts; all 40 statuses agree.
`independent-physical-audit.json` retains a separate physical audit, including the
seven traceback log hashes and three recorded fall stops.
`common-condition-summary.json` is a separately labeled descriptive intersection
identified **after** the runtime failures: four conditions have valid evidence for
all eight policies. On these four, authored recovery completes 8/8 placements and
4/4 episodes; each learned candidate completes 0/8 placements and 0/4 episodes.
This does not establish a complete five-condition comparison or universal superiority.
The fifth condition remains visible in the full matrix.

Conditions were excluded from candidate training but previously exercised by the
original authored baseline. Co-simulated robots are correlated. Full clocks,
available raw sensor/actions, replay poses, source bindings, failures and resource
receipts remain available. Traces are compressed losslessly. Missing artifacts are
not reconstructed. The independent audit checks recorded actuator projection and
sampled physical settling; it does not independently reconstruct dynamics.

The frozen plan remains unchanged in `../final-plan`. `admission/` preserves exact
peer-input verification, queue-admission receipts and all six immutable queues.
Models use hash-verified relative links to frozen portable bundles. Use a normal
Git checkout with symlink support: raw GitHub symlink URLs contain link text,
not downloadable model weights. Original timestamp-failure cases must not be
replaced by any separately versioned bug-fix diagnostic.
