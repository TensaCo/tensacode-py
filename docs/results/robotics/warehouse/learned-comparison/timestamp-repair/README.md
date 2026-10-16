# Separate timestamp runtime-repair diagnostic

These seven runs repeat only the original long-camera-outage executions that
terminated with a shared scene timestamp ordering exception. The narrow runtime
amendment, source diff, exact frozen models and all physical outcomes are preserved.
The bookkeeping fix prevents the metadata exception; stale perception still
causes the authored controller to time out, with 0/2 placements. Every one of
the seven repaired policy runs also has 0/2 placements. This is not an improvement
in task performance.

The original 40-cell matrix in `../final` remains unchanged: 33 audited runs and
seven runtime-incomplete runs. This diagnostic does not replace those cells or
turn the original matrix into a complete matched benchmark. Diffusion512-primary
is excluded because its original long-outage execution was already audited.

`final-audit.json` is the exact independent root audit; `audit.json` independently
checks the packaged evidence. `independent-physical-audit.json` and
`independent-source-diff.json` preserve the additional independent checks.
`plan/` contains the exact diagnostic protocol, seven-cell execution manifest,
condition and source archive. `amendment.json` records original failure paths and
runtime changes. No candidate was retrained or selected using these outcomes.

Full clocks, sensor/action episodes, qpos, compressed records, failed physical
outcomes and resource receipts are retained. Co-simulated robot placements are
correlated. Audits verify recorded physical settling and actuator projection;
they do not independently reconstruct dynamics. This is a targeted development
diagnostic, not evidence of universal policy superiority.

Identical runtime sources and frozen model files use hash-verified relative
symlinks. Use a normal Git checkout with symlink support: raw GitHub symlink URLs
contain link text, not downloadable weights. Manifests include stored and
losslessly decompressed hashes where applicable.
