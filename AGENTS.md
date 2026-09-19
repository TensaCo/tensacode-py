# Repository workflow

Work on `main` only. The owner uses this branch for the current experimentation
phase; do not create or switch to exploration branches or isolated worktrees.
Preserve existing work and coordinate file ownership when agents work in parallel.
The owner authorizes pushing completed, verified milestones to `origin/main` as
work progresses. Keep checkpoints coherent and report verification and limitations.

# Agent architecture

The governing objective is `docs/revival/36-structured-cognitive-workspace.md`:
unstructured evidence becomes revisable structured interpretations, where reasoning,
planning, hypothesis formation, and learning operate before action and language
realization. Vision includes holistic scene understanding and relational organization,
not only element identification. Preserve uncertainty and source evidence.

For cognitive changes, distinguish supplied models, authored policies, and graph
fixtures from capabilities learned or inferred from real inputs. Report the active
behavior improved and remaining gaps; new schemas alone do not establish cognition.

Do not restore the removed image-to-claim API, first-reader execution default, or
bundled semantic seeds through a compatibility flag or a renamed fallback. See
`docs/revival/39-removing-implicit-semantic-authority.md`. Tests may explicitly
supply authored knowledge to isolate mechanisms; production must not import it.
