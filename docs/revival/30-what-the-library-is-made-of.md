# 30 — What the library is made of

Measured on 2026-09-18, at `c32667e`, by walking the import graph of `src/tensorcode`
(resolving relative imports and submodule imports by name) and then asking which modules
are reachable from `tensorcode.agent`.

```
library                     17,246 lines   63 modules
reachable from the agent     8,129 lines   26 modules
not reachable                9,117 lines   37 modules
```

Nothing is dead: every one of the 37 unreached modules is imported by a test, an example,
or an eval script. That is the finding, and it is worse than dead code would be. Dead code
is deleted in an afternoon. What is here is a **second library** — with its own tests and
its own measured results — that the only agent in the repo never calls.

## The 9,117 lines, by what they are

| group | lines | modules | what it is |
|---|---:|---:|---|
| cognitive architecture | 6,250 | 20 | `awareness`, `cues`, `priming`, `permanence`, `wants`, `social`, `memory`, `metacognition`, `cognition`, `expectation`, `frames`, `chunking`, `change`, `causal`, `relation`, `temporal`, `quantity`, `answer_type`, `semantics_bridge`, `control` |
| learning | 1,078 | 6 | `learning/{induce,verify,certificate,library,literals}` — imported only by each other |
| backends | 750 | 4 | `builtin`, `linear`, `neural`, `hf_local` — implementations for ops, reached only from examples and eval |
| domain knowledge in code | 552 | 1 | `language/domains/desktop.py` |
| grammar induction | 257 | 1 | `language/induce.py` — the substitutable and congruential learners |
| vision | 144 | 1 | `vision/hierarchy.py` — the layer-wise k-means hierarchy |

## Verdicts

**The cognitive architecture (6,250 lines) is the decision this repo has been avoiding.**
Each module is coherent and tested in isolation. None of them is reachable from a turn. Two
honest options, and the choice is the owner's:

1. *Adopt* — the agent's turn is expressed in them (a turn raises expectations, primes cues,
   forms wants, and keeps them across turns), which is a much larger job than any phase in
   [29](29-architecture-review.md) and would have to earn its place on the assay; or
2. *Remove before 0.2* — they leave with their tests, and the repo stops implying a
   cognitive system it does not run.

Keeping them unreached is the one option that is not honest, because the package's surface
advertises capabilities no measurement covers. Nothing here should be adopted *because it
exists*: the assay decides, and the note in 29 §29.4 applies — a module earns its way in by
moving a number that a control does not move.

**`learning/` is an island.** Its five modules import each other and nothing else imports
them — not the agent, not the tests, not the assay. Whatever it once served, the induction
work that is measured today lives in `language/induce.py` and in the formal-language tasks.
Decide with the block above; it has the weakest claim of anything in the list.

**`language/domains/desktop.py` (552 lines) is the smell the owner named.** It is desktop
knowledge written as code: what a folder is, what lives on a desktop, which words name
which actions. The computerworld plugin replaced it with discovery by experiment
(`examples/general_agent/discover.py`), which finds the same facts by running commands and
watching what changes. This one has an answer already — it should go when the examples stop
importing it, and the import is the only thing keeping it.

**The backends are not a problem, they are the missing wiring.** `builtin` and `linear` are
exactly the kind of implementation [`agent/operations.py`](../../src/tensorcode/agent/operations.py)
now registers. They were unreachable because nothing called `ops`; after the operations
work they are candidates to register rather than candidates to delete.

**`vision/hierarchy.py` and `language/induce.py` stay as they are.** Both are reached by
eval scripts, which is the correct relationship: they are hypotheses under measurement, and
the assay records that the hierarchy *lost* to a flat model (62.7% vs 57.1% on the CIFAR
diagnostic) and which formal languages the learners get exactly. A hypothesis with a
measured negative result is worth more than one that was quietly dropped.

## The eval directory

126 scripts. Fifteen of them are the assay (`eval/suite/`); the other 111 are the
historical record, and 91 are cited by name in the docs — they are the provenance of claims
this repo makes about itself. They are not clutter to be swept up, and deleting them would
silently orphan measured statements.

What they need is a stated relationship, not a purge:

* `eval/suite/` is the **current measurement surface**. A claim about how the agent does
  belongs to a registered task, with controls and an append-only row in
  `eval/results/assay.jsonl`.
* everything else in `eval/` is **the record of an experiment that has already run**. It
  stays runnable and stays cited; it is not where new measurement goes.

Twenty scripts outside the suite are cited by no document. Those are the only candidates
for removal, and each should be checked for a result worth keeping before it goes.

## What changed since 29 was written

`29-architecture-review.md` §29.1 reported "23 modules / ~6,500 lines unreachable" from a
narrower reading. Walking the graph properly gives **37 modules / 9,117 lines**, and adds
the important qualification that all of them have tests or eval users. The conclusion is
unchanged and sharper: the agent and the library are two programs in one repository, and
[the operations work](../../src/tensorcode/agent/operations.py) has started joining them at
four points — `parse`, `choose`, `rank`, `verify`.
