# 35 — Model-based planning and inspectable refinement

> **Subsequent breaking change, 2026-09-19:** [39](39-removing-implicit-semantic-authority.md)
> removes default language commitment, automatic reference resolution, and bundled project
> recipes. The examples below record this earlier checkpoint. Current language execution
> requires explicit interpretation and any necessary reference/domain knowledge; structured
> goals and filesystem action models remain available.

> **Objective update, 2026-09-19:** this milestone strengthens execution of supplied
> specifications and action models. [36 — The structured cognitive workspace](36-structured-cognitive-workspace.md)
> sets the next direction: constructing and revising those specifications from uncertain
> language and visual evidence, then forming and learning models within the same loop.
> Authored refinements remain authored knowledge; configurable conventions are not learned
> understanding. The implementation claims below retain their narrower scope.

*2026-09-19. Implementation follow-up to [34](34-cognitive-primitives-and-implementation.md).
This advances the structured project episode and one real language-to-filesystem path.
It does not replace the historical cognitive benchmark results in [33](33-the-cognitive-fronts.md).*

## What now works, and through which interface

A `GoalSpec` can now lead to multiple actions selected by bounded search, with fresh
precondition checks, observed effects after each step, and a final observation of the
whole task specification. It can hold explicit conditions across these steps, suspend
after a caller-specified number of verified actions, and resume by replanning from
current observations. Caller-driven revision preserves task identity and earlier attempts.

The real [`FileSystemPlugin`](../../src/tensorcode/agent/filesystem.py) supports directory
creation and exclusive creation of UTF-8 text files beneath a supplied root. A packaged,
replaceable refinement convention supplies the meaning of a minimal Python project.
The ordinary `Agent.turn("make a python project called hello")` path creates a directory
and a runnable `main.py`. Named-project requests using `create` and `build` have also been
observed working. Its integration regression executes that file with Python and
checks its output. This is actual filesystem work, not the computerworld simulator.

These claims have different scopes:

| Path | Implemented behavior | Remaining boundary |
|---|---|---|
| Explicit `GoalSpec` | Multi-action planning, independent task-condition verification, held conditions, suspension and caller-driven revision/resumption | Caller supplies the specification and revision; no English correction interpretation |
| Parsed named Python project | Semantic interpretation → inspectable recipe → generic planner → real filesystem → verification | A narrow hand-authored project convention, dependent on parser/resource coverage |
| Original “make a python hello world project” | Inspected as an unresolved failure | Both the treebank and grammar routes misinterpret the compound/valency structure; no whole-phrase workaround was added |
| Desktop capability verification | Refuses success when declared effects remain unsupported | Does not give the desktop plugin the new filesystem action model or prove general desktop decomposition |

The inspection of the original wording is empirical diagnosis, not a formal new benchmark.
A passing named-project request must not be reported as resolving that original probe.
Names can also trigger lexical ambiguity: inspections with names such as “notes” and
“mounted” exposed verbal readings. Supported examples do not establish arbitrary-name
coverage.

The example server can mount the real adapter explicitly:

```bash
.venv/bin/python -m examples.general_agent.server --reader grammar --plugin filesystem:/existing/root
```

Replace `/existing/root` with a supplied, existing directory. Bare `filesystem`, missing
roots, and non-directory roots are rejected; mounting does not create a root implicitly.
Existing server defaults are unchanged. This mount does not require the computerworld
plugin and has no desktop view.

## Generic search, explicit modeling assumptions

[`plan_goal`](../../src/tensorcode/agent/planning.py) performs breadth-first search over
plugin-enumerated grounded calls. It seeks a shortest action sequence within explicit
bounds: by default depth 12, 10,000 states, and 1,000 candidate actions. A plugin supplies
capabilities, preconditions, effects, candidate groundings, and observations; the planner
contains no Python-project recipe or utterance matching.

Conditions use exact predicate, role, and value equality. Initial truth is open-world:
unsupported observations and disagreement between observers leave a condition unknown.
Unknown preconditions do not enable actions. Predicted effects update search states, not
the agent's observed world. Unmentioned atoms retain their values within this finite action
model. This frame assumption is useful but cannot account for undeclared side effects.

A `no_plan` result means no plan in the supplied groundings and evidence, not that the
real-world task is impossible. Budget exhaustion is separately reported. An empty plan
means current observations already support the goal. Grounding remains domain work: the
filesystem adapter derives ancestor directories from paths and offers matching create/write
actions. General search does not invent missing task knowledge or action models.

The executor in [`core.py`](../../src/tensorcode/agent/core.py) checks declared preconditions
before dispatch, observes all modeled step effects, checks held conditions before and after
steps, and finally observes every task condition and invariant. A successful receipt alone
does not establish success. Final task verification is independent of matching a goal to a
capability's effect list, although observations still come from the participating plugins;
this is not an independently implemented world grader for every domain.

Execution uses detached capability-model snapshots and rejects changed declarations before
dispatch, including changes inside role mappings. Cross-plugin observations may establish a
precondition the executor cannot itself observe, but conflicts and malformed observations
block dispatch. Contradictory explicit conditions and contradictory action effects are also
rejected on the legacy single-capability path.

Retry checks inspect every step receipt in the current task revision: a rejected last step
cannot conceal an earlier mutation. Failed attempts also trigger held-condition observations.
Suspension after verified work remains a separate operation, resumed through fresh planning.
The legacy failure reporter now uses actual candidate diagnostics, removing its rules that
interpreted English error substrings and substituted speculative taxonomy explanations.

A held condition must already be established. Search rejects modeled intermediate states
that violate it. Runtime observations can detect violations at step boundaries; neither
these checks nor the model prove continuous preservation during an opaque action, absence
of unmodeled effects, or freedom from concurrent changes.

## Domain knowledge with provenance, rather than hidden recipe code

[`RefinementLibrary`](../../src/tensorcode/agent/refinements.py) reads declarative recipes
matching predicates and entity features. Expressions support literal values, bindings, and
path joins; they cannot execute arbitrary code. Ambiguous applicable recipes and unconsumed
qualifiers are explicit failures. Other goal conditions, invariants, and existing basis
entries survive refinement.

The then-packaged [`project_refinements.json`](../../tests/fixtures/project_refinements.json)
(now retained only as a test fixture)
is **hand-authored knowledge**: an unspecified Python project means a directory with
`main.py` printing “Hello, world!”. Its default name, location, entrypoint, and contents
are conventions, not deductions or learned knowledge. Callers can replace the library or
set `refinements=False`/`None`. Applied recipe IDs and sources enter `GoalSpec.basis` and
the execution trace. Naming and location bindings retain surface path text rather than
silently substituting noun lemmas. The location binding explicitly declares
`frame.roles.location` as a fallback to `entity.location`; when both are present, their
extracted values must agree. This is recipe data, not a project-specific parser branch.

Lexical accommodations are also explicit data: the recipe can omit the `made_of` relation
that VerbNet supplies with an unspecified material, and `be(Precondition=addressee)` for an
implicit addressee. Such omissions apply only to the declared implicit lexical conditions
and are recorded in the basis. Explicit conditions,
unhandled partial relations, unmapped lexical roles, and unconsumed frame features are not
silently deleted to make planning work.

Grammar and dependency interpretation now preserve ordered `(relation, value)` modifiers,
including repeated modifiers and nested dependency compounds. Scalar `quality`/`name`
compatibility aliases remain only when unambiguous. Grammar generation preserves the tested
modifier sequences through realization and reparsing, instead of selecting one modifier
and losing the rest. This repairs a representation loss; it does not solve compound parse
ranking or decide what every modifier means.

Recipes consume modifier metadata only when each relation maps through a declared alias to
a feature already matched or bound, with matching values. Unknown, repeated, or mismatched
modifiers still block refinement. The recipe does not blanket-ignore the new metadata to
restore a passing example.

Indirect requests now use replaceable, language-owned
[`request_conventions.json`](../../tests/fixtures/request_conventions.json)
(now retained only as a test fixture),
with convention IDs and sources exposed by interpretation. The modal-question convention
is a defeasible neutral-register seed, not proof of intent. Empty overrides disable it;
invalid explicit configuration fails visibly. Moving the old modal list to inspectable data
removes a hidden agent-level assumption, but does not constitute learned pragmatics or
resolve literal ability-question ambiguity.

These changes follow the correction to [32](32-knowledge-written-as-code.md): making
knowledge inspectable and replaceable is useful, while relocating a schema does not by
itself improve its semantics. Predicates, role conventions, finite conjunctions, and action
model contracts remain explicit limitations, not a claim that schemas have disappeared.

## A runnable structured correction episode

This example was executed successfully against the implementation. It uses a temporary
real directory and disables project recipes so the source of every desired condition is
clear. The caller supplies both the initial interpretation and the destination correction.

```python
from pathlib import Path
from tempfile import TemporaryDirectory
from tensorcode.agent import Agent, Condition, GoalSpec
from tensorcode.agent.filesystem import FileSystemPlugin

with TemporaryDirectory() as directory:
    root = Path(directory)
    (root / "README.md").write_text("Keep this.\n")
    agent = Agent([FileSystemPlugin(root, refinements=False)])
    held = (Condition("content", {"path": "README.md", "text": "Keep this.\n"}),)

    def specification(destination):
        return GoalSpec(
            (Condition("content", {
                "path": f"{destination}/main.py",
                "text": 'print("Hello, world!")\n',
            }),),
            invariants=held,
            basis=("caller-supplied minimal project specification",),
        )

    first = agent.pursue(specification("scratch"), max_steps=1)
    assert first.status == "suspended"
    agent.tasks.revise(
        first.task_id, specification("Documents/hello"),
        reason="Caller changed the destination after directory creation",
    )
    finished = agent.pursue(task_id=first.task_id)
    assert finished.status == "done"
    assert (root / "Documents/hello/main.py").is_file()
    assert (root / "README.md").read_text() == "Keep this.\n"
    assert (root / "scratch").is_dir()
    assert len(agent.tasks.get(first.task_id).attempts) == 2
```

Calling `agent.pursue(task_id=first.task_id)` on a suspended task also resumes without a
revision when the intended specification has not changed. It replans; it does not replay
a cached sequence. A revision never rolls back completed work, which is why `scratch`
remains above. Completed tasks require explicit revision before another attempt.

The ledger records each attempt's plan and step calls, receipts, and verification results.
It retains detached snapshots of revisions and attempts during the agent instance's lifetime.
This is in-memory history, not restart persistence. Recorded basis and planning rationale
make decisions inspectable; they do not yet implement conversational explanation quality.

The filesystem adapter rejects parent traversal and symbolic links, validates parent bindings,
and uses exclusive file creation to avoid overwriting an existing file. It supplies no
replacement/delete action. It is not an isolation boundary against another process changing
directory topology concurrently. Preserving a README here exercises an explicit invariant,
but the current create-only action set also limits the ways that invariant can be threatened.

## Evidence and remaining work

Focused regressions cover the search model, alternative plans, unknown evidence, bounded
search, modeled intermediate invariant violations, real filesystem artifacts, existing
contents, path rejection, recipe provenance and ambiguity, request conventions, and agent
execution/history integration. Relevant suites include
[`test_agent_planning.py`](../../tests/test_agent_planning.py),
[`test_filesystem_planning.py`](../../tests/test_filesystem_planning.py),
[`test_goal_refinements.py`](../../tests/test_goal_refinements.py), and
[`test_request_conventions.py`](../../tests/test_request_conventions.py),
[`test_planned_tasks.py`](../../tests/test_planned_tasks.py),
[`test_modifier_preservation.py`](../../tests/test_modifier_preservation.py), and
[`test_filesystem_mount.py`](../../tests/test_filesystem_mount.py).
Test-suite counts and historical cognitive benchmark scores are different evidence; no
cognitive-score improvement is claimed here.

The next integration boundary is meaning: interpret corrections as revisions to an existing
task, preserve compound structure and valency in the original project request, and make
semantic losses govern downstream decisions. Unconsumed entity features remain reported
losses in general projection; reporting is not interpretation of quantity, scope, or time.
The full natural-language project/correction/constraint/explanation episode remains open.

Nothing here solves broad hypothesis generation, experimental discrimination, quantified
constraint translation, collection semantics, relational analogy, or learned procedure
acquisition. The implemented advance is that explicit specifications and an inspectable
bounded convention can now govern real multi-step work, with honest failure boundaries.

## Verification of this milestone

- `.venv/bin/python -m pytest -q`: **1,398 passed, 5 skipped**, 104.92 seconds.
  This includes regressions for partial-mutation retries, capability-model drift,
  contradictory legacy models, malformed shared observations, and occupied-path
  type conflicts. File and directory creation establish `path_exists` and require
  a vacant target, preventing an impossible file/directory plan before mutation.
- `uv build --wheel --out-dir <temporary directory>` succeeded. The built wheel
  contains both packaged declarative resources: project refinements and request
  conventions. No build artifacts were added to the repository.
- Documentation links resolve and `git diff --check` passes.
- Historical cognitive assay results remain unchanged; these are implementation,
  integration, adversarial, and packaging checks, not a new benchmark scorecard.
