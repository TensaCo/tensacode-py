# 56 — Contingent planning from experience

*Implementation checkpoint, 2026-09-19. This advances the one-step bridge in
[48](48-planning-from-learned-transitions.md) toward the governing objective in
[36](36-structured-cognitive-workspace.md). This is bounded planning over observed finite states, not generalized planning.*

## The capability gap

The subsequent [task integration](57-revision-bound-empirical-tasks.md) adds a
bounded plan/execute loop with persistent in-memory task identity and revision
guards. The four explicitly invoked steps measured below remain the original
component-level acceptance case; the later report measures task pause/resumption.

The earlier one-step experience planner compares supplied actions against an explicit one-step
target. It can retain predictions, guard one execution, and check its observed result.
That interface cannot establish a route whose intermediate states are not themselves the goal.
Empirical model investigation in [55](55-empirical-model-investigation.md) preserves
multiple observed outcomes, but choosing a useful experiment is a different problem
from finding a contingent route through those outcomes.

The new milestone constructs a finite observed transition graph from actual retained
before/action/after evidence. A bounded AND-OR reachability planner then seeks actions
whose **every retained empirical outcome** has a supported route to an explicitly
supplied goal. Its output is a contingent policy, not a promise that an unobserved
outcome cannot occur. Execution applies one freshly checked step, observes what actually
happened, and retains the result before any further planning or action.

## Supplied representation and empirical content

The caller supplies a `StateProjection`, the action set, goal states, exploration schedule,
and sample-support policy. These establish what counts as a state and which distinctions
the planner can represent. The environment's transition table must not be imported as a
model. Edges must instead be supported by actual reset/step trials and retained receipts
and observations, with their sample identities inspectable.

This is empirical graph construction in an authored representation. It does not learn
state abstraction, infer a goal from language, discover action meanings, or form a visual
scene graph. Distinct raw situations can collapse into the same projected state; that is
a representation limitation, not evidence that the situations are interchangeable.
Observed outcomes may be incomplete even when every configured action has been sampled.

State identity and terminal status must be preserved precisely enough that a terminal
failure cannot masquerade as a traversable intermediate state. Unobserved actions or
insufficiently supported edges remain unknown. Missing support must not become an invented
identity transition, a deterministic majority edge, or a reachability proof.

## Search and execution boundaries

The planner accounts for alternative observed outcomes rather than choosing a favorable
branch. Its resource bounds and unresolved frontier remain explicit. A cycle cannot count
as a successful finite route merely because it revisits a previously considered node.
Depth/work exhaustion means that this bounded search did not establish a policy; it is
not a proof that the real environment has no solution.

Planning and execution remain separate. A plan records its evidence, projection/model
identity, initial observation, target, and supplied policy. Before one selected step is
applied, the agent checks fresh observations and the current capability/model binding.
The applied receipt alone cannot certify arrival at the goal. Unexpected outcomes or
changed evidence require an explicit unresolved/replanning result, not continued dispatch
from a stale assumed state. Reusing a consumed step proposal must not repeat its action.

The existing task ledger is in memory and retains goal revisions, attempts, and receipts.
This milestone uses a separate empirical proposal/assessment API: the measured run creates
no task-ledger entries. Four steps require four explicit fresh replanning/execution calls.
It does not demonstrate automatic cross-turn task orchestration, restart recovery, or
revision propagation to dependent beliefs and plans.

## API and search contract

`StateProjection(name, state, provenance)` supplies typed primitive/tuple state identity;
booleans and integers remain distinct. `fit_dynamics` takes actual transitions plus disjoint
training/evaluation attempt IDs and a `DynamicsPolicy`. Default eligibility requires at
least two training and one evaluation sample **for each observed successor**. Minority,
training-only, and evaluation-only successors are retained; insufficient support for any
successor makes that edge unavailable rather than deleting its difficult outcome.

The planner's `plan(model, current_state, goal_state, calls, max_depth=12,
max_states=10000, max_edges=100000)` expands reachable states forward and establishes
finite goal-reaching ranks backward. Every empirical successor of a chosen call must have
a lower rank. A self-loop plus a goal outcome cannot establish finite worst-case arrival.
Equal shortest first actions remain tied, with no implicit first-action selection.
Hard state/edge-budget exhaustion suppresses action choices; a depth-limited failure is
`no_supported_policy_within_bounds`, not a proof of real-world impossibility.

The agent bridge is:

```python
proposal = agent.propose_empirical_plan(model, observation_source_id, calls, goal_state,
                                        max_depth=4)
execution = agent.execute_empirical_plan(proposal.id, call=explicit_best_first_call)
```

An explicit call must belong to the retained best first-action set. Each proposal is
consumed once. Each later step needs a new proposal from the latest successful observation.
Fresh guards recheck model evidence, projection identity, provider/capability binding, and
raw observation before dispatch. A regression exposed a capability callback that could
unmount its provider while returning the old capability description. The final guard now
rechecks the mounted provider identity after callbacks, before invocation; the matching
regression test requires zero action execution. Intermediate supported observations return
`step_observed_replan_required`; only an actual goal observation verifies the goal.

Evidence validation reproduces original fitted rows and accepts a larger retained history.
A later transition from the same provider with the same modeled state/full call but an
unseen successor blocks further model use until refitting. Different states, actions, or
providers do not become counterexamples to an unrelated edge. Unmodeled calls such as reset
are filtered before projecting their state. This assumes the provider's history shares the
model's context; hidden-context classification is not inferred here.

## Actual FrozenLake measurement

[The report](../../eval/results/empirical_planning.json) is reproduced with:

```sh
.venv/bin/python -m eval.learning.evaluate_empirical_planning
```

The shared fixture uses Gymnasium FrozenLake with the authored map `SFF / FFF / FFG`,
`is_slippery=False`, and reset seed zero. The goal is explicitly `(8, True, False)`:
discrete observation eight, terminated, not truncated. Terminal status is represented by
the supplied projection; the planner does not infer it from reward or numeric state ID.
This hole-free deterministic map does not measure stochastic risk or visual perception.

Eight authored prefixes reach the nonterminal states. For each prefix, four possible actions
are sampled in three separate reset/step trials. Only the final prefix/action transition
is assigned to fitting: 64 training and 32 evaluation attempts, disjoint by ID. Prefix
traversal and reset actions supply access to the sampled states but are not fitted edges.
The resulting graph contains nine states and 32 eligible state/action edges, each with
one observed successor supported by exactly two training and one evaluation attempt.
These counts cover this configured fixture, not all future environment possibilities.

The explicit route `(2, 2, 1, 1)` is absent from all 96 complete exploration episodes.
It means right, right, down, down in this supplied environment. The caller chooses these
actions among retained shortest ties; the planner establishes support for those choices
but does not learn this route preference. The observed states are:

| Step | Remaining empirical depth | Actual state after the action | Result |
| --- | ---: | --- | --- |
| Right | 4 | `(1, False, False)` | Replan required |
| Right | 3 | `(2, False, False)` | Replan required |
| Down | 2 | `(5, False, False)` | Replan required |
| Down | 1 | `(8, True, False)` | Goal observed |

The initial plan examines nine states and 32 predictions; subsequent plans examine
9/28, 6/12, and 4/4 states/predictions respectively under shrinking depth limits. State
and edge caps remain 10,000 and 100,000. The report retains all explored policy nodes,
edges, sample identities, exploration episodes, receipts, and actual observation sources.

All eleven audit checks pass: novel complete episode, disjoint splits, edge/support
eligibility, exactly four applied steps, action-free proposals, blocked replays, observed
goal, non-goal intermediate observations, no added beliefs, and no invented capability
effects. No task was created. A novel complete episode composed familiar edges; this is
not generalization to unseen states/actions or a language-derived task.

The final report records Gymnasium 1.3.0. All twelve recorded source hashes agree after
the run and still match the final runtime. Setup took 0.807 seconds; the four
proposal/execution/replay-check cycles took 340.83, 395.66, 400.03, and 345.12 milliseconds.
These shared-host timings include evidence validation and copying and are descriptive,
not a throughput comparison.

The report was rerun after final source freezes: first for unrelated reset observations
being filtered before projection, then for clearer no-action plan reasons and the final callback/provider-identity guards, including shared one-step bridge guards. Earlier runs
passed their own checks but are not the final runtime evidence. The fixed route, sample
schedule, policy, and successful behavior were unchanged; no quality-based tuning selected
the final run. Checkpoint-wide verification is appended after the full suite.

## Limits and acceptance scope

Pure planner tests exercise branching support, unsuccessful branches, cycles, terminal
fixtures, ties, and hard budgets. Integration tests cover actual Gym execution, horizon
failure, stale world/capability evidence, foreign-workspace samples, and replay prevention.
The actual map has only deterministic successors; branching conservatism is separately
established with authored transition fixtures, not stochastic Gym performance.

State meanings, goals, actions, exploration, support thresholds, and tie choices remain
supplied. The graph records empirical transitions; it is not a discovered abstraction or
exhaustive causal model. No generalized language, vision, or cognition claim follows from
solving this finite supplied-state environment.

## Checkpoint verification

The final frozen implementation passed the full repository suite: **1,990 passed,
five skipped in 280.14 seconds**. The focused empirical model/planner/agent tests
cover 44 cases; the earlier planning/investigation guard suites passed 39 tests,
including provider unmounts during the final capability callback and after action.
The measured FrozenLake audit passed all eleven checks, with all twelve recorded
source hashes matching the implementation at this checkpoint. These checks verify
finite empirical planning and its execution boundary, not generalized cognition.
