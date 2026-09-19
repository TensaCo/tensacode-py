# 33 — The cognitive fronts

**Visual scope correction, 2026-09-19:** the frontier includes constructing and revising
holistic scene interpretations. Preserve global layout, grouping, relational visual graphs,
events, affordances, and competing explanations alongside local evidence. Element recognition
is one input to that process. [36](36-structured-cognitive-workspace.md) requires relational
acceptance cases where identical element inventories imply different actions because their
organization differs. Graphs supplied by tests and graphs inferred from unfamiliar pixels
must be reported separately; neither record storage nor classifier accuracy closes this front.

> **Objective update, 2026-09-19:** [36 — The structured cognitive workspace](36-structured-cognitive-workspace.md)
> is the current direction. The fronts below must converge on an agent that constructs and
> revises the structured problem it reasons about, starting from language and images.
> Stronger downstream planning cannot compensate for an early, incorrect interpretation.
> Read 36 for the shared objective, implementation sequence, and behavioral acceptance gates.

> **2026-09-19 architectural reassessment:** the dated probes below remain historical
> evidence. The prescriptions “mostly plumbing,” “pure wiring,” and “six … downstream of a
> plan being a sequence” are superseded by the assessment at the end and
> [34 — Cognitive primitives and implementation](34-cognitive-primitives-and-implementation.md).
> A representable value or callable solver is not yet an adequate cognitive model.
> The [first implementation slice](34-cognitive-primitives-and-implementation.md#first-implementation-slice-2026-09-19)
> adds structured task identity, precondition checks, and loss reporting; the historical
> scores below have not been replaced by a new benchmark run.

Written 2026-09-18 after the owner probed the running agent with four messages and it failed
all four. Three of the fronts are his: **multi-turn, goal decomposition, puzzles**. The rest
are the ones those four failures and the assay's zeros actually imply. The organising question
is not "what would a cognitive architecture have" — the repo already has 6,250 lines of that
nobody calls — but **what does a person doing real work need the agent to hold in its head?**

Each front below says what breaks now, what substrate already exists, and what would count as
evidence. Ordered by value-per-unit-of-work, not by ambition.

## 1. Conversation as its own thing — *started*

`hello` came back as "a phrase that is not a statement, question or request", because acts were
derived from grammatical mood and anything else was a fragment. Fixed for conventional moves
(WordNet says what class an expression is; the adjacency pairs are seeded). What is still
missing is the larger point: **an imperative is not necessarily a request to change the world.**
"explain your reasoning", "list the options", "walk me through it" are requests whose product is
*discourse*. VerbNet already derives `has_information(Topic=…)` for `explain` and `tell`, so
nothing needs inventing in the language layer — what is missing is a capability whose effect is
information. *Evidence*: "explain what you just did" answers from the trace, which already
records every operation.

`describe`, `predict`, `infer` and `answer` have **no result state in VerbNet at all**, so they
are unreachable by construction. Discourse acts need their own account of what they achieve;
that is a schema question, not a lookup.

## 2. Goal decomposition — *the biggest single unlock*

"make a python hello world project" is a folder, a file inside it, and a line of text in the
file. The agent has all three capabilities and did none of it, because a request becomes **one**
goal matched against **one** capability. Nothing in the design forbids more: the planner already
reasons over VerbNet result-state conditions, which is what goal regression needs.

*What it needs*: conditions that are not yet true become sub-goals; a capability's
preconditions become conditions to achieve first; the plan is a sequence, verified step by step.
*Evidence*: the native desktop set grows a multi-step job — a project directory with a file and
contents — graded by the world.

## 3. Multi-turn state

Right now each message is read fresh. Facts persist in the store, but there is no notion of the
conversation itself: no current task, no "do it again", no "no, the other one", no repair of a
misunderstanding. `Context` handles reference within a message only.

*Substrate*: the store holds propositions with valid time and supersession, so "the meeting moved
to Thursday" is representable and nothing uses it. *Evidence*: a two-turn item where the second
turn only makes sense given the first ("put it in documents instead").

## 4. Hypothesis formation and experiment design — *the most valuable for real work*

The owner's third probe: a hidden deterministic machine, five observations, eight experiments
allowed, then infer the rules and say which parts are determined by evidence versus merely the
simplest fit. **This is what debugging is**, and what this session consisted of — measure,
diagnose, change one thing, re-measure. The agent has no machinery for it at all.

*Substrate*: more than it looks. Propositions can carry `modality="hypothesised"`; the store
keeps competing claims without resolving them; `ops.choose` takes an objective and hard
constraints, which is the shape of "pick the experiment that discriminates most". *Evidence*:
give it a hidden function, a budget of probes, and score the rule it induces on held-out inputs —
and separately, score whether it correctly labels which parts of its model are forced.

## 5. Constraint puzzles

Three boxes, three statements, exactly one true. Requires enumerating candidate worlds and
eliminating those that violate a stated constraint. *Substrate*: n-ary propositions with
polarity and modality can state each box's claim and the meta-constraint; what is missing is
anything that searches over assignments. *Evidence*: a small suite of stated-constraint puzzles
with unique answers, plus the requirement that it **shows the elimination** rather than
guessing the answer.

## 6. Quantity

`reasoning.gsm8k` is 0/12 and "how many" questions abstain. No counting, no sums, no units, no
comparison. *Substrate*: `quantity.py` exists and is unreachable from the agent — one of the
9,117 unreached lines, and the first candidate for adoption rather than removal.

## 7. Clarification instead of refusal

`pragmatics.ambiguous` is 0/12: it declines where it should ask. **The metric is gameable** — a
question mark scores — so the work here is graded-confidence deferral with the *question* naming
the actual ambiguity, and a judge that checks the question is about the right thing.

## 8. Constraints held across a task

"don't touch anything outside this folder", "only the .txt files". Negative and scoped
constraints, held while acting. The safety invariant today is blunter: it changes nothing at all,
which is why 84/84 looks perfect and means little.

## 9. Learning inside the conversation

"no, I meant the other file" → repair and retry. "a widget is a kind of gadget" → the word works
afterwards. `learning.in_use` is registered with a kill criterion (recover ≥2 of 5 deleted
constructions) and has no data yet.

## 10. Analogy

"do the same for the other folder" — reapplying a plan with substituted arguments. Cheap once
goal decomposition exists, because the plan is then a value that can be re-bound.

## What this changes about the order of work

The referring failures and the `knowledge-as-code` list stay worth fixing, but they are
maintenance. The fronts that change what the thing *is* are **2 (decomposition)**, **1
(discourse acts)** and **4 (hypothesis and experiment)**, and the first two are mostly
plumbing over machinery that already exists. 4 is a research bet and should be treated like the
learning phase: a pre-registered prediction and a kill criterion, not an open-ended build.

The honest framing for all of them: today the agent answers **8.7%** of what it is asked and
beats a do-nothing control on four measurements out of thirty-four. Every front above is a way
of raising coverage *without* raising the wrong-answer count, and any front that raises coverage
by guessing has made things worse.

---

## Status, 2026-09-19

Measured, not claimed. Probes are single messages to the full configuration
(`desktop + vision + quantity + self`, treebank reader); task numbers are dev-split rows in
`eval/results/assay.jsonl`.

| # | front | state | the evidence |
|---|---|---|---|
| 1 | discourse acts | **works, narrowly** | "what are your capabilities?" lists the real capability set; "explain your reasoning" answers from the recorded trace. Several phrasings are still unreachable: "what can you do?", "explain what you just did", "what do you know about X". |
| 2 | goal decomposition | **not done** | "make a python hello world project" → *what I can do brings that about only for directory or file*. Still one goal, one capability. `cognition.multi_step` 0.143 (1/7, and that 1 is a decline item). |
| 3 | multi-turn | **half** | reference across turns works — "make a folder called notes" then "put readme-first.txt in it" puts the file in the folder. Task state does not: "do it again" → *it didn't ask me to do or answer anything*. `cognition.multi_turn` 0.125. |
| 4 | hypothesis & experiment | **not started** | nothing exists. |
| 5 | constraint puzzles | **machinery only** | the solver answers three boxes uniquely with an elimination trace, and no English ever reaches it. `cognition.constraint_puzzles` 0.000. |
| 6 | quantity | **machinery only** | the plugin sums amounts and cancels units when handed a frame it can read, and "Shondra has 3 plants / 4 plants / how many plants?" still abstains, because the reader drops the wh-phrase's noun and `to_propositions` flattens the count into `Ref("entity:…")`. GSM8K 0 correct / 0 wrong / 12 abstained. |
| 7 | clarification | **not done** | "delete the file" → *there is nothing called 'file' under home*, where the right move is to ask which file. |
| 8 | held constraints | **not done** | "tidy my desktop but don't touch anything in documents" → *no VerbNet class for 'tidy'*, and the prohibition was read as a **touch request** with negative modality. `held_constraints` 0.167 accuracy with 1.000 on prohibitions — every prohibition respected because nothing was done. |
| 9 | learning in conversation | **not done** | told "a widget is a kind of folder", it still cannot make a widget. Being told does not extend the taxonomy. |
| 10 | analogy | **not done** | "do the same for b" → not understood. |

**The finding that matters most.** Mounting the new plugins changed **no measured number**:
`cognition.*` and the native desktop set score identically with and without `quantity` and
`self`. Four fronts have machinery and two of them (5, 6) have no path from English to that
machinery at all. Building the capability and connecting it are separate jobs, and only the
first is done.

Overall: it says something to **9.3%** of what it is asked (23/247 over the tasks with rows),
with **1 wrong answer** — and the 94 "correct" are mostly tasks where correct means *changed
nothing*.

**What the probes say to do next**, in order: (2) decomposition, because six of the ten
fronts are downstream of a plan being a sequence rather than a single capability; then the
English→machinery joints for (5) and (6), which are pure wiring over parts that already work;
then (7), which is the cheapest real gain in coverage and the only one that trades against the
zero-wrong invariant, so it needs the graded-deferral judge before the feature.

## Architectural reassessment, 2026-09-19

The failure inventory remains useful. Its explanation of the missing work was too narrow.
This reassessment is based on architectural inspection, not a rerun of the numbers above.
In particular, the substrate bullets describe storage possibilities and isolated mechanisms;
they do not establish sufficient semantics, a language path, or integrated behavior.

| Front | Distinction the initial prescription misses | Revised requirement |
|---|---|---|
| 1. Discourse | Information production versus a communicative achievement; an operation trace versus a reason | Represent the question addressed, communicative purpose, commitments, and response adequacy. An explanation must connect a choice to the task and evidence. |
| 2. Decomposition | Goal interpretation versus domain refinement versus action search | Supply explicit task success criteria, refinement knowledge, and action preconditions. A generic planner cannot invent what counts as a Python project. |
| 3. Multi-turn | Persistent facts and references versus an enduring task | Identify tasks, pending questions, alternatives, commitments, and corrections to interpretations. “Instead” edits a task; “again” needs a referent and scope. |
| 4. Hypothesis and experiment | A hypothesized proposition versus a predictive model | Declare candidate families, predictions, exclusion rules, discriminating probes, and stopping criteria. “Forced” is relative to assumptions and a model class. |
| 5. Constraint puzzles | Solving formal constraints versus constructing the right formal problem | Preserve quantification, logical scope, completeness assumptions, and the distinction between a statement and its truth. |
| 6. Quantity | Arithmetic versus the objects and changes being counted | Model collections, membership, overlap, ownership, measurements, rates, and temporal changes before choosing an operation. |
| 7. Clarification | Low confidence versus a useful question | Name alternatives and determine whether an answer is available from the user and could change the decision. Compare asking with inspecting and proceeding. |
| 8. Held constraints | A negative proposition versus a restriction on execution | Give restrictions scope, lifetime, and exceptions; check intermediate states and indirect effects as well as the final result. |
| 9. Learning | Reference repair versus vocabulary acquisition versus belief revision versus learning a procedure | Use distinct updates with evidence, provenance, persistence scope, correction, and transfer tests. |
| 10. Analogy | Rebinding a procedure versus mapping different relational structures | Establish correspondences, applicability, adaptation, and limits. Argument substitution is a useful first case, not the whole faculty. |

N-ary propositions are a useful substrate, but nesting and arbitrary predicate names do not
by themselves establish binding, quantification, time, or attitude semantics. Consumers must
agree on those meanings. Likewise, VerbNet should contribute candidate interpretations;
missing lexical result states should not define the boundary of possible intentions.

The common dependency is **persistent, revisable task state**, together with explicit meaning
at the boundaries between interpretation, planning, and action. A sequence is one part of a
plan, not the foundation of six faculties. Quantity and constraint-language integration are
substantive modeling work, not established to be pure wiring. The dated prohibition scores
also demonstrate why observed zero errors must not become an objective that rewards inactivity.

The proposed next episode combines project creation, a destination correction, preservation
of an existing file, world-state verification, and explanation of the revised plan. It must
exercise shared structures through the agent. See [34](34-cognitive-primitives-and-implementation.md)
for the staged implementation and evaluation contract; this paragraph does not claim that
episode works today.

## Implementation update, 2026-09-19

[35 — Model-based planning and inspectable refinement](35-model-based-planning.md) records
new implementation evidence for parts of fronts 2, 3, and 8: generic bounded planning,
real filesystem execution, caller-driven revision/resumption with stable task identity,
step histories, and held conditions checked against models and observations. The ordinary
request “make a python project called hello” now reaches a hand-authored project refinement
and verified multi-step filesystem work. The original “make a python hello world project”
remains a compound/valency interpretation failure in both inspected parser routes.

These are different input paths, not interchangeable evidence. Natural-language correction,
full conversation state, continuous temporal restrictions, and the broader cognitive fronts
remain open. No historical scores above were rerun or replaced by this implementation note.
