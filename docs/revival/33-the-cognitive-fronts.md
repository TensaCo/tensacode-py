# 33 — The cognitive fronts

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
