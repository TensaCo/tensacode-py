# Control and volition

Four faculties a generally-capable cognitive system has and this one did not: a task set you can
put something into and take it back out of, a way of choosing between goals that is not their
arrival order, a reason to stop trying that is not a constant, and a way for a practised skill to
stop being deliberated. Each one was diagnosed, given the smallest honest implementation, and
measured against a prediction stated before the measurement.

The method is the one this project keeps confirming: *examples in, structure and calibration out*.
No training was involved in any of this, and the reason is the twice-confirmed finding that
training moves numbers where the schema has somewhere to put the answer and cannot move them
where it does not. All four of these were missing *places to put an answer*.

New library modules: `src/tensacode/control.py` (task set, arbitration, effort) and
`src/tensacode/chunking.py` (automatization). Both follow the repo's convention that state lives
in the store as claims with provenance, so a half-finished goal survives inspection and
`explain()` can walk it. Measurements: `eval/control/`, results in
`eval/results/control_{resumption,arbitration,effort,chunking}.json`.

---

## 1. Interruption and resumption

**The organ that was missing.** A request could be `new`, `running`, `awaiting`, `done`, `failed`
or `dropped`. There was no state for *set aside*. So when the user said something else while a
question was open, the only available move was `Drop`: the request, and everything it had already
done, was destroyed, and the agent apologised ("Okay, skipping …"). Recovery was the user saying
the whole thing again.

**What it needed.** Almost nothing, and that is the interesting part. The interpreter already
keeps the frame, the program counter and the environment in the store as claims. A suspended
request is therefore not a save/restore problem — suspension is a *state change*. What was
missing was the control-layer decision (suspend rather than drop), a way to refer to a goal that
is not the current one ("carry on"), and the arithmetic for choosing which set-aside goal comes
back.

**Structure added.** `Suspend` and `Resume` intentions with their decoders; `suspended` as a live
status; `suspended_from`, `unanswered_question`, `re_asked_at`, `resumed` on the request;
`control.suspend/resume/task_set/progress_kept` in the library; a `_CARRY_ON` pattern that is
control, not vocabulary ("carry on", "where were we", "as you were", …).

**Measured** (`eval/results/control_resumption.json`; environment ours — a fake shell driving the
real control layer, interpreter and procedures; grader ours — step executions read back off the
store; the "before" arm is a reconstruction of the old control layer, because the assistant tree
is untracked and there is no commit to diff against).

The number is *redone work*: a step of a goal that executes again after already having executed
for that goal. A goal is keyed by the words the user said, so restating a dropped request counts
as the same goal — which is exactly what "drop it and apologise" costs.

| case | arm | goal finished | steps redone |
|---|---|---|---|
| organize, interrupted by a create (designed against) | before, restated | yes | **9** |
| | before, only "carry on" | **no** | 0 |
| | after | yes | **0** |
| ambiguous delete, interrupted by a count (held out) | before, restated | yes | **8** |
| | before, only "carry on" | **no** | 0 |
| | after | yes | **0** |

**Prediction outcome: confirmed.** It could not resume before; it now completes both requests with
no redone steps; the falsifier (resumption requires re-running finished steps) did not trigger.

Two things the measurement showed that the prediction did not ask for:

* restart is not merely slower, it is *differently wrong*: in the before arm the restated "tidy up
  my desktop" surveyed a desktop that now had three items instead of two, because the interrupting
  request had created a folder in it. Resumption keeps the survey it had already done.
* a question re-asked on resumption can go stale. In the two-question case, answering the first
  question moved a file that the second question's options named. Nothing re-validates a
  suspended question's options when it comes back. That is a real gap, and it is new — before,
  there was nothing to go stale.

## 2. Priority arbitration

**The organ that was missing.** Intentions carried a static `priority` number. Nothing represented
urgency, the cost of switching, the cost of abandoning, or a deadline; and the order of work was
the order of arrival.

**Structure added.** `control.Stance` (stickiness as hysteresis, aging against starvation, a cost
weight), `weigh`/`urgency`/`score_goal`/`arbitrate`, and `Choice`, which records what was chosen,
what it scored, and what it beat — written into the store as `goal:chosen_because`.

**Measured** (`eval/results/control_arbitration.json`; synthetic goal sets, ours; the assistant
conversation for the second arm). One unit of work per tick to whichever goal the policy picks.
"Today" is the rule the assistant actually used: oldest first, run to completion, never switch.

| goal set | policy | switches per completed goal | mean latency | worst latency |
|---|---|---|---|---|
| cheap-and-urgent vs expensive-and-important (designed against) | today | 0.0 | 19.5 | 20 |
| | arbitrate | 0.5 | **12.0** | 22 |
| six mixed goals (held out) | today | 0.0 | 34.8 | 41 |
| | arbitrate | 0.5 | **13.3** | 49 |
| a stream of cheap arrivals plus one long goal (held out) | today | 0.0 | 11.7 | 25 |
| | arbitrate (aging 0.05) | 0.29 | 6.7 | 43 |
| | arbitrate (aging 0.25) | 0.03 | 11.0 | **27** |
| | arbitrate, no aging | 0.61 | **4.3** | **63** |

**Prediction outcome: confirmed for latency, and the thrash it predicted never appeared.** The
cheap urgent goal's latency falls from 19 ticks to 2. Switching stays well under one switch per
completed goal in every arm.

**What did not pay: stickiness.** Its ablation barely moves anything (0.35 → 0.29 switches per
goal in the stream arm, and nothing at all in the other two), because cost-to-go already provides
hysteresis: a goal's score *rises* as it is worked, so the mind does not want to leave it. The
structure is redundant with a term that was already there. Aging, by contrast, does pay, and the
stream arm is the only place it could show: worst-case latency is 27 / 43 / 63 ticks at aging 0.25
/ 0.05 / 0. Without it the long goal is only finished once the stream of cheap arrivals dries up.

**Two honest limitations, both schema gaps rather than policy bugs.**

* *Urgency is not in the language.* `hear` can weigh a request's cost (steps remaining in its
  procedure, read off the procedure) but nothing in the parser produces value or urgency, so live
  arbitration is cost-first plus aging and nothing else. "Do this first, it's urgent" is not
  representable today.
* *Clause dependency is not in the graph.* Nothing says whether the second clause of an utterance
  depends on the first — "make a folder then put a file in it" is two requests and one dependency —
  so reordering within an utterance is unsafe. Arbitration is therefore confined to *between*
  utterances and between a fresh ask and a set-aside one. This is written into `_pick_fresh` as a
  restriction, not a preference.

A third restriction was found by measurement: a goal suspended while *awaiting an answer* is not
competing for the agent's effort, it is waiting on the user. Letting it compete produced a
Suspend↔Resume live-lock (the fresh ask suspends the question; arbitration prefers the question;
resuming it re-asks; the re-asked question is again suspended by the still-unstarted fresh ask).
Both halves were fixed: arbitration considers only *runnable* goals, and "you said something else
after I asked" is measured from the last time the question was asked rather than the first.

## 3. Persistence versus flexibility

**The organ that was missing.** Retry counts were constants: `MAX_UNSAFE_SUBMITS, MAX_SUBMITS =
3, 6`. An agent should keep trying while the expectation of success exceeds the cost, and the
project already had the machinery to measure an expectation (`expectation.Predictor`, whose
prediction-of-change was measured at 0.79).

**Structure added.** `control.Effort` and `control.should_try_again(history, …)`, which stops for a
reason it can state: already succeeded; the last attempt may already have applied (refused
outright rather than priced, because repeating it could double the effect); refused; the hard cap;
or `p × value ≤ cost`, where `p` is a measured frequency or an explicitly *named* stated prior.
`Verdict("unknown", …)` when there is neither — a mind that does not know should say so.

One correctness fix this forced: predicting the *likeliest outcome* and reading `1 − p` as the
chance of success is only valid with two outcomes. Here there are three (succeeded, transient,
ambiguous), so `observe_attempt` records the binary aspect too and `should_try_again` reads that.

**Measured on the real access app, headless.** Nothing on disk was edited: the raised-failure arms
serve the app's own HTML with its two probability constants rewritten at serve time, and every row
is labelled with the rate it ran at. The safety rails are identical in every arm (at most three
attempts that may have had an effect; never resubmit a request already listed in Recent
submissions — the task's own `tc.Constraint`). Grader: the app's own `window.__score`, i.e. the
environment author's.

| served failure rate | retry rule | items correct | duplicate effects | escalated | actions/episode |
|---|---|---|---|---|---|
| 15% (**as shipped**) | fixed (6 submits, 3 that may have applied) | 40/40 = 1.00 | 0 | 0 | 48.0 |
| | expectation, unconfirmed = may have applied | 40/40 = 1.00 | 0 | 0 | 48.0 |
| | expectation, unconfirmed resolved by looking | 40/40 = 1.00 | 0 | 0 | 48.0 |
| 45% (rewritten) | fixed | 40/40 = 1.00 | 0 | 0 | 50.5 |
| | expectation, may have applied | 39/40 = 0.975 | 0 | 1 | 50.3 |
| | expectation, resolved by looking | 40/40 = 1.00 | 0 | 0 | 50.5 |
| 65% (rewritten) | fixed | 36/40 = 0.90 | 0 | 4 | 53.5 |
| | expectation, may have applied | 38/40 = 0.95 | 0 | 2 | 54.6 |
| | expectation, resolved by looking | **40/40 = 1.00** | 0 | 0 | 54.9 |

10 episodes of 4 tickets per arm, same seeds in every arm, 360 episodes over both runs.

**Prediction outcome: confirmed where the cap binds, and unmeasurable where it does not.** At the
rate the app actually ships with, all three rules score 1.00 on every item and take the same
number of actions: a 6-attempt budget essentially never binds at a 15% failure rate, so the retry
rule *cannot* matter and claiming an improvement there would be claiming noise. At 65% the fixed
budget gives up on 4 items that persistence gets right (0.90 → 1.00) for 1.4 extra actions per
episode, and duplicate effects stayed at **0 in all nine arms** — the rails did their job.

**What did not pay, and it is the more interesting half.** *Expected value by itself is not what
helped.* The strict reading — treat an unconfirmed submit as may-already-have-applied and stop —
is **worse than the shipped constants** at 45% (0.975 vs 1.00) and only better at 65%. All of the
gain comes from the other variable: reading an unconfirmed submit against what the agent already
perceives in Recent submissions, and retrying only when the request is demonstrably not there.
The app itself says so in its toast ("Check Recent submissions before retrying"). The shipped
constants were a good compromise for a mind that does not do that check; expected value is not a
substitute for the check, and with the check the budget hardly matters.

**A failure of mine, recorded because it is instructive.** The first run of this experiment
(`scratchpad/control_effort_noprior.json`, kept out of `eval/results`) gave the expectation policy
no stated prior. `should_try_again` correctly returned `unknown` for the first attempts of each
arm — fewer than three observations is not a calibration — and the adapter treated `unknown` as
"stop". So it gave up on tickets for lack of evidence *about itself*, scoring 0.95 at 45% where
the fixed rule scored 1.00. `Effort.prior` exists exactly for that cold start; the fix was to
state one and name it, not to lower the threshold. An agent that stops because it has not yet
learned how often it succeeds has confused not knowing with knowing it will fail.

## 4. Automatization

**The organ that was missing.** A learned skill replayed step by step forever: every guard
evaluated, every step recorded, on the thousandth run exactly as on the first. That is not
carefulness; it is an inability to learn *how* it does something as opposed to *that* it works.

**Structure added.** `chunking.Chunks`: a path through a procedure that has run identically three
times compiles into a `Chunk` — the steps taken, the guards whose answer never varied, and what
trouble (if any) each step normally shows. Running a chunk skips the deliberation, not the acts: a
click is still a click, a command is still typed. What it skips is asking a guard whose answer has
never varied and writing a decision record per step. The expanded form is never discarded, and a
chunk is retired the moment a step diverges from what the recorded runs saw — which is what makes
the whole thing safe. Opt-in: `Host.chunks = None` is the shipped behaviour, unchanged.

**Measured** (`eval/results/control_chunking.json`; environment ours, grader ours). One skill
replayed 20 times.

| | deliberated steps per replay (settled) | total | claims at end | seconds/replay | distinct replies |
|---|---|---|---|---|---|
| chunking off | 11 | 220 | 2813 | 0.0072 | 1 |
| chunking on | **0** | **33** | **1929** | 0.0049 | 1 |

**Prediction outcome: confirmed.** Deliberated steps per replay go to zero once both the skill and
the procedure it calls have compiled (two chunks: `count`, and the `resolve` it calls); the store
holds 31% fewer claims; the replies are byte-identical with chunking on and off; and the graph
shows one act per run (`frame ran_chunk <shape>`) instead of one per step. Wall time falls to
0.68× — read as a ratio only, since the machine was running other work.

**The falsifier was tested, not assumed.** One step of the skill was made to fail at replay 10,
after the chunk was live:

```
replay  9:  0 steps  ~/Desktop/notes.txt has 2 words.
replay 10:  3 steps  I couldn't count: wc: cannot read input: Input/output error
replay 11:  8 steps  ~/Desktop/notes.txt has 2 words.
```

The chunk retired ("errors, which the recorded runs never had"), the run finished by deliberating
from that point, the reply was the procedure's own honest error message — **identical to what the
same failure produces with chunking off** — and the following replays deliberated in full until
three identical runs earned a new chunk. Recovery did not degrade.

**Two costs found while building it, both worth knowing.**

* *Divergence is only visible where the procedures already mark it.* The first version treated
  `error_summary` as a trouble mark; that binding is always populated (it falls back to the first
  line of a perfectly good output), so every chunk would have retired on first use. It also missed
  trouble entirely for steps that name their results (`"as": "sib"` binds `sib_ok`, `sib_errors`).
  Both are fixed, but the general point stands: an automatized skill inherits the blind spots of
  the error detector it relies on. A failure that does not *look* like an error to
  `output_facts` — a command whose message the error regex does not match — is invisible to the
  fallback, and the chunk keeps running.
* *A step that always fails is normal for that step.* "Stat the folder before creating it" expects
  "no such file". So divergence cannot mean "an error happened"; it has to mean "different from
  what the recorded runs saw", in both directions — trouble where there was none, and no trouble
  where there always was.

---

## Which of the four would be cut

**Priority arbitration**, in its current form. Its latency win is real, but in the live assistant
it is doing very little work: the server is strictly turn-based, so a running request cannot be
interrupted at all, and the only competition that arises is between set-aside *questions*, which
are waiting on the user rather than on the agent. Its two inputs are also mostly absent —
urgency is not in the language, and clause dependency is not in the graph, which forbids exactly
the reorderings that would make it valuable. What is worth keeping from it is the small part that
pays for itself: aging, which bounds how long a goal can be passed over. The rest is a policy
waiting for a representation.

The other three earn their place. Resumption is the one that changes what the assistant *can do*
rather than how well it does it. Automatization is the largest measured effect for the least
behavioural risk, with a tested fallback. Effort is the one whose result was most surprising, and
the surprise is worth more than the structure: the shipped constants were better calibrated for
this app than the expectation policy that replaced them.
