# 18. How we measure general cognitive performance

> **Pixel pipeline retirement:** the fixed geometry/control/prompt semantic path,
> pixel providers, and associated end-to-end/fusion runners described in these
> historical results are retired. Saved measurements are not current runnable
> capabilities. See [70 — Retiring pixel semantic rules](70-retiring-pixel-semantic-rules.md)
> for the retained components and present vision limits.

Before this, the answer was: we did not. There were task scores with uneven provenance and no
common frame. This is the frame — a standing harness (`eval/profile/`) that recomputes from
scratch and writes `eval/results/cognitive_profile.json`, which generates this document.

**There is deliberately no single score.** A single number is exactly what invited the problems
the evidence audit found (§11): it can be raised by choosing an environment, a grader, or a
threshold. So every row below carries its floor, its control, who wrote the environment, who
graded it, whether the test was fixed before tuning, and its source file. Accuracies are
luck-corrected where the chance credit is known. An axis that cannot be grounded says so
rather than reporting a zero or quietly disappearing.

**9 of 10 axes are grounded.** Ungrounded: `sample_efficiency`.

## At a glance

| Axis | Question | Standing |
|---|---|---|
| `transfer` | Does a capability survive moving to material it was not built on? | partly grounded |
| `horizon` | How does success scale with the number of sequential decisions? | well grounded |
| `sample_efficiency` | How many examples does a new skill take? | **not grounded** |
| `calibration_and_abstention` | When it does not know, does it know that? | well grounded |
| `compositionality` | How far does accuracy hold as clauses are chained in one message? | partly grounded |
| `world_modeling` | Can it predict what its own actions will do? | partly grounded |
| `belief_revision` | Does a new fact replace an old one, and does the world override memory? | partly grounded |
| `grounding` | Is an answer traceable to something that actually supports it? | partly grounded |
| `robustness` | Does it hold up under perturbation? | partly grounded |
| `cost_and_evidence_integrity` | What does it cost, and who graded it? | well grounded |

Read that column as *how much the evidence is worth*, not as how well we score. `horizon` is
well grounded and the news there is bad; `world_modeling` is partly grounded and the news is good.

**Where we are strong:** verifiable structured automation — long runs of short-horizon decisions
with zero model calls, checked against a simulator's own state; belief revision that re-perceives
rather than trusting memory; prediction of our own actions' effects, graded by the environment.

**Where we are weak:** open-domain knowledge and answer projection (§12, §13), genuinely long
tasks (nothing completes one), capability-aware abstention, and evidence integrity — most headline
rows are still graded by code we wrote in environments we wrote.

**Unmeasured:** sample efficiency as a rate (only two skills have ever been learned), and causal
discrimination (the intervention machinery runs but has not yet separated cause from co-occurrence).

## transfer

*Does a capability survive moving to material it was not built on?*

Hand-written perception degrades gradually; a learned component transferred best when its training data was widened deliberately; and a fix validated in one environment did not survive a change of engine. Transfer is measured, and it is the axis where our claims have most often been too strong.

**visible text read, hand-written perceiver: tuned apps -> unseen apps -> unseen OS**

- value: 0.885 -> 0.766 -> 0.709
- n = 3723
- floor: n/a (recall of a known word list)
- control: two ablations run (no icon memory, no text candidates)
- environment author: us · grader: our own code · agent can influence grader: no
- held out: apps and OS split before the rules were written; some rule edits came after held-out inspection (audit: perception-fusion)
- source: `eval/results/vision_perception.json`
- note: degrades monotonically as the environment moves away from where it was tuned

**clicking the right control by name, hand-written perceiver: tuned -> unseen apps -> unseen OS**

- value: text 0.731 -> 0.738 -> 0.724 | icon 0.746 -> 0.241 -> 0.085 | field 0.816 -> 0.378 -> 0.094
- n = 1529
- floor: wrong-click rate reported alongside (text 0.106 on unseen apps)
- control: three ablations; DOM perceiver as the reference arm
- environment author: us · grader: our own code · agent can influence grader: no
- held out: apps and OS split before the rules were written
- source: `eval/results/vision_perception.json`
- note: text-labelled controls transfer (0.74 -> 0.72); icons and outside-the-box field labels do not (0.24 and 0.09 on unseen OS)

**learned role model, textbox recall: trained apps -> unseen apps -> unseen OS**

- value: 1.000 -> 0.824 -> 0.831
- n = 6199
- floor: majority class 0.961 (which the model is BELOW overall on unseen OS: 0.874)
- control: majority baseline per split
- environment author: us · grader: our own code · agent can influence grader: no
- held out: apps/OS held out before training
- source: `eval/results/learned_role_model.report.json`
- note: the clearest training win in the repo, and it states its own cost: better fields, worse overall accuracy than guessing the majority on unseen OS

**OCR fine-tune, exact word accuracy on unseen FONTS (narrow corpus vs broad)**

- value: baseline 0.668 | apps-only 0.658 | apps+rendered 0.794
- n = 2274
- floor: n/a
- control: three corpora compared on one held-out split
- environment author: us · grader: our own code · agent can influence grader: no
- held out: 2 fonts and 8 apps never trained on
- source: `eval/results/recognizer_comparison.json`
- note: training on the narrow corpus went BELOW baseline on unseen fonts; only the broad corpus transferred

**the same OCR fix moved to a different engine (episodes verified, debugged seeds vs held out)**

- value: engine scene (reference): A 10/10 B 10/10 | pixels, stock recognizer: A 7/10 B 5/10 | pixels, apps fine-tune: A 7/10 B 4/10 | pixels, apps+rendered fine-tune: A 7/10 B 5/10 | pixels, apps+rendered + task invariants: A 4/10 B 5/10
- n = 20
- floor: n/a
- control: two splits, one never inspected while fixing; three recognizers compared
- environment author: computerworld (third party) + our task · grader: our own code · agent can influence grader: no
- held out: split B never looked at while writing the fixes
- source: `eval/results/cw_pixel_e2e.json`
- note: the 2->0 false-success gain from the fine-tune did NOT travel: all three recognizers make the same three errors here, because the failure mode changed (lost hyphens, not digits)

## horizon

*How does success scale with the number of sequential decisions?*

The hand-written objective holds ~0.986 across ~103k episodes of 20-45 decisions; a cloned policy at 0.79 per decision collapses to ~1% of episodes. Nothing we have completes a genuinely long, open task: the long-horizon suite scored 0, and its runs were throughput-bound rather than capability-bound.

**hand-written objective, items correct over long continuous running**

- value: 290909/296674 = 0.9806 over 102899 episodes (live_run7); other runs: live_run6 0.9857 over 3447 episodes
- n = 296674, 95% CI [0.9801, 0.9811]
- floor: none run (no random-policy control on these apps)
- control: none run
- environment author: us · grader: our own code · agent can influence grader: no
- held out: live-wall seeds differ from bench seeds; apps randomize themselves
- source: `eval/results/browser_agents_recovered.json`
- note: strongest capability number we have, and both environment and grader are ours (audit verdict: weak evidence)

**learned ranker vs hand-written objective in closed loop**

- value: access: hand 1.00 vs learned 0.05 (mean actions 46) | shop: hand 1.00 vs learned 0.50 (mean actions 6) | recon: hand 1.00 vs learned 0.00 (mean actions 24) | chart: hand 1.00 vs learned 0.00 (mean actions 5)
- n = 195
- floor: hand-written objective as the reference arm
- control: same seeds both arms
- environment author: us · grader: our own code · agent can influence grader: no
- held out: seeds 9001+ not used in training
- source: `eval/results/learned_intention_ranker.json`
- note: 0.79 per-decision accuracy over 20-45 sequential decisions leaves ~1% of episodes intact: the horizon multiplies errors, so per-decision accuracy near 0.999 is what these tasks demand

**long-horizon tasks with hidden graders (CAD / EEG / coding)**

- value: 0/8 passed
- n = 8
- floor: 0 expected from an agent with no such skills
- control: hidden test sets the agent cannot read
- environment author: us · grader: hidden tests (independent of the agent) · agent can influence grader: no
- held out: tasks authored before the runs; one held-out split
- source: `eval/results/longhorizon.json`
- note: runs were throughput-bound: both arms hit a 30-minute cap after ~9 model calls at ~200 s per call, so this bounds nothing about the architecture yet

## sample_efficiency

*How many examples does a new skill take?*

**Not grounded.** n=2 skills in total: enough to show the mechanism runs, far too few for a rate. The teacher model is not running, so no new learning could be measured now.

The one-demonstration mechanism works and has a real adoption gate, but with two skills ever learned this axis is a demonstration, not a measurement.

**skills learned from one demonstration, and whether they were ever reused**

- value: 2 learned (1 adopted, 1 still on trial); 1 total reuses
- n = 2
- floor: n/a
- control: trial->adopted requires success on a request it was not learned from
- environment author: us · grader: our own code · agent can influence grader: no
- held out: adoption requires different slot values
- source: `skills-practical-before.json, skills-replay.json`
- note: one demonstration per skill is the design; the sample is far too small to quote a rate

## calibration_and_abstention

*When it does not know, does it know that?*

Abstention exists, is measurable, and refuses rather than inventing. But it is keyed to evidence strength rather than to capability: on public data the cheap tier is worse than the model precisely where it commits, it refuses 54 answerable questions, and its answers are net negative on SQuAD 2.0. False successes are the one error type we have reduced to zero, and structure (declared invariants) did that, not training.

**SQuAD 2.0, rules arm: accuracy over attempted, and what is left after chance**

- value: 0.0909 raw; 0.0636 after removing chance credit
- n = 55, 95% CI [0.0395, 0.1958]
- floor: always-abstain scores 0.48 overall; the arm scores 0.42
- control: random-span floor 0.0133; model arm on the same items
- chance credit: 1.5 of 5 correct answers were expected by chance
- environment author: public dataset (SQuAD 2.0) · grader: public labels · agent can influence grader: no
- held out: disjoint 150-item calibration and 300-item test slices
- source: `eval/results/open_domain.json, eval/results/schema_brittleness.json`
- note: the arm's answers are net negative: refusing everything scores higher than answering

**the two error types kept apart, on SQuAD 2.0**

- value: 54 wrongful refusals vs 23 answers to unanswerable questions
- n = 300
- floor: n/a
- control: n/a
- environment author: public dataset · grader: public labels · agent can influence grader: no
- held out: test slice
- source: `eval/results/schema_brittleness.json`
- note: miscalibrated in both directions, so abstention is not simply conservative

**HotpotQA: does the cheap tier beat the model on the items it chose to answer?**

- value: rules 0.0584 vs model 0.4927 on the same 274 items
- n = 274, 95% CI [0.0363, 0.0927]
- floor: chance ~0.007 for a free-form span
- control: model arm restricted to exactly the items the rules attempted
- environment author: public dataset (HotpotQA) · grader: public labels · agent can influence grader: no
- held out: 300-item test slice
- source: `eval/results/open_domain.json`
- note: the abstention is capability-blind: where it commits, it is far worse than the tier it was meant to protect

**false successes (claiming a result that did not happen), pixels only**

- value: engine scene (reference): 0+0 false, 0+0 refusals | pixels, stock recognizer: 3+0 false, 0+0 refusals | pixels, apps fine-tune: 3+0 false, 0+0 refusals | pixels, apps+rendered fine-tune: 3+0 false, 0+0 refusals | pixels, apps+rendered + task invariants: 0+0 false, 3+1 refusals
- n = 20
- floor: the structured-scene arm has 0 false successes on both splits
- control: four arms, two splits, one never inspected while fixing
- environment author: computerworld (third party) + our task · grader: our own code · agent can influence grader: no
- held out: split B held out
- source: `eval/results/cw_pixel_e2e.json`
- note: task-declared invariants remove them (pixels, apps+rendered + task invariants); the recognizer fine-tunes did not

**task-declared invariants on the recorded failure frames**

- value: baseline: 1 refused, 7 corroborated correctly, 0 corroborated wrongly, of 11 | apps+rendered fine-tune: 1 refused, 7 corroborated correctly, 0 corroborated wrongly, of 11
- n = 11
- floor: n/a
- control: checked on the frames that had already failed, including the false-success frame
- environment author: computerworld (third party) + our task · grader: our own code · agent can influence grader: no
- held out: frames recorded before the invariants were written
- source: `eval/results/invariant_eval.json`
- note: it refuses the frame that produced a false success and corroborates nothing wrongly; n=11, so this is a demonstration rather than a rate

## compositionality

*How far does accuracy hold as clauses are chained in one message?*

Composition itself holds: four chained clauses in one message were carried out correctly, each verified against the machine. The 5-clause case fails on a broken act, not on depth. The more serious finding is recovery: one failing clause poisons the rest of the conversation permanently.

**clauses carried out correctly, by chained-clause depth in ONE message**

- value: depth 1: 1/1 | depth 2: 2/2 | depth 3: 3/3 | depth 4: 4/4 | depth 5: 4/5
- n = 15, 95% CI [0.7018, 0.9881]
- floor: n/a (each clause is checked against the machine's own state)
- control: the same five clauses, measured at every depth
- environment author: us · grader: the simulator's shell, read by a separate process · agent can influence grader: no (it cannot see or change the checks)
- held out: fresh names per run; the ladder was written before it was run
- source: `eval/profile/probes.py, run against the live assistant`
- note: 14/15 clauses; the only failure is the 5th clause, which is a broken `copy` act rather than a composition failure

**does a single broken clause end the conversation?**

- value: yes: after the failing clause, every later turn in that conversation also fails
- n = 1
- floor: n/a
- control: the same later turns succeed in a fresh conversation
- environment author: us · grader: observed replies · agent can influence grader: no
- held out: n/a
- source: `eval/profile/probes.py; reproduced three times`
- note: the crashed request stays selected forever, so there is no recovery short of a restart

## world_modeling

*Can it predict what its own actions will do?*

Prediction is the best-graded thing in the whole profile: the environment scores it, we do not, and it improves with experience (0.745 -> 0.841 within one run). The framing matters more than the mechanism: predicting CHANGE works (~0.79) while predicting absolute values fails (~0.06), which is a statement about representation, not about effort. The causal half is NOT yet demonstrated: intervention found 14 links against correlation's 13, every aspect was judged 'caused', and the case designed to show a co-occurring signal being rejected produced a zero effect in both framings. Prediction: strong. Causal discrimination: unproven.

**predicting the screen BEFORE acting, then scoring the prediction**

- value: best arm (change/action+state) 0.7934; first half 0.7453 -> second half 0.8411
- n = 213, 95% CI [0.7341, 0.8423]
- floor: the absolute-value framing scores 0.0563 on the same steps
- control: four framings compared (absolute vs change, with and without state)
- environment author: computerworld (third party) · grader: the engine's own next state · agent can influence grader: no
- held out: predictions are made before the action is taken
- source: `eval/results/structures_expectation.json`
- note: ground truth is free and exact here; accuracy rises with experience, and it refuses to predict where it has too few trials

**telling cause from correlation using interventions**

- value: 14 causal links from active controls vs 13 from correlation alone; every aspect tested came back 'caused'
- n = 25
- floor: correlational baseline finds 13 links from 14 observations
- control: label shuffle: 12 links, 2 misattributed
- environment author: computerworld (third party) · grader: the engine's own state under fork/restore · agent can influence grader: no
- held out: held-out interventions
- source: `eval/results/structures_causal.json`
- note: the same state run with and without an action is a real controlled experiment, which the engine's fork/restore makes possible. The clock case, which was meant to show a co-occurring-but-uncaused signal being rejected, did NOT demonstrate it: the clock never entered the correlational links (clock_in_correlational_links=False) and BOTH framings measured an effect of 0.0, so there is no non-zero contrast here to speak of

## belief_revision

*Does a new fact replace an old one, and does the world override memory?*

All four cases pass, including the one with independent ground truth: it re-perceives rather than trusting what it did earlier. This is what snapshot scopes were for, and it is the axis where the architecture most visibly earns its keep.

**told, replaced, forgotten, and re-perceived after a change made behind its back**

- value: 4/4 cases pass: told-fact-recalled=ok, replacement-supersedes=ok, forgetting-takes-effect=ok, stale-belief-after-world-change=ok
- n = 8
- floor: n/a
- control: the stale-belief case changes the world outside the agent and re-asks
- environment author: us · grader: the simulator's shell, read by a separate process · agent can influence grader: no
- held out: written before it was run; fresh conversation
- source: `eval/profile/probes.py, run against the live assistant`
- note: the strongest case is the last: a folder it created was deleted behind its back, and it did not report it afterwards

## grounding

*Is an answer traceable to something that actually supports it?*

Every answer the assistant gives now names its source, and where we can check the source independently it holds. On public multi-hop data the same mechanism covers under a third of the gold evidence, so checkable citation is a mechanism we have and good retrieval is not.

**does the source an answer cites actually support it?**

- value: 3/3 cases pass: told-fact-cites-being-told=ok, screen-answer-matches-independent-read=ok, pixel-answer-cites-pixels=ok
- n = 8
- floor: n/a
- control: the dock count is checked against our own DOM read, on a different code path from the agent's perceiver
- environment author: us · grader: our own DOM read plus facts we told it · agent can influence grader: no
- held out: written before it was run
- source: `eval/profile/probes.py, run against the live assistant`
- note: 'how many icons' has two defensible answers (13 launchers, or 14 including the app-grid button); it gave 13 and named all 13, so the row scores either reading and says so

**citation coverage on public data (HotpotQA supporting facts)**

- value: 31.6% of gold supporting sentences covered on average; ALL of them on 3.3% of items
- n = 300
- floor: n/a
- control: citations checked against published supporting facts
- environment author: public dataset (HotpotQA) · grader: public labels · agent can influence grader: no
- held out: 300-item test slice
- source: `docs/revival/12-open-domain.md, eval/results/open_domain.json`
- note: the citation mechanism is real and checkable; the retrieval behind it is thin

## robustness

*Does it hold up under perturbation?*

It holds up well against the variations the apps were built to throw (0.944-1.000 per task across ~103k episodes), and it abstains rather than guessing on the genuinely unanswerable case. But the perturbations are ones we anticipated, so this is robustness to known unknowns.

**accuracy per task under each app's own randomization (label variants, injected 503s, dialogs, pagination)**

- value: access 0.9999 | chart 0.9998 | desktop 1.0000 | inbox 0.9439 | recon 0.9971 | shop 1.0000
- n = 305330
- floor: none run
- control: none run
- environment author: us · grader: our own code · agent can influence grader: no
- held out: the apps randomize every episode; live seeds differ from bench seeds
- source: `eval/results/browser_agents_recovered.json`
- note: inbox is the weakest at ~0.944 and its misses are abstentions; the perturbations are ones we built, so this measures robustness to anticipated variation

**the one environment we changed after a failure, restored and re-measured**

- value: original generator 0.9950 with 2 abstentions and 0 wrong; changed generator 1.0000
- n = 800, 95% CI [0.9872, 0.9981]
- floor: n/a
- control: on the 5 known tie seeds the original scores 0.00 with 5 abstentions and 0 wrong
- environment author: us · grader: our own code · agent can influence grader: no
- held out: 400 fresh seeds never used for tuning
- source: `eval/results/chart_environment_sweep400.json, chart_environment_tie_seeds.json`
- note: the post-hoc change was worth 0.5 points; where it bites, the agent abstains every time instead of guessing

## cost_and_evidence_integrity

*What does it cost, and who graded it?*

Zero model calls at ~0.986 items correct is real and cheap. The integrity picture is the uncomfortable half: most headline rows are graded by code we wrote, in environments we wrote, and one environment was changed after a failure (since restored and re-measured).

**share of our own headline results graded by someone other than us**

- value: 5/19 headline rows have a public or hidden grader; 13/19 have both environment and grader ours
- n = 19
- floor: n/a
- control: n/a
- environment author: us · grader: the audit reads the result files themselves · agent can influence grader: no
- held out: n/a
- source: `eval/results/evidence_audit.json`
- note: verdicts across all rows: {"strong evidence": 7, "moderate evidence": 6, "weak evidence": 9, "self-graded": 5, "unverified": 1}; post-hoc environment changes: ['browser-agents-6task']

**cost per item: cheap tier vs model tier**

- value: rules 0.0001 s and 0 model calls; model 0.366 s and 300 calls (55 tok/s, 138669 new tokens over 2055 calls)
- n = 300
- floor: n/a
- control: same items both arms
- environment author: public dataset · grader: public labels · agent can influence grader: no
- held out: test slice
- source: `eval/results/open_domain.json`
- note: the cheap tier is ~3,600x cheaper per item and far less accurate; cost is the one axis where it wins outright

**environment cost after moving to a third-party engine**

- value: {"date": "2026-09-17 14:07", "episodes": 10, "items_correct": 30, "items": 30, "episodes_fully_correct": 10, "duplicate_effects": 0, "episodes_per_second": 56.7, "actions_per_episode": 41, "actions_pe
- floor: none run
- control: none run
- environment author: computerworld (third party) · grader: our own code · agent can influence grader: no
- held out: not stated
- source: `eval/results/computerworld.json`
- note: determinism by state hash makes repeated measurement cheap, which is what let the prediction axis exist at all

## What we could not ground, and why

- **Sample efficiency as a rate.** Two skills have ever been learned (one adopted after succeeding
  on a request it was not learned from, one still on trial, one reuse in total). The mechanism and
  its adoption gate demonstrably run; a rate from n=2 would be noise. The teacher model was not
  running when this was computed, so no new learning could be measured.
- **Causal discrimination.** Interventions execute and the engine's fork/restore makes a real
  controlled experiment possible, but in the recorded run intervention found 14 links against
  correlation's 13, every aspect was judged `caused`, and the case built to show a co-occurring
  signal being *rejected* measured zero effect in both framings. The apparatus is there; the
  discrimination is not shown.
- **Robustness to unanticipated perturbation.** Every perturbation family we measure is one we
  built into our own apps. That is robustness to known unknowns.

## Two defects this harness found while being built

Both were found by driving the live assistant rather than by reading it, which is the pattern
that keeps holding in this project (§9, §11, §16).

1. **`copy` is broken.** `copy note.txt to <folder>` produces a run step with no command and
   raises `TypeError: 'NoneType' object is not subscriptable`. Reproduced in a fresh conversation
   in four separate turns, so it is the act itself and not a depth effect.
2. **A failed request ends the conversation.** After that crash, every later turn fails the same
   way — including turns that succeed in a fresh conversation — because the crashed request stays
   selected. There is no recovery short of a restart. This is the more serious of the two: one
   malformed act makes the agent permanently unusable rather than degrading one answer.

Both are in files this harness does not own, and are reported rather than patched.

## The three measurements that would most change our beliefs

1. **Capability-aware abstention, re-measured against the oracle gap.** §12 showed a router with
   perfect foresight gains only 1.3 and 2.0 points, so routing cannot be rescued by a better
   signal — but abstention itself is miscalibrated in both directions (54 wrongful refusals, 23
   answers to unanswerable questions). Measuring a calibrated head against those two error types
   separately would tell us whether `Unknown` is a real capability or a threshold. *Cost:* hours,
   reusing the existing 150/300 splits; no new environment.
2. **Horizon with a working long task.** Everything we know about long horizons comes from a
   0/8 result whose runs were throughput-bound at ~200 s per model call. Re-running the
   hidden-grader suite with a faster model would separate 'the architecture cannot do this' from
   'we never gave it enough steps'. *Cost:* a faster model or a day of local throughput work; the
   tasks and hidden checkers already exist.
3. **Prediction on an environment we did not write.** The prediction axis is the best-graded one
   we have, and it runs on a third-party engine — but on a task we authored. Pointing it at a
   public interactive benchmark would test whether expectation-learning is a property of the
   mechanism or of our world. *Cost:* the adapter, plus whatever the benchmark needs; the
   prediction machinery is unchanged.

## Running it

```sh
# live probes (each suite gets a fresh assistant instance: a crashed request is unrecoverable)
python -m examples.browser_agents.assistant.server --port 8773 --no-open &
python -m eval.profile.run_probes compositionality http://127.0.0.1:8773 probe_comp.json
python -m eval.profile.run_probes belief_revision   http://127.0.0.1:8774 probe_belief.json
python -m eval.profile.run_probes grounding         http://127.0.0.1:8775 probe_ground.json
# assemble, then regenerate this document
python -m eval.profile.profile --probes <dir-with-probe-json>
python -m eval.profile.report
```
