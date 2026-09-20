# 11. Evidence audit: who wrote the environment, and who graded the answer

> **Pixel pipeline retirement:** the fixed geometry/control/prompt semantic path,
> pixel providers, and associated end-to-end/fusion runners described in these
> historical results are retired. Saved measurements are not current runnable
> capabilities. See [70 — Retiring pixel semantic rules](70-retiring-pixel-semantic-rules.md)
> for the retained components and present vision limits.

**Headline: of the 19 results this project has quoted as headlines, 13/19 run in an environment we wrote AND are graded by code we wrote. Only 5/19 are graded by a public dataset's labels or by hidden tests we cannot see.**

The conclusions that survive that filter are mostly failures or narrow component results: a TF-IDF classifier beating a general model on in-domain intents (public labels); every arm of the long-horizon evaluation failing while an unstructured ReAct loop scored higher (hidden tests); and, in [12 — open-domain benchmarks](12-open-domain.md), the no-model arm answering almost nothing off its home turf while the proposed cascade comes out *worse* than the same model prompted plainly on 3 of 4 public benchmarks.

This file exists because the user asked whether wireheading had crept in. It had, in three specific ways, listed first so they are not buried.

## The three real problems

**1. One environment was changed after the agent failed in it. Now restored, and re-measured.** After the chart agent mis-read two pixel-identical bars, the generator was changed to redraw until the tallest bar is visibly tallest, and the score went to 100%. The original generator is now the DEFAULT again; the changed one survives only as `?ties=avoid`, a separate configuration that must not be quoted as the headline. Re-measured on fresh seeds (`eval/results/chart_environment_sweep400.json`, `chart_environment.py`):

| Generator | 400 fresh seeds | The 5 known tie seeds |
| --- | ---: | ---: |
| **Original (ties possible) — the honest headline** | **99.5%** (796/800), 2 near-tie abstentions, 0 wrong answers | **0/10**, 5 abstentions, 0 wrong answers |
| Changed (ties avoided) | 100.0% (800/800) | 10/10 |

So the change was worth 0.5 points on random seeds, because a pixel tie occurs in about 0.5% of them — the inflation was small. But on the seeds where it bites, the original environment is genuinely unanswerable and the agent scores zero while abstaining every time and never guessing wrong. That is the behaviour worth reporting, and it is only visible in the environment we did not edit. The practice was still wrong even though the effect was small.

**2. The most-quoted number in the project had been overwritten, and is now recovered at larger n.** "Six agents, ~100% of items, 0 model calls" was quoted throughout this work, but `eval/results/browser_agents.json` holds only one task (recon, 125/125): a sibling's single-task re-run overwrote the six-task bench file. Rebuilding it from the live-wall recordings gives **102,899 episodes and 290,909/296,674 items (98.06%) with 0 model calls** — shop and desktop at 100.00%, access 99.99%, chart 99.98%, recon 99.71%, inbox 94.39% — now written to `eval/results/browser_agents_recovered.json`. The sample is ~750x the bench run, and none of it changes the fact that we wrote the apps, the graders inside them and the agents. A very large score against an exam you wrote yourself is still an exam you wrote yourself.

**3. The language benchmark grades the regexes against cases written from the regexes.** `language_benchmark.json` reports the 105 regexes at 152/152 (100%) and the grammar at 92/152 (60.5%). The 152 cases were authored alongside those regexes, so the 100% is true by construction. It stays as a **regression suite** and stops being quoted as coverage. The replacement already exists and has independent provenance: `tests/test_civ_language_demands.py`, 193 cases generated from every claim shape the civilization's minds actually speak across four diverged dialects, which found five copula failures, a lost quantifier, a mis-stemmed verb, an inexpressible tense and a partitive gap while the fork's own suite passed. Quote that, and the assistant's 152 utterances, which predate the grammar.

**The two relaxed retry policies, reviewed: both are genuine correctness fixes, not difficulty reductions.** In `tasks/access.py` and `tasks/chart.py` the cap on attempts that *might have applied an effect* is still 3; what changed is that a 503 explicitly saying nothing was saved no longer counts toward it (total attempts capped at 6). Retrying an attempt the app states had no effect cannot duplicate an effect, and refusing to is a correctness error, not caution — it is the same principle the recovery agent already encodes. Duplicate effects remain 0 in every result file that records them. No restoration needed.

Two further practices are worth naming, both already disclosed in their own docs but easy to lose: perception rules were changed after held-out metrics were first computed (`docs/revival/07` §7.4 lists five such changes, plus two more after the final run), and the civilization's defaults (population, world size, band count) were chosen by measurement after a smaller world collapsed.

## The pattern: self-authored sets pass while running worlds fail

Three independent cases in this session, from three different directions, are the same failure:

1. **The grammar's own test sets.** None of the sets the language fork authored contained a copular sentence with a third-person subject. Every set it wrote passed. The first consumer it did not write — the civilization, with 17 spoken sentence shapes — broke on five of its commonest constructions immediately. The blind spot was systematic, so more cases by the same author would never have found it.
2. **The OCR fine-tune.** "2 false successes -> 0" was measured on the ten episodes of the environment where those two failures had been observed, with a corpus built to target the digit confusions seen there. Moved to `computerworld`, all three recognizers make the same three errors, and the failures are lost hyphens — a class the corpus never covered. The fix was real for the failure we had already seen.
3. **The open-domain cascade.** The cascade looked good on Banking77, where the abstention threshold was fitted on in-domain validation data. On four public benchmarks it is *worse* than the same model prompted plainly on three of them, because the abstention signal predicts lexical overlap rather than whether the cheap tier can answer. See [12](12-open-domain.md).

The common shape: a mechanism is validated against the failure mode we had already seen, in the environment where we saw it, by the author of both. Every one of these passed its own tests. What broke them was contact with something we did not write — a consumer, a different simulator, a public dataset.

**The credit side, and the cheapest fix available.** The same session produced the counter-example: `tests/test_civ_language_demands.py` holds 193 cases with *independent provenance* — not invented to exercise the grammar, but generated from every claim shape the civilization's minds actually speak, across four diverged dialects. It found five copula failures, a lost quantifier, a mis-stemmed verb, an inexpressible tense and a partitive gap, all in the commonest constructions, while the grammar's own 152-case suite passed. Fixing those also removed a spurious mechanism in the simulation, where ungrammatical output was being counted as unfamiliar vocabulary and gave four common sentence types a ~45% chance of being "misunderstood".

Two components we wrote, one consuming the other, is weaker evidence than an outside grader and far stronger than one component grading itself. It is also nearly free. **The practical rule this session earns: a capability claim should be graded by the consumer that needs it, never by a set written alongside the implementation.** For the grammar that means quoting the 193-case demand set and the assistant's 152 utterances (which predate the grammar), and keeping the fork's own act benchmark as the regression suite it is.

## What good practice we did follow

- **Hidden graders, once.** The long-horizon tasks use hidden tests and pre-registered geometric checks the agent cannot read or edit, plus a plain-ReAct control with the same model. It is the only evaluation here built to be failed, and it failed.
- **Disconfirming arms.** The recovery simulation includes an arm where the agent is told something false, and duplicates return (0 -> 0.92%). The doc states that the safety comes from the stated facts, not from the framework.
- **A control that triggered a refusal.** Rule induction rejected a 100%-accurate artifact because its verdicts changed under renaming. That is the mechanism working against its author's interest.
- **Mechanically true labels.** The OCR corpus is labelled by the string the agent typed or the string the DOM reported, not by our judgement, and an apps-only corpus was reported even though it regressed.
- **State-based grading.** The desktop chore is graded by reading the simulator's files and git history, not by the agent's report. On `computerworld` the episode is reproducible by `state_hash`.

## Structural weaknesses that no single fix addresses

- **Same-author bias is total on the browser agents:** we wrote the five web apps, the graders inside them, the agents, and the difficulty distribution. Robustness to fresh seeds is real; robustness to an app we did not write is untested.
- **The perception fusion table cannot measure DOM.** The DOM read is simultaneously one provider and the ground truth, so its row is 1.00 recall by construction. Only the vision and fused rows are measurements.
- **Small n on several headline claims:** false successes 2 -> 0 is two events over ten episodes; the computerworld comparison is ten episodes; the real-desktop acting test is one task in one window.
- **No random or scrambled control was ever run on the browser agents, the desktop chore, the civilization dynamics, or the invariant check.** 9 of 28 audited rows have no control.
- **The civilization has no external validity by construction.** Its defensible parts are its invariants (goods conserved to ~1e-14) and its published corrections against its own earlier claims.

## Counts

| Measure | Value |
| --- | ---: |
| Audited claims | 28 |
| Environment written by us | 23/28 |
| Graded by our own code | 23/28 |
| Both environment and grader ours | 22/28 |
| Headline claims, both ours | 13/19 |
| Headline claims, public labels or hidden tests | 5/19 |

| Verdict | All claims | Headline claims |
| --- | ---: | ---: |
| strong evidence | 7 | 6 |
| moderate evidence | 6 | 3 |
| weak evidence | 9 | 7 |
| self-graded | 5 | 2 |
| unverified | 1 | 1 |

## Every claim, one row each

| # | Claim | Environment author | Grader | Held-out discipline | Control | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| 1 ★ | **banking77-cascade**: classify cascade: rules -> learned -> model, selective accuracy and coverage | public dataset (Banking77, PolyAI CC-BY-4.0, sha256 recorded) | public labels | official held-out test split (3,080 items); Wilson 95% CIs reported | no random control run; Banking77 majority-class floor is ~1.3% (77 classes), so the numbers are far above chance | strong evidence |
| 2 ★ | **banking77-escalation-harm**: zero-shot general-model escalation HURT on in-domain fine-grained intents | public dataset | public labels | same official test split | direct comparison against the tier it replaced, on the same items | strong evidence |
| 3 ★ | **hotpot-context-recall**: BM25 -> pack beats document order for supporting-fact recall at a token budget | public dataset (HotpotQA distractor validation, 7,405 questions, sha256 recorded) | public gold supporting-fact labels | whole validation set, no split needed (no fitting) | document-order baseline on identical inputs acts as the floor | strong evidence |
| 4 ★ | **language-act-benchmark**: the 105 regexes read 152/152 of the assistant's utterance benchmark; the grammar reads 92 | us | our expected (act, slots) labels | none: no held-out utterance split existed until the open-vocabulary set was added | none | unverified |
| 5 | **language-compositional**: the grammar handles compositional constructions the regexes cannot | us | our expected readings | none; cases and grammar authored by the same agent, in the same session | the regex arm is a genuine contrast (0/15), which is informative | self-graded |
| 6 ★ | **induction-controls**: rule induction adopts a real structural rule and REFUSES a vocabulary-reading artifact | us (synthetic routing task) | our labels | held-out split plus three controls, fixed before adoption | random (0.250), shifted-question (0.317), rename-invariance: all run, and the rename control is what triggered a refusal | weak evidence |
| 7 ★ | **browser-agents-6task**: six browser agents score ~100% of items on fresh seeds with 0 model calls at 20-28 UI actions/s | us (we wrote the five web apps AND the agents that solve them) | our code: window.__score() inside our own page, or our own checker | fresh seeds were used as a held-out set (seeds 2000-2150 after tuning on 1-40), which is real discipline; the recovered live runs cover ~103k episodes across unseen seeds; but the apps' difficulty distribution is ours | none: no scrambled-agent or random-policy control was ever run on these apps | weak evidence |
| 8 ★ | **desktop-chore-seed**: the desktop agent completes a multi-step shell chore and verifies it | third party for the engine (the user's Seed, then the user's computerworld), us for the task and the world definition | simulator state (file contents and git history read back through a privileged API), not the agent's report | fresh seeds; on computerworld, episodes are exactly reproducible via state_hash | none | weak evidence |
| 9 ★ | **computerworld-speed**: the same desktop chore runs ~450x faster per episode on computerworld than on the browser-based simulator | third party (the user's engine) | simulator state plus the engine's own state_hash | n=10 episodes for the engine row; determinism checked on 2 seeds | the old path is the baseline; a state_hash mismatch check acts as a negative control on determinism | moderate evidence |
| 10 ★ | **perception-fusion**: fusing DOM and vision lifts word coverage; vision repairs a thinned accessibility tree | us (frames captured from our apps and the user's Seed machines) | the DOM read at the same instant | explicit tune / test_app / test_os splits, with 2 fonts and 8 apps held out for the recognizer work | a dom-degraded arm (icon names stripped) serves as an ablation; no random control | weak evidence |
| 11 ★ | **recognizer-finetune**: a wide corpus of free labels fixes the systematic OCR misreads | us (rendered crops) plus DOM-labelled app text | mechanically exact labels: the string the agent typed, or the string the DOM reported | 2 fonts and 8 apps never trained on | the apps-only arm is a genuine negative control and it regressed below baseline on unseen fonts | moderate evidence |
| 12 ★ | **false-success-fix**: the retrained recognizer removes confident false successes end to end | us (task) in a third-party simulator | simulator state | same 10 seeds throughout, which is a fixed set but a tiny one | three failed guards act as alternatives; no random control | weak evidence |
| 13 | **real-desktop-acting**: the agent completed a real GTK task on a real desktop through the accessibility tree; fusion failed the same task | us (we wrote the scratch app) on a real OS | the app's own status label, read back by the agent | none; n=1 task, 1 window, 2 providers | the fused arm is a contrast | weak evidence |
| 14 | **invariant-check**: task-stated invariants catch the confident-misread case without refusing correct reads | us | our recorded frames and known seeds | none; these are the frames on which the failure was originally observed | none | self-graded |
| 15 ★ | **longhorizon**: the assistant can do sophisticated multi-step work (CAD, EEG, coding with feedback) | us for the tasks, third party for the tools (cadquery/trimesh, pytest, real sandbox shell) | HIDDEN tests and geometric probes the agent cannot see or edit | best in the project: tasks authored before tuning, hidden graders, held-out split, budgets, and a plain-ReAct baseline with the same model | plain ReAct with the same model is the control, and it BEAT both tensacode arms (0.25 vs 0.0) | strong evidence |
| 16 ★ | **recovery-sim**: the recovery agent never duplicates an effect under a correctly described target system | us (simulation, 5,000 sampled worlds, fault model written by the same author as the agent) | our simulator's ground truth | all policies run on identical sampled worlds; a mis-specified-facts sensitivity arm is included | three baseline policies plus the mis-specified-facts arm, which BREAKS the result (duplicates return) | weak evidence |
| 17 ★ | **civ-sim-dynamics**: emergent money, deforestation caps, emergent settlements, ideology equilibrium, 43% belief correctness after the parser swap | us, entirely: world, economy, minds, grammar and graders | our simulator's own state | none applicable; 3 seeds, determinism per seed | a no-minds arm exists for COST comparison; no control for the dynamics claims | self-graded |
| 18 | **civ-conservation**: the simulation conserves all goods | us | our invariant test | n/a (an invariant, not a sample) | n/a | moderate evidence |
| 19 | **assistant-parity**: the data-driven procedure engine reproduces the old generator engine exactly | us | differential test against the previous implementation | n/a (equivalence, not capability) | the old engine IS the oracle | strong evidence |
| 20 | **store-scaling**: the claim store scales to ~112k claims with measured ingest, query and patch costs | us (synthetic uniform data) | wall-clock and memory measurement | n/a (performance) | n/a | moderate evidence |
| 21 | **legacy-representation**: the proposed records represent the same objects losslessly where the legacy TCIR was lossy or crashed | us for the fixtures, third party for the legacy code under test | round-trip equality against real legacy behaviour in a sandbox | fixtures chosen by us | the legacy implementation is the comparator | moderate evidence |
| 22 ★ | **tensacode-overhead**: the runtime's own overhead is negligible next to any real backend | us (microbenchmark) | wall-clock measurement with output equality asserted | n/a | direct calls to the same tiers, bypassing the runtime, as the comparator | strong evidence |
| 23 ★ | **language-independent-provenance**: the grammar's coverage, measured by a set it did not author | us, but a DIFFERENT author than the grammar: the cases are generated from what research/civ_sim/minds.py actually puts into agents' mouths, not invented to exercise the parser | structural assertions (round-trip recovery; no doubled copula, no bare 'am' with a third-person subject, no partitive without 'of', no raw booleans, one full stop) | independent provenance rather than a held-out split: the consumer's needs were fixed before the grammar met them | the fork's own 152-case act benchmark is the contrast, and it passed while this set failed | moderate evidence |
| 24 | **running-world-found-a-defect**: a running world exposes defects no authored set catches (the credit side of the same pattern) | us (two independent components, one consuming the other) | the simulation's own measured misunderstanding rate before and after | n/a | before/after comparison of the misunderstanding rate | weak evidence |
| 25 ★ | **finetune-did-not-travel**: the OCR fine-tune's removal of confident false successes (2 -> 0) held up | us (task) in two different simulators | simulator state | held-out fonts and apps for the crop metrics; the same 10 seeds for the end-to-end number | three recognizers compared in the new environment, and they agree | weak evidence |
| 26 ★ | **copula-gap**: the grammar's authored test sets measured its coverage | us | our expected readings | none: the blind spot was systematic, so no amount of the same author's cases would have found it | the civilization's 17 spoken shapes acted as an independent set and found five real bugs | self-graded |
| 27 ★ | **open-domain**: tensacode's cognition on public open-domain benchmarks, against the same model alone and a cascade | public datasets (SQuAD 2.0, HotpotQA distractor, GSM8K, ARC-Easy) | public labels; no model judged anything | 300 test items per benchmark drawn from a fixed shuffle, 150 disjoint calibration items, Wilson CIs | majority/frequency floor and a random control on every benchmark, a forced-guess variant on ARC-Easy, plus a plain-model arm and an equal-coverage comparison | strong evidence |
| 28 | **test-suite**: the test suite passes | us | our assertions | n/a | n/a | self-graded |

★ = quoted as a headline result.

## Detail per claim

### banking77-cascade — strong evidence

- **Claim:** classify cascade: rules -> learned -> model, selective accuracy and coverage
- **Number:** rules 97.1% @5.6% cov; learned 94.3% @93.8%; cascade 94.1% @93.9%; +model 91.0% @99.5%
- **Lives in:** eval/results/banking77.json, docs/revival/05-evaluation.md §6.1
- **Environment author:** public dataset (Banking77, PolyAI CC-BY-4.0, sha256 recorded)
- **Grader:** public labels
- **Can the agent influence the grader?** no
- **Changes after seeing results:** rules revised once on a TRAIN holdout, then frozen before test; threshold chosen on a 1,003-item validation split; no test-set fitting claimed
- **Held-out discipline:** official held-out test split (3,080 items); Wilson 95% CIs reported
- **Control:** no random control run; Banking77 majority-class floor is ~1.3% (77 classes), so the numbers are far above chance
- **Note:** The strongest capability evidence in the repo, and it is about a small TF-IDF classifier, not about cognition.

### banking77-escalation-harm — strong evidence

- **Claim:** zero-shot general-model escalation HURT on in-domain fine-grained intents
- **Number:** model correct on 38.2% (31.2-45.6) of the 173 items it answered; discarded learned argmax was 46.6% on the same items
- **Lives in:** eval/results/banking77.json, docs/revival/05-evaluation.md §6.1
- **Environment author:** public dataset
- **Grader:** public labels
- **Can the agent influence the grader?** no
- **Changes after seeing results:** single zero-shot prompt, no prompt search (stated deliberately)
- **Held-out discipline:** same official test split
- **Control:** direct comparison against the tier it replaced, on the same items
- **Note:** A negative result about our own proposed escalation, on public labels. This is the shape of evidence the rest of the project lacks.

### hotpot-context-recall — strong evidence

- **Claim:** BM25 -> pack beats document order for supporting-fact recall at a token budget
- **Number:** 41.3% / 59.3% / 74.4% recall at budgets 64 / 128 / 256 vs 8.8% / 13.8% / 24.2%
- **Lives in:** eval/results/context_hotpot.json, docs/revival/05-evaluation.md §6.2
- **Environment author:** public dataset (HotpotQA distractor validation, 7,405 questions, sha256 recorded)
- **Grader:** public gold supporting-fact labels
- **Can the agent influence the grader?** no
- **Changes after seeing results:** none claimed; no threshold fitted
- **Held-out discipline:** whole validation set, no split needed (no fitting)
- **Control:** document-order baseline on identical inputs acts as the floor
- **Note:** Measures EVIDENCE RECALL, not answer quality. 26-59% of gold evidence is still missed. The claim is narrower than 'context selection works'.

### language-act-benchmark — unverified

- **Claim:** the 105 regexes read 152/152 of the assistant's utterance benchmark; the grammar reads 92
- **Number:** regex 152/152 (100%); grammar 92/152 (60.5%)
- **Lives in:** eval/results/language_benchmark.json, docs/revival/09-language-and-induction.md
- **Environment author:** us
- **Grader:** our expected (act, slots) labels
- **Can the agent influence the grader?** no, but the SET was written alongside the regexes it grades
- **Changes after seeing results:** the grammar was repeatedly fixed against failures on this same set (57.9% -> 61.8% -> 60.5% as behaviour changed)
- **Held-out discipline:** none: no held-out utterance split existed until the open-vocabulary set was added
- **Control:** none
- **Note:** The regexes' 100% is circular by construction: the cases were authored from the regexes' own behaviour. It is a regression suite, not a measurement of language coverage. Quoting it as '100% vs 60%' overstates the regexes and understates the grammar.

### language-compositional — self-graded

- **Claim:** the grammar handles compositional constructions the regexes cannot
- **Number:** in-lexicon 15/15 and open-vocabulary 19/19 for the grammar; regexes 0/15 and 0/19
- **Lives in:** eval/results/language_benchmark.json
- **Environment author:** us
- **Grader:** our expected readings
- **Can the agent influence the grader?** no
- **Changes after seeing results:** the open-vocabulary set was written AFTER the parent supplied four failing transcripts, then the parser was fixed until it passed them
- **Held-out discipline:** none; cases and grammar authored by the same agent, in the same session
- **Control:** the regex arm is a genuine contrast (0/15), which is informative
- **Note:** The contrast against regexes is real; the absolute 15/15 and 19/19 are fit-to-own-tests. The parser's honest generalization number is unknown.

### induction-controls — weak evidence

- **Claim:** rule induction adopts a real structural rule and REFUSES a vocabulary-reading artifact
- **Number:** real task: held-out 1.000, floor 0.750, random control 0.250, shifted control 0.317, rename-invariant -> adopted. Vocabulary task: 100% held-out but rename-variant -> refused
- **Lives in:** eval/results/language_benchmark.json (induction), docs/revival/09
- **Environment author:** us (synthetic routing task)
- **Grader:** our labels
- **Can the agent influence the grader?** no
- **Changes after seeing results:** two of the agent's own controls were wrong first and were fixed after their tests caught them (disclosed)
- **Held-out discipline:** held-out split plus three controls, fixed before adoption
- **Control:** random (0.250), shifted-question (0.317), rename-invariance: all run, and the rename control is what triggered a refusal
- **Note:** A synthetic task, but the discipline is the best in the repo: a 100%-accurate artifact was rejected because it failed a control. That mechanism is evidence even though the task is ours.

### browser-agents-6task — weak evidence

- **Claim:** six browser agents score ~100% of items on fresh seeds with 0 model calls at 20-28 UI actions/s
- **Number:** RECOVERED at much larger n from the live-wall recordings (eval/results/browser_agents_recovered.json): 102,899 episodes, 290,909/296,674 items = 98.06%, 0 model calls. Per task: shop 100.00%, desktop 100.00%, access 99.99%, chart 99.98%, recon 99.71%, inbox 94.39%
- **Lives in:** eval/results/browser_agents.json holds only recon (125/125); the six-task bench file was overwritten. Recovered from $SP/live_run{6,7}.jsonl into eval/results/browser_agents_recovered.json
- **Environment author:** us (we wrote the five web apps AND the agents that solve them)
- **Grader:** our code: window.__score() inside our own page, or our own checker
- **Can the agent influence the grader?** not directly (the score function is not exposed to perception), but same-author bias is total
- **Changes after seeing results:** YES, and the clearest case in the project: after the chart agent failed on pixel-identical bars, the chart GENERATOR was changed to redraw until the tallest bar is visibly tallest. NOW RESTORED: ties are possible again by default (99.5% on 400 fresh seeds, 2 abstentions, 0 wrong answers; 0/10 with 5 abstentions on the 5 known tie seeds), and the changed generator survives only as the opt-in ?ties=avoid. The access/chart retry relaxations were reviewed and are genuine correctness fixes: the cap on possibly-effectful attempts is still 3, and only attempts the app said had no effect are exempt.
- **Held-out discipline:** fresh seeds were used as a held-out set (seeds 2000-2150 after tuning on 1-40), which is real discipline; the recovered live runs cover ~103k episodes across unseen seeds; but the apps' difficulty distribution is ours
- **Control:** none: no scrambled-agent or random-policy control was ever run on these apps
- **Note:** The bench file was overwritten by a sibling's single-task re-run, so the quoted per-task tallies are gone; the same claim is recoverable from the live recordings at 750x the sample size (102,899 episodes) and holds at 98.06% overall with 0 model calls. What does NOT change: we wrote the apps, the graders inside them and the agents, and the chart generator was made easier after a failure — now restored to ties-possible by default and re-measured at 99.5% on 400 fresh seeds, with the changed version kept only as `?ties=avoid`. Large n against your own exam is still your own exam. Note the live figures are continuous-run episodes, not the bench harness, so seeds differ.

### desktop-chore-seed — weak evidence

- **Claim:** the desktop agent completes a multi-step shell chore and verifies it
- **Number:** 120/120 items over 40 episodes (Seed); 10/10 episodes and 30/30 checks (computerworld)
- **Lives in:** eval/results/browser_agents.json (historical), eval/results/computerworld.json
- **Environment author:** third party for the engine (the user's Seed, then the user's computerworld), us for the task and the world definition
- **Grader:** simulator state (file contents and git history read back through a privileged API), not the agent's report
- **Can the agent influence the grader?** the agent writes the files being graded, but cannot alter the checker
- **Changes after seeing results:** scoring was changed mid-project when Seed kept git history after folder deletion (counting only commits made during the episode); a random ls marker was removed when porting, which is what made episodes reproducible
- **Held-out discipline:** fresh seeds; on computerworld, episodes are exactly reproducible via state_hash
- **Control:** none
- **Note:** State-based grading is a genuine step up from self-report. The task is still one we wrote for an agent we wrote.

### computerworld-speed — moderate evidence

- **Claim:** the same desktop chore runs ~450x faster per episode on computerworld than on the browser-based simulator
- **Number:** 0.0167 s/episode and 56.7-60.2 episodes/s vs 7.07 s/episode; perception 0.42 ms; 30/30 items; determinism confirmed by state_hash
- **Lives in:** eval/results/computerworld.json, docs/revival/10-computerworld.md
- **Environment author:** third party (the user's engine)
- **Grader:** simulator state plus the engine's own state_hash
- **Can the agent influence the grader?** no
- **Changes after seeing results:** none; the comparison is across different seed ranges, which the doc labels as order-of-magnitude rather than paired
- **Held-out discipline:** n=10 episodes for the engine row; determinism checked on 2 seeds
- **Control:** the old path is the baseline; a state_hash mismatch check acts as a negative control on determinism
- **Note:** A performance and determinism claim, not a capability claim, and it is sound as such. n=10 is small.

### perception-fusion — weak evidence

- **Claim:** fusing DOM and vision lifts word coverage; vision repairs a thinned accessibility tree
- **Number:** words covered 0.66-0.78 (DOM) -> 0.95-0.99 (fused); icon naming on a stripped tree 0.00 -> 0.73/0.24/0.09
- **Lives in:** eval/results/perception_fusion.json, docs/revival/07 §7.6.2
- **Environment author:** us (frames captured from our apps and the user's Seed machines)
- **Grader:** the DOM read at the same instant
- **Can the agent influence the grader?** no, but the DOM is BOTH a provider and the ground truth, so the DOM row is 1.00 recall by construction
- **Changes after seeing results:** several perception changes were made after held-out metrics were first computed (disclosed in docs/revival/07 §7.4: weak-detection retention, prompt pattern, phrase splitting, title fallback, recognizer swap), plus two more after the final run
- **Held-out discipline:** explicit tune / test_app / test_os splits, with 2 fonts and 8 apps held out for the recognizer work
- **Control:** a dom-degraded arm (icon names stripped) serves as an ablation; no random control
- **Note:** The fused-vs-DOM comparison is meaningful. Any row where DOM is the ground truth cannot be read as DOM's accuracy.

### recognizer-finetune — moderate evidence

- **Claim:** a wide corpus of free labels fixes the systematic OCR misreads
- **Number:** held-out fonts exact 0.668 -> 0.794, digits 0.828 -> 0.882, tilde words 0/11 -> 8/11
- **Lives in:** eval/results/recognizer_comparison.json, docs/revival/07 §7.6.6
- **Environment author:** us (rendered crops) plus DOM-labelled app text
- **Grader:** mechanically exact labels: the string the agent typed, or the string the DOM reported
- **Can the agent influence the grader?** no
- **Changes after seeing results:** an apps-only corpus was tried first and made unseen fonts worse; the wide corpus was the fix (both reported)
- **Held-out discipline:** 2 fonts and 8 apps never trained on
- **Control:** the apps-only arm is a genuine negative control and it regressed below baseline on unseen fonts
- **Note:** The labels here are not our judgement, they are mechanically true, which makes this one of the better-grounded measurements in the perception work.

### false-success-fix — weak evidence

- **Claim:** the retrained recognizer removes confident false successes end to end
- **Number:** false successes 2 -> 0; verified episodes 6/10 -> 7/10; checks 21/30 unchanged
- **Lives in:** eval/results/vision_desktop_e2e_pixels*.json, docs/revival/07 §7.6.4
- **Environment author:** us (task) in a third-party simulator
- **Grader:** simulator state
- **Can the agent influence the grader?** no
- **Changes after seeing results:** three earlier guards were tried and all measured worse (cross-place digit check, corroboration policy, confidence gate); all three are reported, two kept behind flags
- **Held-out discipline:** same 10 seeds throughout, which is a fixed set but a tiny one
- **Control:** three failed guards act as alternatives; no random control
- **Note:** n=10 episodes. 2 -> 0 on ten episodes is 2 events. Directionally supported by the crop-level numbers, which is the real evidence.

### real-desktop-acting — weak evidence

- **Claim:** the agent completed a real GTK task on a real desktop through the accessibility tree; fusion failed the same task
- **Number:** AT-SPI verified, 2.1 s, perception 48-92 ms; AT-SPI+vision refused to type, unverified
- **Lives in:** eval/results/real_acting_report.json, docs/revival/07 §7.6.3
- **Environment author:** us (we wrote the scratch app) on a real OS
- **Grader:** the app's own status label, read back by the agent
- **Can the agent influence the grader?** the agent writes the label's content indirectly by acting; it reports what it reads
- **Changes after seeing results:** focus verification and coordinate-frame handling were added in response to failures during this very test
- **Held-out discipline:** none; n=1 task, 1 window, 2 providers
- **Control:** the fused arm is a contrast
- **Note:** The VALUE here is the four failure modes it exposed (X11 focus stealing, AT-SPI cache, coordinate frames, fusion rivalry), not the pass/fail. Those are real-world facts simulation hid.

### invariant-check — self-graded

- **Claim:** task-stated invariants catch the confident-misread case without refusing correct reads
- **Number:** on 11 recorded frames: 1 refusal (the false-success frame), 7 corroborated, 0 corroborated-but-wrong
- **Lives in:** eval/results/invariant_eval.json, docs/revival/07 §7.6.7
- **Environment author:** us
- **Grader:** our recorded frames and known seeds
- **Can the agent influence the grader?** no
- **Changes after seeing results:** the checker had to be told which lines are data after it flagged 4 of 11 (the prompt carries the prior task's directory)
- **Held-out discipline:** none; these are the frames on which the failure was originally observed
- **Control:** none
- **Note:** n=11, selected because they contained the failure. Suggestive mechanism, no generalization evidence.

### longhorizon — strong evidence

- **Claim:** the assistant can do sophisticated multi-step work (CAD, EEG, coding with feedback)
- **Number:** every arm failed: as-is 0.0, improved 0.0, plain-ReAct 0.25 on CAD; coding 0.0 (12 hidden tests failed). 9-32 model calls, 378-2836 s, most runs hit a wall-clock cap
- **Lives in:** eval/results/longhorizon.json, docs/revival/08-long-horizon.md
- **Environment author:** us for the tasks, third party for the tools (cadquery/trimesh, pytest, real sandbox shell)
- **Grader:** HIDDEN tests and geometric probes the agent cannot see or edit
- **Can the agent influence the grader?** no, by design (hidden test directory, pre-registered checks)
- **Changes after seeing results:** none possible on the held-out task; checks were pre-registered before runs
- **Held-out discipline:** best in the project: tasks authored before tuning, hidden graders, held-out split, budgets, and a plain-ReAct baseline with the same model
- **Control:** plain ReAct with the same model is the control, and it BEAT both tensacode arms (0.25 vs 0.0)
- **Note:** The most disciplined evaluation we ran, and it is a failure: our structure did not beat an unstructured loop, and nothing passed. Also measured: ~200 s per model call under contention, so the binding constraint was model throughput.

### recovery-sim — weak evidence

- **Claim:** the recovery agent never duplicates an effect under a correctly described target system
- **Number:** 0 duplicates (0-0.08%) vs naive retry 4.76% and verify-then-retry 12.74%; 0.92% duplicates when the agent is told keys are honoured and they are not
- **Lives in:** eval/results/recovery.json, docs/revival/05 §6.3
- **Environment author:** us (simulation, 5,000 sampled worlds, fault model written by the same author as the agent)
- **Grader:** our simulator's ground truth
- **Can the agent influence the grader?** no
- **Changes after seeing results:** none claimed
- **Held-out discipline:** all policies run on identical sampled worlds; a mis-specified-facts sensitivity arm is included
- **Control:** three baseline policies plus the mis-specified-facts arm, which BREAKS the result (duplicates return)
- **Note:** The doc already states the honest conclusion: 'the safety comes from stated facts, not from TensaCode'. Self-authored fault model, but with the disconfirming arm run and reported.

### civ-sim-dynamics — self-graded

- **Claim:** emergent money, deforestation caps, emergent settlements, ideology equilibrium, 43% belief correctness after the parser swap
- **Number:** wood becomes numeraire in 1 of 3 seeds; Gini 0.38->0.70; 0 raids in 2 of 3 seeds; belief-vs-truth 43.1% correct (was 0%); 14.6% of utterances not understood
- **Lives in:** eval/results/civ_slice.json, civ_slice_dynamics.json, civ_language.json, docs/civ-sim/slice-results.md
- **Environment author:** us, entirely: world, economy, minds, grammar and graders
- **Grader:** our simulator's own state
- **Can the agent influence the grader?** the agents ARE the system being measured; there is no external truth
- **Changes after seeing results:** many, and disclosed: commons rationing, starvation myopia, caravan loss cap, livestock cap, the 240/72-squared/3-band default chosen by measurement after a 2-band world collapsed
- **Held-out discipline:** none applicable; 3 seeds, determinism per seed
- **Control:** a no-minds arm exists for COST comparison; no control for the dynamics claims
- **Note:** Zero external validity by construction. The internally valid parts are the invariants (goods conserved to ~1e-14) and the corrections it published against its own earlier claims (the 'rumours diverge' story was measured false).

### civ-conservation — moderate evidence

- **Claim:** the simulation conserves all goods
- **Number:** absolute error <=2e-8 over 1,440 days; 6e-15 to 9.2e-15 relative with herds running
- **Lives in:** eval/results/civ_slice.json, tests/test_civ_economy.py
- **Environment author:** us
- **Grader:** our invariant test
- **Can the agent influence the grader?** no
- **Changes after seeing results:** a stale conservation test formula was fixed when new sinks were added (disclosed as a test bug, not a leak)
- **Held-out discipline:** n/a (an invariant, not a sample)
- **Control:** n/a
- **Note:** An invariant check is legitimately strong for what it asserts: internal bookkeeping. It says nothing about cognition.

### assistant-parity — strong evidence

- **Claim:** the data-driven procedure engine reproduces the old generator engine exactly
- **Number:** 45 scripted requests: identical replies, identical commands, identical end state; 448 tests pass
- **Lives in:** tests/test_assistant_procedures.py
- **Environment author:** us
- **Grader:** differential test against the previous implementation
- **Can the agent influence the grader?** no
- **Changes after seeing results:** one deliberate wording change (a stale 'I ask first' line) synced across both engines so parity holds
- **Held-out discipline:** n/a (equivalence, not capability)
- **Control:** the old engine IS the oracle
- **Note:** Strong evidence of refactor equivalence. It is not evidence of capability, and should never be quoted as such.

### store-scaling — moderate evidence

- **Claim:** the claim store scales to ~112k claims with measured ingest, query and patch costs
- **Number:** see docs/revival/02 §2.8
- **Lives in:** eval/results/graph_bench.json
- **Environment author:** us (synthetic uniform data)
- **Grader:** wall-clock and memory measurement
- **Can the agent influence the grader?** no
- **Changes after seeing results:** none
- **Held-out discipline:** n/a (performance)
- **Control:** n/a
- **Note:** Sound as a performance number; synthetic and single-threaded, as the doc says.

### legacy-representation — moderate evidence

- **Claim:** the proposed records represent the same objects losslessly where the legacy TCIR was lossy or crashed
- **Number:** 22/44 nodes vs 2/3 records; lossy vs lossless; crashes vs round-trips
- **Lives in:** eval/results/representation.json, docs/revival/02 §2.3
- **Environment author:** us for the fixtures, third party for the legacy code under test
- **Grader:** round-trip equality against real legacy behaviour in a sandbox
- **Can the agent influence the grader?** no
- **Changes after seeing results:** none
- **Held-out discipline:** fixtures chosen by us
- **Control:** the legacy implementation is the comparator
- **Note:** Comparing against real code that really crashes is meaningful; fixture choice is ours.

### tensacode-overhead — strong evidence

- **Claim:** the runtime's own overhead is negligible next to any real backend
- **Number:** ~19 us per call; 0.008-0.013 ms for trivial implementations; batching preserves outputs and is 6.5x faster
- **Lives in:** eval/results/banking77.json, docs/revival/05 §6.1
- **Environment author:** us (microbenchmark)
- **Grader:** wall-clock measurement with output equality asserted
- **Can the agent influence the grader?** no
- **Changes after seeing results:** none
- **Held-out discipline:** n/a
- **Control:** direct calls to the same tiers, bypassing the runtime, as the comparator
- **Note:** A microbenchmark with the right comparator. Holds.

### language-independent-provenance — moderate evidence

- **Claim:** the grammar's coverage, measured by a set it did not author
- **Number:** 193/193 cases pass: every claim shape the civilization's minds actually speak, crossed with four diverged dialects, asserting say->hear recovers predicate and object AND that the surface is well-formed English
- **Lives in:** tests/test_civ_language_demands.py
- **Environment author:** us, but a DIFFERENT author than the grammar: the cases are generated from what research/civ_sim/minds.py actually puts into agents' mouths, not invented to exercise the parser
- **Grader:** structural assertions (round-trip recovery; no doubled copula, no bare 'am' with a third-person subject, no partitive without 'of', no raw booleans, one full stop)
- **Can the agent influence the grader?** no; the demands come from the consumer, not the implementer
- **Changes after seeing results:** the grammar was fixed in response to this set, which is the normal direction; the set itself was generated from the sim's claim shapes, not tuned
- **Held-out discipline:** independent provenance rather than a held-out split: the consumer's needs were fixed before the grammar met them
- **Control:** the fork's own 152-case act benchmark is the contrast, and it passed while this set failed
- **Note:** This is the honest coverage measurement for src/tensorcode/language, and it should be quoted instead of the act benchmark. It found five copula failures, a lost quantifier, a mis-stemmed verb, an inexpressible tense and a partitive gap: all in the commonest constructions, none on anyone's list.

### running-world-found-a-defect — weak evidence

- **Claim:** a running world exposes defects no authored set catches (the credit side of the same pattern)
- **Number:** fixing the grammar removed a spurious misunderstanding mechanism in the simulation: ungrammatical output counted as unfamiliar vocabulary, giving four common sentence types a ~45% chance of being misunderstood
- **Lives in:** docs/civ-sim/slice-results.md, research/civ_sim/language.py, docs/revival/09-language-and-induction.md
- **Environment author:** us (two independent components, one consuming the other)
- **Grader:** the simulation's own measured misunderstanding rate before and after
- **Can the agent influence the grader?** no
- **Changes after seeing results:** the defect was found by the consumer, not by the implementer's tests
- **Held-out discipline:** n/a
- **Control:** before/after comparison of the misunderstanding rate
- **Note:** Worth crediting because it is the mechanism working: one component we wrote held another to account, and it found something real. Two self-authored components checking each other is weaker than an outside grader but much stronger than one component grading itself.

### finetune-did-not-travel — weak evidence

- **Claim:** the OCR fine-tune's removal of confident false successes (2 -> 0) held up
- **Number:** it did not survive a change of environment: on computerworld all three recognizers make the same three errors, and the failures are lost hyphens, not the digit confusions the corpus targeted
- **Lives in:** docs/revival/07-vision-perception.md, eval/results/vision_desktop_e2e_pixels*.json vs the computerworld re-run
- **Environment author:** us (task) in two different simulators
- **Grader:** simulator state
- **Can the agent influence the grader?** no
- **Changes after seeing results:** the corpus was built to target the digit-confusion class observed in the OLD environment
- **Held-out discipline:** held-out fonts and apps for the crop metrics; the same 10 seeds for the end-to-end number
- **Control:** three recognizers compared in the new environment, and they agree
- **Note:** Same pattern as the language sets: a fix validated against the failure mode we had already seen, in the environment where we saw it. Moving environments produced a different failure class the corpus never covered. 2 -> 0 on ten episodes was two events; it should never have been quoted as a solved problem.

### copula-gap — self-graded

- **Claim:** the grammar's authored test sets measured its coverage
- **Number:** none of the fork's own authored sets contained a copular sentence with a third-person subject; every set it wrote passed, while a live world broke on five of its commonest sentence shapes
- **Lives in:** docs/revival/09-language-and-induction.md, research/civ_sim (17 spoken shapes)
- **Environment author:** us
- **Grader:** our expected readings
- **Can the agent influence the grader?** the same agent wrote the grammar and the cases that graded it
- **Changes after seeing results:** the bugs were found only when a running world used the grammar, not by any authored test
- **Held-out discipline:** none: the blind spot was systematic, so no amount of the same author's cases would have found it
- **Control:** the civilization's 17 spoken shapes acted as an independent set and found five real bugs
- **Note:** The sharpest demonstration in the project that self-authored test sets measure the author's imagination. A consumer we did not write found five bugs in the commonest constructions on first contact.

### open-domain — strong evidence

- **Claim:** tensacode's cognition on public open-domain benchmarks, against the same model alone and a cascade
- **Number:** rules: SQuAD2 9.1% @18.3% cov, HotpotQA 5.8% @91.3%, GSM8K 0.0% @5.3%, ARC-Easy 0% (refuses all). Model alone: 55.1% / 51.3% / 81.3% / 80.3%. Cascade WORSE than the model on 3 of 4. SQuAD2 rule arm 42.0% overall is below the 48.0% 'always abstain' floor
- **Lives in:** eval/results/open_domain.json, docs/revival/12-open-domain.md
- **Environment author:** public datasets (SQuAD 2.0, HotpotQA distractor, GSM8K, ARC-Easy)
- **Grader:** public labels; no model judged anything
- **Can the agent influence the grader?** no
- **Changes after seeing results:** none: the abstention threshold was chosen on a calibration slice disjoint from the test slice, and a test-slice threshold sweep is reported as a curve rather than tuned
- **Held-out discipline:** 300 test items per benchmark drawn from a fixed shuffle, 150 disjoint calibration items, Wilson CIs
- **Control:** majority/frequency floor and a random control on every benchmark, a forced-guess variant on ARC-Easy, plus a plain-model arm and an equal-coverage comparison
- **Note:** The least flattering measurement in the project and the most independent. It disconfirms the cascade story off home turf: a cheap tier whose abstention is capability-blind makes the system worse than the model alone.

### test-suite — self-graded

- **Claim:** the test suite passes
- **Number:** 448 tests (count varies by session as forks land work)
- **Lives in:** tests/
- **Environment author:** us
- **Grader:** our assertions
- **Can the agent influence the grader?** the same agents write the tests and the code
- **Changes after seeing results:** continuous
- **Held-out discipline:** n/a
- **Control:** n/a
- **Note:** Unit tests are a correctness floor, not evidence of capability. Several were written after the behaviour they describe.
