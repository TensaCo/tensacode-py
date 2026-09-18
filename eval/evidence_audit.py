"""Audit every measured claim in this project for who authored the environment and who graded it.

Emits eval/results/evidence_audit.json and docs/revival/11-evidence-audit.md.

The point is not to produce a flattering table. A claim is only as strong as the
independence of its grader: an environment we wrote, graded by code we wrote,
against a task we also wrote, is a statement about our own consistency.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "eval" / "results"

# verdict vocabulary
STRONG = "strong evidence"       # third-party environment or public labels, grader independent of us
MODERATE = "moderate evidence"   # environment or labels mechanically true, but task/split chosen by us
WEAK = "weak evidence"           # our environment and our grader, but with controls or state-based grading
SELF = "self-graded"             # our environment, our grader, no independent control
UNVERIFIED = "unverified"        # true by construction, or not reproducible from files on disk


@dataclass
class Row:
    id: str
    claim: str
    number: str
    source: str                   # file or doc the number lives in
    environment_author: str       # us | third party | public dataset
    grader: str                   # our code | hidden test | public labels | simulator state | self-report
    agent_can_influence_grader: str
    post_hoc_changes: str         # tuning or environment changes made after seeing held-out results
    heldout_discipline: str
    control: str                  # scrambled/random control, and what it showed
    verdict: str
    env_ours: bool                # did we author the environment the agent acts in?
    grader_independent: bool      # public dataset labels or hidden tests only: not "our code, honestly written"
    has_control: bool             # was a random / scrambled / alternative-policy control actually run?
    note: str = ""
    headline: bool = False        # is this one of the numbers we have quoted as a headline result


ROWS: list[Row] = [
    # ------------------------------------------------------------------ public datasets
    Row(
        id="banking77-cascade",
        claim="classify cascade: rules -> learned -> model, selective accuracy and coverage",
        number="rules 97.1% @5.6% cov; learned 94.3% @93.8%; cascade 94.1% @93.9%; +model 91.0% @99.5%",
        source="eval/results/banking77.json, docs/revival/05-evaluation.md §6.1",
        environment_author="public dataset (Banking77, PolyAI CC-BY-4.0, sha256 recorded)",
        grader="public labels",
        agent_can_influence_grader="no",
        post_hoc_changes="rules revised once on a TRAIN holdout, then frozen before test; threshold chosen on a 1,003-item validation split; no test-set fitting claimed",
        heldout_discipline="official held-out test split (3,080 items); Wilson 95% CIs reported",
        control="no random control run; Banking77 majority-class floor is ~1.3% (77 classes), so the numbers are far above chance",
        verdict=STRONG,
        env_ours=False,
        grader_independent=True,
        has_control=False,
        note="The strongest capability evidence in the repo, and it is about a small TF-IDF classifier, not about cognition.",
        headline=True,
    ),
    Row(
        id="banking77-escalation-harm",
        claim="zero-shot general-model escalation HURT on in-domain fine-grained intents",
        number="model correct on 38.2% (31.2-45.6) of the 173 items it answered; discarded learned argmax was 46.6% on the same items",
        source="eval/results/banking77.json, docs/revival/05-evaluation.md §6.1",
        environment_author="public dataset",
        grader="public labels",
        agent_can_influence_grader="no",
        post_hoc_changes="single zero-shot prompt, no prompt search (stated deliberately)",
        heldout_discipline="same official test split",
        control="direct comparison against the tier it replaced, on the same items",
        verdict=STRONG,
        env_ours=False,
        grader_independent=True,
        has_control=True,
        note="A negative result about our own proposed escalation, on public labels. This is the shape of evidence the rest of the project lacks.",
        headline=True,
    ),
    Row(
        id="hotpot-context-recall",
        claim="BM25 -> pack beats document order for supporting-fact recall at a token budget",
        number="41.3% / 59.3% / 74.4% recall at budgets 64 / 128 / 256 vs 8.8% / 13.8% / 24.2%",
        source="eval/results/context_hotpot.json, docs/revival/05-evaluation.md §6.2",
        environment_author="public dataset (HotpotQA distractor validation, 7,405 questions, sha256 recorded)",
        grader="public gold supporting-fact labels",
        agent_can_influence_grader="no",
        post_hoc_changes="none claimed; no threshold fitted",
        heldout_discipline="whole validation set, no split needed (no fitting)",
        control="document-order baseline on identical inputs acts as the floor",
        verdict=STRONG,
        env_ours=False,
        grader_independent=True,
        has_control=True,
        note="Measures EVIDENCE RECALL, not answer quality. 26-59% of gold evidence is still missed. The claim is narrower than 'context selection works'.",
        headline=True,
    ),
    # ------------------------------------------------------------------ our own graders, our own sets
    Row(
        id="language-act-benchmark",
        claim="the 105 regexes read 152/152 of the assistant's utterance benchmark; the grammar reads 92",
        number="regex 152/152 (100%); grammar 92/152 (60.5%)",
        source="eval/results/language_benchmark.json, docs/revival/09-language-and-induction.md",
        environment_author="us",
        grader="our expected (act, slots) labels",
        agent_can_influence_grader="no, but the SET was written alongside the regexes it grades",
        post_hoc_changes="the grammar was repeatedly fixed against failures on this same set (57.9% -> 61.8% -> 60.5% as behaviour changed)",
        heldout_discipline="none: no held-out utterance split existed until the open-vocabulary set was added",
        control="none",
        verdict=UNVERIFIED,
        env_ours=True,
        grader_independent=False,
        has_control=False,
        note="The regexes' 100% is circular by construction: the cases were authored from the regexes' own behaviour. It is a regression suite, not a measurement of language coverage. Quoting it as '100% vs 60%' overstates the regexes and understates the grammar.",
        headline=True,
    ),
    Row(
        id="language-compositional",
        claim="the grammar handles compositional constructions the regexes cannot",
        number="in-lexicon 15/15 and open-vocabulary 19/19 for the grammar; regexes 0/15 and 0/19",
        source="eval/results/language_benchmark.json",
        environment_author="us",
        grader="our expected readings",
        agent_can_influence_grader="no",
        post_hoc_changes="the open-vocabulary set was written AFTER the parent supplied four failing transcripts, then the parser was fixed until it passed them",
        heldout_discipline="none; cases and grammar authored by the same agent, in the same session",
        control="the regex arm is a genuine contrast (0/15), which is informative",
        verdict=SELF,
        env_ours=True,
        grader_independent=False,
        has_control=True,
        note="The contrast against regexes is real; the absolute 15/15 and 19/19 are fit-to-own-tests. The parser's honest generalization number is unknown.",
    ),
    Row(
        id="induction-controls",
        claim="rule induction adopts a real structural rule and REFUSES a vocabulary-reading artifact",
        number="real task: held-out 1.000, floor 0.750, random control 0.250, shifted control 0.317, rename-invariant -> adopted. Vocabulary task: 100% held-out but rename-variant -> refused",
        source="eval/results/language_benchmark.json (induction), docs/revival/09",
        environment_author="us (synthetic routing task)",
        grader="our labels",
        agent_can_influence_grader="no",
        post_hoc_changes="two of the agent's own controls were wrong first and were fixed after their tests caught them (disclosed)",
        heldout_discipline="held-out split plus three controls, fixed before adoption",
        control="random (0.250), shifted-question (0.317), rename-invariance: all run, and the rename control is what triggered a refusal",
        verdict=WEAK,
        env_ours=True,
        grader_independent=False,
        has_control=True,
        note="A synthetic task, but the discipline is the best in the repo: a 100%-accurate artifact was rejected because it failed a control. That mechanism is evidence even though the task is ours.",
        headline=True,
    ),
    # ------------------------------------------------------------------ self-authored environments
    Row(
        id="browser-agents-6task",
        claim="six browser agents score ~100% of items on fresh seeds with 0 model calls at 20-28 UI actions/s",
        number="RECOVERED at much larger n from the live-wall recordings (eval/results/browser_agents_recovered.json): 102,899 episodes, 290,909/296,674 items = 98.06%, 0 model calls. Per task: shop 100.00%, desktop 100.00%, access 99.99%, chart 99.98%, recon 99.71%, inbox 94.39%",
        source="eval/results/browser_agents.json holds only recon (125/125); the six-task bench file was overwritten. Recovered from $SP/live_run{6,7}.jsonl into eval/results/browser_agents_recovered.json",
        environment_author="us (we wrote the five web apps AND the agents that solve them)",
        grader="our code: window.__score() inside our own page, or our own checker",
        agent_can_influence_grader="not directly (the score function is not exposed to perception), but same-author bias is total",
        post_hoc_changes="YES, and the clearest case in the project: after the chart agent failed on pixel-identical bars, the chart GENERATOR was changed to redraw until the tallest bar is visibly tallest. NOW RESTORED: ties are possible again by default (99.5% on 400 fresh seeds, 2 abstentions, 0 wrong answers; 0/10 with 5 abstentions on the 5 known tie seeds), and the changed generator survives only as the opt-in ?ties=avoid. The access/chart retry relaxations were reviewed and are genuine correctness fixes: the cap on possibly-effectful attempts is still 3, and only attempts the app said had no effect are exempt.",
        heldout_discipline="fresh seeds were used as a held-out set (seeds 2000-2150 after tuning on 1-40), which is real discipline; the recovered live runs cover ~103k episodes across unseen seeds; but the apps' difficulty distribution is ours",
        control="none: no scrambled-agent or random-policy control was ever run on these apps",
        verdict=WEAK,
        env_ours=True,
        grader_independent=False,
        has_control=False,
        note="The bench file was overwritten by a sibling's single-task re-run, so the quoted per-task tallies are gone; the same claim is recoverable from the live recordings at 750x the sample size (102,899 episodes) and holds at 98.06% overall with 0 model calls. What does NOT change: we wrote the apps, the graders inside them and the agents, and the chart generator was made easier after a failure — now restored to ties-possible by default and re-measured at 99.5% on 400 fresh seeds, with the changed version kept only as `?ties=avoid`. Large n against your own exam is still your own exam. Note the live figures are continuous-run episodes, not the bench harness, so seeds differ.",
        headline=True,
    ),
    Row(
        id="desktop-chore-seed",
        claim="the desktop agent completes a multi-step shell chore and verifies it",
        number="120/120 items over 40 episodes (Seed); 10/10 episodes and 30/30 checks (computerworld)",
        source="eval/results/browser_agents.json (historical), eval/results/computerworld.json",
        environment_author="third party for the engine (the user's Seed, then the user's computerworld), us for the task and the world definition",
        grader="simulator state (file contents and git history read back through a privileged API), not the agent's report",
        agent_can_influence_grader="the agent writes the files being graded, but cannot alter the checker",
        post_hoc_changes="scoring was changed mid-project when Seed kept git history after folder deletion (counting only commits made during the episode); a random ls marker was removed when porting, which is what made episodes reproducible",
        heldout_discipline="fresh seeds; on computerworld, episodes are exactly reproducible via state_hash",
        control="none",
        verdict=WEAK,
        env_ours=True,
        grader_independent=False,
        has_control=False,
        note="State-based grading is a genuine step up from self-report. The task is still one we wrote for an agent we wrote.",
        headline=True,
    ),
    Row(
        id="computerworld-speed",
        claim="the same desktop chore runs ~450x faster per episode on computerworld than on the browser-based simulator",
        number="0.0167 s/episode and 56.7-60.2 episodes/s vs 7.07 s/episode; perception 0.42 ms; 30/30 items; determinism confirmed by state_hash",
        source="eval/results/computerworld.json, docs/revival/10-computerworld.md",
        environment_author="third party (the user's engine)",
        grader="simulator state plus the engine's own state_hash",
        agent_can_influence_grader="no",
        post_hoc_changes="none; the comparison is across different seed ranges, which the doc labels as order-of-magnitude rather than paired",
        heldout_discipline="n=10 episodes for the engine row; determinism checked on 2 seeds",
        control="the old path is the baseline; a state_hash mismatch check acts as a negative control on determinism",
        verdict=MODERATE,
        env_ours=False,
        grader_independent=False,
        has_control=True,
        note="A performance and determinism claim, not a capability claim, and it is sound as such. n=10 is small.",
        headline=True,
    ),
    Row(
        id="perception-fusion",
        claim="fusing DOM and vision lifts word coverage; vision repairs a thinned accessibility tree",
        number="words covered 0.66-0.78 (DOM) -> 0.95-0.99 (fused); icon naming on a stripped tree 0.00 -> 0.73/0.24/0.09",
        source="eval/results/perception_fusion.json, docs/revival/07 §7.6.2",
        environment_author="us (frames captured from our apps and the user's Seed machines)",
        grader="the DOM read at the same instant",
        agent_can_influence_grader="no, but the DOM is BOTH a provider and the ground truth, so the DOM row is 1.00 recall by construction",
        post_hoc_changes="several perception changes were made after held-out metrics were first computed (disclosed in docs/revival/07 §7.4: weak-detection retention, prompt pattern, phrase splitting, title fallback, recognizer swap), plus two more after the final run",
        heldout_discipline="explicit tune / test_app / test_os splits, with 2 fonts and 8 apps held out for the recognizer work",
        control="a dom-degraded arm (icon names stripped) serves as an ablation; no random control",
        verdict=WEAK,
        env_ours=True,
        grader_independent=False,
        has_control=True,
        note="The fused-vs-DOM comparison is meaningful. Any row where DOM is the ground truth cannot be read as DOM's accuracy.",
        headline=True,
    ),
    Row(
        id="recognizer-finetune",
        claim="a wide corpus of free labels fixes the systematic OCR misreads",
        number="held-out fonts exact 0.668 -> 0.794, digits 0.828 -> 0.882, tilde words 0/11 -> 8/11",
        source="eval/results/recognizer_comparison.json, docs/revival/07 §7.6.6",
        environment_author="us (rendered crops) plus DOM-labelled app text",
        grader="mechanically exact labels: the string the agent typed, or the string the DOM reported",
        agent_can_influence_grader="no",
        post_hoc_changes="an apps-only corpus was tried first and made unseen fonts worse; the wide corpus was the fix (both reported)",
        heldout_discipline="2 fonts and 8 apps never trained on",
        control="the apps-only arm is a genuine negative control and it regressed below baseline on unseen fonts",
        verdict=MODERATE,
        env_ours=True,
        grader_independent=False,
        has_control=True,
        note="The labels here are not our judgement, they are mechanically true, which makes this one of the better-grounded measurements in the perception work.",
        headline=True,
    ),
    Row(
        id="false-success-fix",
        claim="the retrained recognizer removes confident false successes end to end",
        number="false successes 2 -> 0; verified episodes 6/10 -> 7/10; checks 21/30 unchanged",
        source="eval/results/vision_desktop_e2e_pixels*.json, docs/revival/07 §7.6.4",
        environment_author="us (task) in a third-party simulator",
        grader="simulator state",
        agent_can_influence_grader="no",
        post_hoc_changes="three earlier guards were tried and all measured worse (cross-place digit check, corroboration policy, confidence gate); all three are reported, two kept behind flags",
        heldout_discipline="same 10 seeds throughout, which is a fixed set but a tiny one",
        control="three failed guards act as alternatives; no random control",
        verdict=WEAK,
        env_ours=True,
        grader_independent=False,
        has_control=True,
        note="n=10 episodes. 2 -> 0 on ten episodes is 2 events. Directionally supported by the crop-level numbers, which is the real evidence.",
        headline=True,
    ),
    Row(
        id="real-desktop-acting",
        claim="the agent completed a real GTK task on a real desktop through the accessibility tree; fusion failed the same task",
        number="AT-SPI verified, 2.1 s, perception 48-92 ms; AT-SPI+vision refused to type, unverified",
        source="eval/results/real_acting_report.json, docs/revival/07 §7.6.3",
        environment_author="us (we wrote the scratch app) on a real OS",
        grader="the app's own status label, read back by the agent",
        agent_can_influence_grader="the agent writes the label's content indirectly by acting; it reports what it reads",
        post_hoc_changes="focus verification and coordinate-frame handling were added in response to failures during this very test",
        heldout_discipline="none; n=1 task, 1 window, 2 providers",
        control="the fused arm is a contrast",
        verdict=WEAK,
        env_ours=True,
        grader_independent=False,
        has_control=True,
        note="The VALUE here is the four failure modes it exposed (X11 focus stealing, AT-SPI cache, coordinate frames, fusion rivalry), not the pass/fail. Those are real-world facts simulation hid.",
    ),
    Row(
        id="invariant-check",
        claim="task-stated invariants catch the confident-misread case without refusing correct reads",
        number="on 11 recorded frames: 1 refusal (the false-success frame), 7 corroborated, 0 corroborated-but-wrong",
        source="eval/results/invariant_eval.json, docs/revival/07 §7.6.7",
        environment_author="us",
        grader="our recorded frames and known seeds",
        agent_can_influence_grader="no",
        post_hoc_changes="the checker had to be told which lines are data after it flagged 4 of 11 (the prompt carries the prior task's directory)",
        heldout_discipline="none; these are the frames on which the failure was originally observed",
        control="none",
        verdict=SELF,
        env_ours=True,
        grader_independent=False,
        has_control=False,
        note="n=11, selected because they contained the failure. Suggestive mechanism, no generalization evidence.",
    ),
    Row(
        id="longhorizon",
        claim="the assistant can do sophisticated multi-step work (CAD, EEG, coding with feedback)",
        number="every arm failed: as-is 0.0, improved 0.0, plain-ReAct 0.25 on CAD; coding 0.0 (12 hidden tests failed). 9-32 model calls, 378-2836 s, most runs hit a wall-clock cap",
        source="eval/results/longhorizon.json, docs/revival/08-long-horizon.md",
        environment_author="us for the tasks, third party for the tools (cadquery/trimesh, pytest, real sandbox shell)",
        grader="HIDDEN tests and geometric probes the agent cannot see or edit",
        agent_can_influence_grader="no, by design (hidden test directory, pre-registered checks)",
        post_hoc_changes="none possible on the held-out task; checks were pre-registered before runs",
        heldout_discipline="best in the project: tasks authored before tuning, hidden graders, held-out split, budgets, and a plain-ReAct baseline with the same model",
        control="plain ReAct with the same model is the control, and it BEAT both tensorcode arms (0.25 vs 0.0)",
        verdict=STRONG,
        env_ours=True,
        grader_independent=True,
        has_control=True,
        note="The most disciplined evaluation we ran, and it is a failure: our structure did not beat an unstructured loop, and nothing passed. Also measured: ~200 s per model call under contention, so the binding constraint was model throughput.",
        headline=True,
    ),
    Row(
        id="recovery-sim",
        claim="the recovery agent never duplicates an effect under a correctly described target system",
        number="0 duplicates (0-0.08%) vs naive retry 4.76% and verify-then-retry 12.74%; 0.92% duplicates when the agent is told keys are honoured and they are not",
        source="eval/results/recovery.json, docs/revival/05 §6.3",
        environment_author="us (simulation, 5,000 sampled worlds, fault model written by the same author as the agent)",
        grader="our simulator's ground truth",
        agent_can_influence_grader="no",
        post_hoc_changes="none claimed",
        heldout_discipline="all policies run on identical sampled worlds; a mis-specified-facts sensitivity arm is included",
        control="three baseline policies plus the mis-specified-facts arm, which BREAKS the result (duplicates return)",
        verdict=WEAK,
        env_ours=True,
        grader_independent=False,
        has_control=True,
        note="The doc already states the honest conclusion: 'the safety comes from stated facts, not from TensaCode'. Self-authored fault model, but with the disconfirming arm run and reported.",
        headline=True,
    ),
    Row(
        id="civ-sim-dynamics",
        claim="emergent money, deforestation caps, emergent settlements, ideology equilibrium, 43% belief correctness after the parser swap",
        number="wood becomes numeraire in 1 of 3 seeds; Gini 0.38->0.70; 0 raids in 2 of 3 seeds; belief-vs-truth 43.1% correct (was 0%); 14.6% of utterances not understood",
        source="eval/results/civ_slice.json, civ_slice_dynamics.json, civ_language.json, docs/civ-sim/slice-results.md",
        environment_author="us, entirely: world, economy, minds, grammar and graders",
        grader="our simulator's own state",
        agent_can_influence_grader="the agents ARE the system being measured; there is no external truth",
        post_hoc_changes="many, and disclosed: commons rationing, starvation myopia, caravan loss cap, livestock cap, the 240/72-squared/3-band default chosen by measurement after a 2-band world collapsed",
        heldout_discipline="none applicable; 3 seeds, determinism per seed",
        control="a no-minds arm exists for COST comparison; no control for the dynamics claims",
        verdict=SELF,
        env_ours=True,
        grader_independent=False,
        has_control=False,
        note="Zero external validity by construction. The internally valid parts are the invariants (goods conserved to ~1e-14) and the corrections it published against its own earlier claims (the 'rumours diverge' story was measured false).",
        headline=True,
    ),
    Row(
        id="civ-conservation",
        claim="the simulation conserves all goods",
        number="absolute error <=2e-8 over 1,440 days; 6e-15 to 9.2e-15 relative with herds running",
        source="eval/results/civ_slice.json, tests/test_civ_economy.py",
        environment_author="us",
        grader="our invariant test",
        agent_can_influence_grader="no",
        post_hoc_changes="a stale conservation test formula was fixed when new sinks were added (disclosed as a test bug, not a leak)",
        heldout_discipline="n/a (an invariant, not a sample)",
        control="n/a",
        verdict=MODERATE,
        env_ours=True,
        grader_independent=False,
        has_control=False,
        note="An invariant check is legitimately strong for what it asserts: internal bookkeeping. It says nothing about cognition.",
    ),
    Row(
        id="assistant-parity",
        claim="the data-driven procedure engine reproduces the old generator engine exactly",
        number="45 scripted requests: identical replies, identical commands, identical end state; 448 tests pass",
        source="tests/test_assistant_procedures.py",
        environment_author="us",
        grader="differential test against the previous implementation",
        agent_can_influence_grader="no",
        post_hoc_changes="one deliberate wording change (a stale 'I ask first' line) synced across both engines so parity holds",
        heldout_discipline="n/a (equivalence, not capability)",
        control="the old engine IS the oracle",
        verdict=STRONG,
        env_ours=True,
        grader_independent=False,
        has_control=True,
        note="Strong evidence of refactor equivalence. It is not evidence of capability, and should never be quoted as such.",
    ),
    Row(
        id="store-scaling",
        claim="the claim store scales to ~112k claims with measured ingest, query and patch costs",
        number="see docs/revival/02 §2.8",
        source="eval/results/graph_bench.json",
        environment_author="us (synthetic uniform data)",
        grader="wall-clock and memory measurement",
        agent_can_influence_grader="no",
        post_hoc_changes="none",
        heldout_discipline="n/a (performance)",
        control="n/a",
        verdict=MODERATE,
        env_ours=True,
        grader_independent=False,
        has_control=False,
        note="Sound as a performance number; synthetic and single-threaded, as the doc says.",
    ),
    Row(
        id="legacy-representation",
        claim="the proposed records represent the same objects losslessly where the legacy TCIR was lossy or crashed",
        number="22/44 nodes vs 2/3 records; lossy vs lossless; crashes vs round-trips",
        source="eval/results/representation.json, docs/revival/02 §2.3",
        environment_author="us for the fixtures, third party for the legacy code under test",
        grader="round-trip equality against real legacy behaviour in a sandbox",
        agent_can_influence_grader="no",
        post_hoc_changes="none",
        heldout_discipline="fixtures chosen by us",
        control="the legacy implementation is the comparator",
        verdict=MODERATE,
        env_ours=True,
        grader_independent=False,
        has_control=True,
        note="Comparing against real code that really crashes is meaningful; fixture choice is ours.",
    ),
    Row(
        id="tensorcode-overhead",
        claim="the runtime's own overhead is negligible next to any real backend",
        number="~19 us per call; 0.008-0.013 ms for trivial implementations; batching preserves outputs and is 6.5x faster",
        source="eval/results/banking77.json, docs/revival/05 §6.1",
        environment_author="us (microbenchmark)",
        grader="wall-clock measurement with output equality asserted",
        agent_can_influence_grader="no",
        post_hoc_changes="none",
        heldout_discipline="n/a",
        control="direct calls to the same tiers, bypassing the runtime, as the comparator",
        verdict=STRONG,
        env_ours=True,
        grader_independent=False,
        has_control=True,
        note="A microbenchmark with the right comparator. Holds.",
        headline=True,
    ),
    Row(
        id="language-independent-provenance",
        claim="the grammar's coverage, measured by a set it did not author",
        number="193/193 cases pass: every claim shape the civilization's minds actually speak, crossed with four diverged dialects, asserting say->hear recovers predicate and object AND that the surface is well-formed English",
        source="tests/test_civ_language_demands.py",
        environment_author="us, but a DIFFERENT author than the grammar: the cases are generated from what research/civ_sim/minds.py actually puts into agents' mouths, not invented to exercise the parser",
        grader="structural assertions (round-trip recovery; no doubled copula, no bare 'am' with a third-person subject, no partitive without 'of', no raw booleans, one full stop)",
        agent_can_influence_grader="no; the demands come from the consumer, not the implementer",
        post_hoc_changes="the grammar was fixed in response to this set, which is the normal direction; the set itself was generated from the sim's claim shapes, not tuned",
        heldout_discipline="independent provenance rather than a held-out split: the consumer's needs were fixed before the grammar met them",
        control="the fork's own 152-case act benchmark is the contrast, and it passed while this set failed",
        verdict=MODERATE,
        env_ours=True,
        grader_independent=False,
        has_control=True,
        note="This is the honest coverage measurement for src/tensorcode/language, and it should be quoted instead of the act benchmark. It found five copula failures, a lost quantifier, a mis-stemmed verb, an inexpressible tense and a partitive gap: all in the commonest constructions, none on anyone's list.",
        headline=True,
    ),
    Row(
        id="running-world-found-a-defect",
        claim="a running world exposes defects no authored set catches (the credit side of the same pattern)",
        number="fixing the grammar removed a spurious misunderstanding mechanism in the simulation: ungrammatical output counted as unfamiliar vocabulary, giving four common sentence types a ~45% chance of being misunderstood",
        source="docs/civ-sim/slice-results.md, research/civ_sim/language.py, docs/revival/09-language-and-induction.md",
        environment_author="us (two independent components, one consuming the other)",
        grader="the simulation's own measured misunderstanding rate before and after",
        agent_can_influence_grader="no",
        post_hoc_changes="the defect was found by the consumer, not by the implementer's tests",
        heldout_discipline="n/a",
        control="before/after comparison of the misunderstanding rate",
        verdict=WEAK,
        note="Worth crediting because it is the mechanism working: one component we wrote held another to account, and it found something real. Two self-authored components checking each other is weaker than an outside grader but much stronger than one component grading itself.",
        env_ours=True,
        grader_independent=False,
        has_control=True,
    ),
    Row(
        id="finetune-did-not-travel",
        claim="the OCR fine-tune's removal of confident false successes (2 -> 0) held up",
        number="it did not survive a change of environment: on computerworld all three recognizers make the same three errors, and the failures are lost hyphens, not the digit confusions the corpus targeted",
        source="docs/revival/07-vision-perception.md, eval/results/vision_desktop_e2e_pixels*.json vs the computerworld re-run",
        environment_author="us (task) in two different simulators",
        grader="simulator state",
        agent_can_influence_grader="no",
        post_hoc_changes="the corpus was built to target the digit-confusion class observed in the OLD environment",
        heldout_discipline="held-out fonts and apps for the crop metrics; the same 10 seeds for the end-to-end number",
        control="three recognizers compared in the new environment, and they agree",
        verdict=WEAK,
        env_ours=True,
        grader_independent=False,
        has_control=True,
        note="Same pattern as the language sets: a fix validated against the failure mode we had already seen, in the environment where we saw it. Moving environments produced a different failure class the corpus never covered. 2 -> 0 on ten episodes was two events; it should never have been quoted as a solved problem.",
        headline=True,
    ),
    Row(
        id="copula-gap",
        claim="the grammar's authored test sets measured its coverage",
        number="none of the fork's own authored sets contained a copular sentence with a third-person subject; every set it wrote passed, while a live world broke on five of its commonest sentence shapes",
        source="docs/revival/09-language-and-induction.md, research/civ_sim (17 spoken shapes)",
        environment_author="us",
        grader="our expected readings",
        agent_can_influence_grader="the same agent wrote the grammar and the cases that graded it",
        post_hoc_changes="the bugs were found only when a running world used the grammar, not by any authored test",
        heldout_discipline="none: the blind spot was systematic, so no amount of the same author's cases would have found it",
        control="the civilization's 17 spoken shapes acted as an independent set and found five real bugs",
        verdict=SELF,
        env_ours=True,
        grader_independent=False,
        has_control=True,
        note="The sharpest demonstration in the project that self-authored test sets measure the author's imagination. A consumer we did not write found five bugs in the commonest constructions on first contact.",
        headline=True,
    ),
    Row(
        id="open-domain",
        claim="tensorcode's cognition on public open-domain benchmarks, against the same model alone and a cascade",
        number=("rules: SQuAD2 9.1% @18.3% cov, HotpotQA 5.8% @91.3%, GSM8K 0.0% @5.3%, ARC-Easy 0% (refuses all). "
                "Model alone: 55.1% / 51.3% / 81.3% / 80.3%. Cascade WORSE than the model on 3 of 4. "
                "SQuAD2 rule arm 42.0% overall is below the 48.0% 'always abstain' floor"),
        source="eval/results/open_domain.json, docs/revival/12-open-domain.md",
        environment_author="public datasets (SQuAD 2.0, HotpotQA distractor, GSM8K, ARC-Easy)",
        grader="public labels; no model judged anything",
        agent_can_influence_grader="no",
        post_hoc_changes="none: the abstention threshold was chosen on a calibration slice disjoint from the test slice, and a test-slice threshold sweep is reported as a curve rather than tuned",
        heldout_discipline="300 test items per benchmark drawn from a fixed shuffle, 150 disjoint calibration items, Wilson CIs",
        control="majority/frequency floor and a random control on every benchmark, a forced-guess variant on ARC-Easy, plus a plain-model arm and an equal-coverage comparison",
        verdict=STRONG,
        env_ours=False,
        grader_independent=True,
        has_control=True,
        note="The least flattering measurement in the project and the most independent. It disconfirms the cascade story off home turf: a cheap tier whose abstention is capability-blind makes the system worse than the model alone.",
        headline=True,
    ),
    Row(
        id="test-suite",
        claim="the test suite passes",
        number="448 tests (count varies by session as forks land work)",
        source="tests/",
        environment_author="us",
        grader="our assertions",
        agent_can_influence_grader="the same agents write the tests and the code",
        post_hoc_changes="continuous",
        heldout_discipline="n/a",
        control="n/a",
        verdict=SELF,
        env_ours=True,
        grader_independent=False,
        has_control=False,
        note="Unit tests are a correctness floor, not evidence of capability. Several were written after the behaviour they describe.",
    ),
]


def summarize(rows: list[Row]) -> dict:
    head = [r for r in rows if r.headline]
    ours_env = [r for r in rows if r.env_ours]
    ours_grader = [r for r in rows if not r.grader_independent]
    both = [r for r in rows if r.env_ours and not r.grader_independent]
    head_both = [r for r in head if r.env_ours and not r.grader_independent]
    head_public = [r for r in head if r.grader_independent]
    verdicts = {v: sum(1 for r in rows if r.verdict == v) for v in (STRONG, MODERATE, WEAK, SELF, UNVERIFIED)}
    head_verdicts = {v: sum(1 for r in head if r.verdict == v) for v in (STRONG, MODERATE, WEAK, SELF, UNVERIFIED)}
    return {
        "rows": len(rows),
        "headline_rows": len(head),
        "environment_written_by_us": f"{len(ours_env)}/{len(rows)}",
        "graded_by_our_own_code": f"{len(ours_grader)}/{len(rows)}",
        "both_environment_and_grader_ours": f"{len(both)}/{len(rows)}",
        "headline_both_ours": f"{len(head_both)}/{len(head)}",
        "headline_public_or_hidden_grader": f"{len(head_public)}/{len(head)}",
        "verdicts": verdicts,
        "headline_verdicts": head_verdicts,
        "post_hoc_environment_changes": [r.id for r in rows if r.post_hoc_changes.startswith("YES")],
        "not_reproducible_from_disk": [r.id for r in rows if "NOT ON DISK" in r.source],
        "controls_run": [r.id for r in rows if r.has_control],
        "no_control_run": [r.id for r in rows if not r.has_control],
    }


def markdown(rows: list[Row], s: dict) -> str:
    L: list[str] = []
    A = L.append
    A("# 11. Evidence audit: who wrote the environment, and who graded the answer")
    A("")
    A(f"**Headline: of the {s['headline_rows']} results this project has quoted as headlines, "
      f"{s['headline_both_ours']} run in an environment we wrote AND are graded by code we wrote. "
      f"Only {s['headline_public_or_hidden_grader']} are graded by a public dataset's labels or by hidden tests we cannot see.**")
    A("")
    A("The conclusions that survive that filter are mostly failures or narrow component results: a TF-IDF "
      "classifier beating a general model on in-domain intents (public labels); every arm of the long-horizon "
      "evaluation failing while an unstructured ReAct loop scored higher (hidden tests); and, in "
      "[12 — open-domain benchmarks](12-open-domain.md), the no-model arm answering almost nothing off its home "
      "turf while the proposed cascade comes out *worse* than the same model prompted plainly on 3 of 4 public "
      "benchmarks.")
    A("")
    A("This file exists because the user asked whether wireheading had crept in. It had, in three specific ways, "
      "listed first so they are not buried.")
    A("")
    A("## The three real problems")
    A("")
    A("**1. One environment was changed after the agent failed in it. Now restored, and re-measured.** After "
      "the chart agent mis-read two pixel-identical bars, the generator was changed to redraw until the tallest "
      "bar is visibly tallest, and the score went to 100%. The original generator is now the DEFAULT again; the "
      "changed one survives only as `?ties=avoid`, a separate configuration that must not be quoted as the "
      "headline. Re-measured on fresh seeds "
      "(`eval/results/chart_environment_sweep400.json`, `chart_environment.py`):")
    A("")
    A("| Generator | 400 fresh seeds | The 5 known tie seeds |")
    A("| --- | ---: | ---: |")
    A("| **Original (ties possible) — the honest headline** | **99.5%** (796/800), 2 near-tie abstentions, 0 wrong answers | **0/10**, 5 abstentions, 0 wrong answers |")
    A("| Changed (ties avoided) | 100.0% (800/800) | 10/10 |")
    A("")
    A("So the change was worth 0.5 points on random seeds, because a pixel tie occurs in about 0.5% of them — "
      "the inflation was small. But on the seeds where it bites, the original environment is genuinely "
      "unanswerable and the agent scores zero while abstaining every time and never guessing wrong. That is the "
      "behaviour worth reporting, and it is only visible in the environment we did not edit. The practice was "
      "still wrong even though the effect was small.")
    A("")
    A("**2. The most-quoted number in the project had been overwritten, and is now recovered at larger n.** "
      "\"Six agents, ~100% of items, 0 model calls\" was quoted throughout this work, but "
      "`eval/results/browser_agents.json` holds only one task (recon, 125/125): a sibling's single-task re-run "
      "overwrote the six-task bench file. Rebuilding it from the live-wall recordings gives **102,899 episodes and "
      "290,909/296,674 items (98.06%) with 0 model calls** — shop and desktop at 100.00%, access 99.99%, chart "
      "99.98%, recon 99.71%, inbox 94.39% — now written to `eval/results/browser_agents_recovered.json`. The "
      "sample is ~750x the bench run, and none of it changes the fact that we wrote the apps, the graders inside "
      "them and the agents. A very large score against an exam you wrote yourself is still an exam you wrote "
      "yourself.")
    A("")
    A("**3. The language benchmark grades the regexes against cases written from the regexes.** "
      "`language_benchmark.json` reports the 105 regexes at 152/152 (100%) and the grammar at 92/152 (60.5%). "
      "The 152 cases were authored alongside those regexes, so the 100% is true by construction. It stays as a "
      "**regression suite** and stops being quoted as coverage. The replacement already exists and has "
      "independent provenance: `tests/test_civ_language_demands.py`, 193 cases generated from every claim shape "
      "the civilization's minds actually speak across four diverged dialects, which found five copula failures, "
      "a lost quantifier, a mis-stemmed verb, an inexpressible tense and a partitive gap while the fork's own "
      "suite passed. Quote that, and the assistant's 152 utterances, which predate the grammar.")
    A("")
    A("**The two relaxed retry policies, reviewed: both are genuine correctness fixes, not difficulty "
      "reductions.** In `tasks/access.py` and `tasks/chart.py` the cap on attempts that *might have applied an "
      "effect* is still 3; what changed is that a 503 explicitly saying nothing was saved no longer counts "
      "toward it (total attempts capped at 6). Retrying an attempt the app states had no effect cannot duplicate "
      "an effect, and refusing to is a correctness error, not caution — it is the same principle the recovery "
      "agent already encodes. Duplicate effects remain 0 in every result file that records them. No restoration "
      "needed.")
    A("")
    A("Two further practices are worth naming, both already disclosed in their own docs but easy to lose: "
      "perception rules were changed after held-out metrics were first computed (`docs/revival/07` §7.4 lists five "
      "such changes, plus two more after the final run), and the civilization's defaults (population, world size, "
      "band count) were chosen by measurement after a smaller world collapsed.")
    A("")
    A("## The pattern: self-authored sets pass while running worlds fail")
    A("")
    A("Three independent cases in this session, from three different directions, are the same failure:")
    A("")
    A("1. **The grammar's own test sets.** None of the sets the language fork authored contained a copular "
      "sentence with a third-person subject. Every set it wrote passed. The first consumer it did not write — "
      "the civilization, with 17 spoken sentence shapes — broke on five of its commonest constructions "
      "immediately. The blind spot was systematic, so more cases by the same author would never have found it.")
    A("2. **The OCR fine-tune.** \"2 false successes -> 0\" was measured on the ten episodes of the environment "
      "where those two failures had been observed, with a corpus built to target the digit confusions seen there. "
      "Moved to `computerworld`, all three recognizers make the same three errors, and the failures are lost "
      "hyphens — a class the corpus never covered. The fix was real for the failure we had already seen.")
    A("3. **The open-domain cascade.** The cascade looked good on Banking77, where the abstention threshold was "
      "fitted on in-domain validation data. On four public benchmarks it is *worse* than the same model prompted "
      "plainly on three of them, because the abstention signal predicts lexical overlap rather than whether the "
      "cheap tier can answer. See [12](12-open-domain.md).")
    A("")
    A("The common shape: a mechanism is validated against the failure mode we had already seen, in the "
      "environment where we saw it, by the author of both. Every one of these passed its own tests. What broke "
      "them was contact with something we did not write — a consumer, a different simulator, a public dataset.")
    A("")
    A("**The credit side, and the cheapest fix available.** The same session produced the counter-example: "
      "`tests/test_civ_language_demands.py` holds 193 cases with *independent provenance* — not invented to "
      "exercise the grammar, but generated from every claim shape the civilization's minds actually speak, "
      "across four diverged dialects. It found five copula failures, a lost quantifier, a mis-stemmed verb, an "
      "inexpressible tense and a partitive gap, all in the commonest constructions, while the grammar's own "
      "152-case suite passed. Fixing those also removed a spurious mechanism in the simulation, where "
      "ungrammatical output was being counted as unfamiliar vocabulary and gave four common sentence types a "
      "~45% chance of being \"misunderstood\".")
    A("")
    A("Two components we wrote, one consuming the other, is weaker evidence than an outside grader and far "
      "stronger than one component grading itself. It is also nearly free. **The practical rule this session "
      "earns: a capability claim should be graded by the consumer that needs it, never by a set written "
      "alongside the implementation.** For the grammar that means quoting the 193-case demand set and the "
      "assistant's 152 utterances (which predate the grammar), and keeping the fork's own act benchmark as the "
      "regression suite it is.")
    A("")
    A("## What good practice we did follow")
    A("")
    A("- **Hidden graders, once.** The long-horizon tasks use hidden tests and pre-registered geometric checks the "
      "agent cannot read or edit, plus a plain-ReAct control with the same model. It is the only evaluation here "
      "built to be failed, and it failed.")
    A("- **Disconfirming arms.** The recovery simulation includes an arm where the agent is told something false, "
      "and duplicates return (0 -> 0.92%). The doc states that the safety comes from the stated facts, not from "
      "the framework.")
    A("- **A control that triggered a refusal.** Rule induction rejected a 100%-accurate artifact because its "
      "verdicts changed under renaming. That is the mechanism working against its author's interest.")
    A("- **Mechanically true labels.** The OCR corpus is labelled by the string the agent typed or the string the "
      "DOM reported, not by our judgement, and an apps-only corpus was reported even though it regressed.")
    A("- **State-based grading.** The desktop chore is graded by reading the simulator's files and git history, "
      "not by the agent's report. On `computerworld` the episode is reproducible by `state_hash`.")
    A("")
    A("## Structural weaknesses that no single fix addresses")
    A("")
    A("- **Same-author bias is total on the browser agents:** we wrote the five web apps, the graders inside them, "
      "the agents, and the difficulty distribution. Robustness to fresh seeds is real; robustness to an app we did "
      "not write is untested.")
    A("- **The perception fusion table cannot measure DOM.** The DOM read is simultaneously one provider and the "
      "ground truth, so its row is 1.00 recall by construction. Only the vision and fused rows are measurements.")
    A("- **Small n on several headline claims:** false successes 2 -> 0 is two events over ten episodes; the "
      "computerworld comparison is ten episodes; the real-desktop acting test is one task in one window.")
    A("- **No random or scrambled control was ever run on the browser agents, the desktop chore, the civilization "
      "dynamics, or the invariant check.** " f"{len(s['no_control_run'])} of {s['rows']} audited rows have no control.")
    A("- **The civilization has no external validity by construction.** Its defensible parts are its invariants "
      "(goods conserved to ~1e-14) and its published corrections against its own earlier claims.")
    A("")
    A("## Counts")
    A("")
    A("| Measure | Value |")
    A("| --- | ---: |")
    A(f"| Audited claims | {s['rows']} |")
    A(f"| Environment written by us | {s['environment_written_by_us']} |")
    A(f"| Graded by our own code | {s['graded_by_our_own_code']} |")
    A(f"| Both environment and grader ours | {s['both_environment_and_grader_ours']} |")
    A(f"| Headline claims, both ours | {s['headline_both_ours']} |")
    A(f"| Headline claims, public labels or hidden tests | {s['headline_public_or_hidden_grader']} |")
    A("")
    A("| Verdict | All claims | Headline claims |")
    A("| --- | ---: | ---: |")
    for v in (STRONG, MODERATE, WEAK, SELF, UNVERIFIED):
        A(f"| {v} | {s['verdicts'][v]} | {s['headline_verdicts'][v]} |")
    A("")
    A("## Every claim, one row each")
    A("")
    A("| # | Claim | Environment author | Grader | Held-out discipline | Control | Verdict |")
    A("| --- | --- | --- | --- | --- | --- | --- |")
    for i, r in enumerate(rows, 1):
        star = " ★" if r.headline else ""
        A(f"| {i}{star} | **{r.id}**: {r.claim} | {r.environment_author} | {r.grader} | {r.heldout_discipline} | {r.control} | {r.verdict} |")
    A("")
    A("★ = quoted as a headline result.")
    A("")
    A("## Detail per claim")
    A("")
    for r in rows:
        A(f"### {r.id} — {r.verdict}")
        A("")
        A(f"- **Claim:** {r.claim}")
        A(f"- **Number:** {r.number}")
        A(f"- **Lives in:** {r.source}")
        A(f"- **Environment author:** {r.environment_author}")
        A(f"- **Grader:** {r.grader}")
        A(f"- **Can the agent influence the grader?** {r.agent_can_influence_grader}")
        A(f"- **Changes after seeing results:** {r.post_hoc_changes}")
        A(f"- **Held-out discipline:** {r.heldout_discipline}")
        A(f"- **Control:** {r.control}")
        A(f"- **Note:** {r.note}")
        A("")
    return "\n".join(L)


def main() -> None:
    s = summarize(ROWS)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "evidence_audit.json").write_text(json.dumps({"summary": s, "rows": [asdict(r) for r in ROWS]}, indent=1))
    (ROOT / "docs" / "revival" / "11-evidence-audit.md").write_text(markdown(ROWS, s))
    print(json.dumps(s, indent=1))


if __name__ == "__main__":
    main()
