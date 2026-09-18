# Long-horizon work: what useful agent work looks like, and how we test for it

This document covers three things: what the serious agent benchmarks actually measure and
how they get gamed; the rules we adopted so this evaluation cannot be wireheaded; and the
nine long-horizon tasks we run, with their pre-registered checkers and results.

Everything here is measured on this machine (NVIDIA GB10, aarch64, 121 GB unified memory)
with a local teacher model and no paid API. Numbers that are estimates say so.

## 1. What useful agent work looks like

Read across the current benchmarks, "useful" converges on a few properties: the task comes
from real work, it takes many steps in a real environment, and success is decided by
executing something rather than by asking a model whether the work looks good.

| Benchmark | Task shape | How success is checked | What it teaches us |
| --- | --- | --- | --- |
| [Agents' Last Exam (ALE)](https://arxiv.org/abs/2606.05405) | Long-horizon, economically valuable tasks from real professional work; 1,000+ tasks across 55 subfields in 13 industry clusters, derived from the O\*NET/SOC occupational taxonomy; ~150 tasks public | Executable verification, not human panels or model judges; each task carries a hidden reference staged only *after* the agent finishes, plus an `evaluate()` scoring 0–1 | The bar for "useful" is an occupational task with an objective grader. Hardest tier is nowhere near saturated (reported average full pass rate 2.6% in the search summary, "below 1%" in the abstract's own wording) |
| [Terminal-Bench 2.0](https://arxiv.org/pdf/2601.11868) | 89 tasks in real terminal environments, 16 categories, each with a Docker environment, human-written oracle solution and test suite | The task's own pytest suite inside the container | A clean template: instruction + environment + hidden tests + oracle. Frontier agents score under 65% |
| [SWE-bench Pro](https://static.scale.com/uploads/654197dc94d34f66c0f5184e/SWEAP_Eval_Scale%20(9).pdf) | Long-horizon repository issues; public (11 repos), held-out (12 repos) and commercial (18 proprietary repos) partitions | Hidden tests in the repo | Public-vs-private gap is the honest measure of generalization: ≤23.3% public vs ≤17.8% commercial in the paper's own numbers |
| [SWE-bench Verified](https://github.com/SWE-bench/SWE-bench/issues/465) and [SWE-bench Pro OSS](https://github.com/scaleapi/SWE-bench_Pro-os/issues/93) | Same shape | Same | The canonical gaming story: containers kept future git objects, so agents read the fix out of `git log`/`git show`. Reported drops after lockdown are large (one blog reports 78.8% → 57.3%) |
| [OSWorld / OSWorld-Verified](https://xlang.ai/blog/osworld-verified) | Open-ended computer-use tasks in real desktop VMs | Per-task scripts that inspect final machine state | Even state-based checkers rot: 300+ issues fixed, and broken selectors had caused a reported 28% underestimation in one section |
| [TheAgentCompany](https://arxiv.org/pdf/2412.14161v1) | 175 tasks in a self-hosted simulated software company (GitLab, ownCloud, Plane, RocketChat), with simulated colleagues | Checkpoint-based partial credit plus execution checks | Long, multi-tool office work with communication; partial credit is explicit, and full completion is separately rewarded |
| [τ²-bench](https://arxiv.org/abs/2506.07982) | Dual-control conversations where agent and user both act on shared state | Programmatic state checks plus a tool-constrained user simulator; `pass^k` for reliability | Reliability, not just peak: a 90% pass@1 agent is ~57% at k=8 |
| [MLE-bench](https://arxiv.org/abs/2410.07095) / [RE-Bench](https://metr.org/blog/2024-11-22-evaluating-r-d-capabilities-of-llms/) / [PaperBench](https://arxiv.org/pdf/2504.01848) | Kaggle-style ML engineering (75 competitions); 7 research-engineering environments with 71 human expert attempts; replicating 20 ICML papers | Leaderboard thresholds; human-calibrated scoring; 8,316-item author-approved rubrics graded by an LLM judge | Human baselines make scores meaningful. PaperBench also shows the cost of rubric/LLM-judge grading, which we avoid |
| [Best practices for agentic benchmarks (ABC)](https://arxiv.org/pdf/2507.02825) | — | — | Two failure classes to guard against: task validity (can the task be solved the intended way?) and outcome validity (does passing mean success?). Their examples: insufficient tests in SWE-bench Verified, and τ-bench counting empty responses as success; errors "up to 100% in relative terms" |

Further reading we used for the shape of the field but did not adopt tasks from: GAIA
(short-answer assistant tasks with a private test split), WebArena and AppWorld
(self-hosted sites/apps with programmatic state checks, AppWorld also penalising
collateral damage), CyBench (professional CTFs verified by flag), BrowseComp
(hard-to-find facts with exact-match answers).

**What this implies for a "truly useful" agent.** It finishes real work in a real
environment over tens of steps; it recovers from its own errors rather than needing a
clean path; it reads long output, docs and error messages; it says what it actually did;
and it is reliable across attempts, not once in eight. None of that is measured by a
model judging its own transcript.

## 2. Anti-wireheading rules for this evaluation

These are the rules we held ourselves to. Each one closes a hole from the table above.

1. **Success is decided outside the agent.** Every task is scored by running hidden tests
   or by inspecting simulator/file-system state after the run. No LLM judges the work, and
   the agent never sees a checker. Hidden tests live outside the sandbox and are mounted
   read-only at verification time, after the agent's shell is dead.
2. **Checkers are pre-registered and validated before any agent runs.** For each task we
   recorded a digest of the instruction and checker, then demonstrated that the checker
   *passes* a reference solution and *fails* an empty workspace
   (`eval/results/longhorizon_validation.json`, 9/9 validated). This is the ABC paper's
   outcome-validity check, done up front.
3. **No leakage channels.** The sandbox has no network, so nothing can be fetched or
   phoned home. Task directories contain no answer keys, no oracle solutions and no tests.
   For the adapted Terminal-Bench tasks we replay only the environment setup, never the
   `solution/` directory. Answer keys for our own tasks live outside the sandbox.
4. **Held-out tasks stay untouched while improving.** Two tasks (`own/coding_ledger`,
   `own/data_cleaning`) are the tuning split; their transcripts were read while making
   improvements. The other seven, including all four Terminal-Bench tasks, are held out:
   we looked only at pass/fail and failure-class counts, never at their transcripts, until
   the improved arm was frozen.
5. **Tasks were authored before the system was tuned for them,** and the improvements are
   general mechanisms (a step budget, a plan in memory, notes, file writing, reading long
   output in slices), not task-specific code. No task name or string appears in the
   assistant.
6. **Realistic ambiguity.** The task briefs are the kind a person would write: they
   under-specify some things, require reading files and error messages, and (in the
   incident task) contain more than one fault with a misleading log.
7. **Budgets are explicit and equal.** Every arm gets the same wall-clock cap (45 min per
   task), the same model-call cap (150), the same sandbox and the same hidden checker.
   Cost is reported, not hidden: model calls, tokens, commands and wall time.
8. **Calibration baselines.** We report a plain ReAct loop on the same local model and
   sandbox with none of tensacode's structure, so an improvement has to beat "just prompt
   the model in a loop". Four of the nine tasks come from a real public benchmark
   (Terminal-Bench 2, Apache-2.0), so we are not only grading ourselves.
9. **Honesty is scored.** Where a task asks for reported numbers (CAD volume, EEG
   accuracies), the checker compares them against an independent measurement, so
   fabricated numbers fail even if the files exist.
10. **Failures are reported with their causes,** including cases where the assistant
    claimed success it did not have. No number in this document comes from anywhere but a
    run recorded in `eval/results/longhorizon.json`.

What these rules still do not give us: only nine tasks (so confidence intervals are wide),
a single local model as teacher, one attempt per task (no `pass^k` reliability measure),
and Terminal-Bench tasks reproduced without Docker (same instructions and tests, slightly
different environment: no root, no network, Python 3.12 from a shared toolbox).

## 3. The task set

Nine tasks, each expected to need tens of steps. `tb2/*` are adapted from Terminal-Bench 2
(Apache-2.0); `own/*` were written for this evaluation.

| Task | Source | Difficulty | Split | What has to be true at the end |
| --- | --- | --- | --- | --- |
| `own/cad_bracket` | own | hard | held out | A watertight STL of a parametric bracket: bounding box 120×80×30 mm, volume within 4% of the reference, material present at 5 probe points and absent at 11 (four bolt holes, the central bore, the counterbore, a chamfered corner, outside the part), plus a `report.json` whose volume matches the model it exported |
| `own/eeg_motor_imagery` | own | hard | held out | Real PhysioNet EEGBCI data for 3 subjects, filtered 7–30 Hz, epoched, CSP+LDA with 5-fold CV; reported accuracies must land within 0.15 of an independent reference run, epoch counts in range, mean above 0.60, a PSD figure and a report quoting its own numbers |
| `own/coding_ledger` | own | medium | tuning | A CLI expense ledger passing a 12-test hidden suite: exact-cent money, strict output formats, ids that never repeat, month/category filters, CSV export quoting, exit code 2 with messages on stderr for bad input, persistence as valid JSON |
| `own/data_cleaning` | own | medium | tuning | Three messy CSV exports merged and cleaned; 7 answers must match a hidden key exactly (row counts, dropped rows, duplicates removed, net revenue, revenue by region, best month, top category) plus a sorted `clean.csv` |
| `own/service_incident` | own | medium | held out | A broken orders API actually serving again: `/health` 200, `/orders` returning all 50 orders with the right keys, `/orders?status=open` filtered correctly, started the production way, plus a postmortem naming the config key and port. Two faults, one of them mentioned only in the log |
| `tb2/log-summary-date-ranges` | Terminal-Bench 2 | medium | held out | `summary.csv` with severity counts across five date ranges, checked by the task's own tests |
| `tb2/sqlite-db-truncate` | Terminal-Bench 2 | medium | held out | Rows recovered from a binary-truncated SQLite file into `recover.json` |
| `tb2/cancel-async-tasks` | Terminal-Bench 2 | hard | held out | An `asyncio` concurrency limiter whose tasks still run their cleanup on cancellation |
| `tb2/git-leak-recovery` | Terminal-Bench 2 | medium | held out | A secret recovered from rewritten git history, then scrubbed from the repo without disturbing anything else |

### The environment the agent works in

A real `bash` on a pty inside `bubblewrap`: no network, read-only system directories, the
task directory mounted at `/app`, a throwaway `/home/agent`, and a shared read-only Python
toolbox at `/opt/tools` (numpy, scipy, pandas, scikit-learn, mne, matplotlib, trimesh,
build123d, pytest). The assistant does not get an API to this shell: it sees a **browser
page** rendering the terminal (prompt lines, output blocks, a "Shell input" box) and types
into it, which is the same way it drives the simulated desktop. Consequences it has to
live with, like a person at a screen: only what is on screen can be read, long output
scrolls away, and each visible block is truncated.

### Arms

- **as-is** — the assistant exactly as it stood before this evaluation: chat grammar,
  hand-written skills, and the teacher-guided learning loop (12 steps, no plan, no file
  authoring).
- **improved** — the same code with long-horizon settings on: 80 steps, a plan of
  subgoals recorded in memory, notes carried across steps, `write_file`/`read_file`
  actions, guidance for reading output too long for one screen, a long multi-line brief
  treated as one piece of work instead of chat clauses, and stricter JSON extraction.
- **react** — the same teacher model in a plain ReAct loop over the same sandbox: task,
  notes, last few steps, tail of the last command's output; actions `run`, `write_file`,
  `note`, `done`. No perception through a screen, no claims, no plan.

### What code these runs used

Nothing is committed, so the runs are pinned by file digest, not by commit. All three arms
ran against this working tree (last commit `6387f54`, 343 files dirty):

| File | sha256 (first 12) |
| --- | --- |
| `examples/browser_agents/assistant/learning.py` | `d46db44cdb9a` |
| `examples/browser_agents/assistant/agent.py` | `678457c76f17` |
| `examples/browser_agents/assistant/programs.py` | `167c373f89a0` |
| `examples/browser_agents/assistant/language.py` | `eed79c32f00a` |
| `eval/longhorizon/tasks.py` | `ef08bc450259` |
| `eval/longhorizon/webterm.py` | `f2cc593fe32b` |
| `eval/longhorizon/react.py` | `9a815d674dc3` |
| `src/tensacode/actions.py` | `37bc4c86f8c1` |

Two things about that state matter for reading the numbers:

- **The safety machinery had been removed repo-wide before these runs started.**
  `tc.invoke`/`plan_order` no longer gate on an `Authorization` (the argument is accepted
  and ignored), and the assistant and its learning loop no longer ask before an action
  with an effect. So these runs show no confirmation round-trips, and the runner's
  question-answering policy was mostly idle. The `effect` flag is still computed and
  recorded in the trace; nothing acts on it.
- **The assistant's parser had a name-extraction fix** before these runs, and my own
  change that a long multi-line brief is treated as one piece of work rather than chat
  clauses is part of the improved arm only.

An earlier pass of the same runs (before I fixed the wait-granularity bug below) is not
reported: a single slow model call exhausted the mind's 2,000-cycle limit in about three
minutes, which ended runs prematurely and would have measured that bug rather than the
agents.

Results, failure classes and before/after follow in section 4.
