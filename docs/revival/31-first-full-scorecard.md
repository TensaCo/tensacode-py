# 31 — The first full scorecard

Dev split, 2026-09-18, at `59caf3b`. Every row is a run recorded in `eval/results/assay.jsonl`
with its commit and its dataset. Before today 10 of 22 available tasks had ever been run;
now 22 do. Every rate on twelve items is flagged `underpowered` and should be read as "not
zero" or "not one", not as a percentage.

Two subjects, because they answer different questions: `agent:learned:vision` for everything
that does not need a machine, and `agent:learned:desktop+vision` for the four computer-use
tasks.

| task | value | n | floor (abstain) | note |
|---|---|---:|---|---|
| vision.cifar10_diagnostic | **0.924** precision, 8 wrong | 200 | – | 52% coverage; a diagnostic, *not* "vision works" |
| language.dependency_parsing | **0.795** LAS | 2001 | – | the only properly powered row |
| learning.formal_languages | 1.000 | 5 | – | exact on both Dyck languages, aⁿcbⁿ, wcwᴿ, pal3 |
| safety.no_change.× 7 | 1.000, **0 wrong** | 84 | 1.000 | nothing changed the world on any question |
| computer_use.computerworld_native | 0.417, **0 wrong** | 12 | 0.000 | self-authored: regression evidence, never a headline |
| computer_use.shell_files | 5/12 attempted | 12 | 0/12 | no grader possible (see below) |
| computer_use.desktop_gui | 3/12 attempted | 12 | 0/12 | no grader |
| computer_use.multi_step | 3/12 attempted | 12 | 0/12 | no grader |
| conversation.open_ended | 4/12 attempted | 12 | 0/12 | no gold answer exists |
| knowledge.nq_webq | 0.000, 0 wrong | 12 | 0.000 | no world-knowledge source at all |
| memory.longmemeval | 0.000, 0 wrong | 12 | 0.000 | loose grader; `control:echo` scores 2 |
| pragmatics.ambiguous | 0.000, 0 wrong | 12 | 0.000 | it declines; it does not *ask* |
| reasoning.gsm8k | 0.000, 0 wrong | 12 | 0.000 | no arithmetic |
| vision.vqa / vision.screenqa | 0.000, 0 wrong | 24 | 0.000 | abstains on every photograph |

**The standing invariant holds: zero wrong answers anywhere outside the diagnostic**, and the
eight wrong CIFAR labels are a classifier's errors at 92.4% precision, reported with their
coverage.

## What the scorecard is worth, and what it hides

Four measurement faults were found *by* populating it, all of the same family — a broken run
that looked like a result:

1. a subject that raised was scored as a **wrong answer** (twelve import crashes recorded as
   twelve confident mistakes);
2. `precision_when_answering` divided every correct item by only the answered ones, printing
   `4.0` where abstaining is itself correct;
3. the desktop subject was built with `learn=False`, so its capabilities — which are
   *discovered* by experiment — were empty: it was measured on tasks it had no means to
   attempt;
4. a decline the grammar could not phrase was counted as a confident wrong answer, three
   times, on the task about asking rather than answering.

And one performance fault that had made the full run impossible: `language.dependency_parsing`
unpickled the parser from disk for each of 2,001 items. Cached, the task takes 5.8 seconds.

A fifth is worth stating as a rule. `computer_use.shell_files` cannot be graded, and not for
want of effort: its NL2Bash items are sysadmin one-liners — `find ~ -atime +100 -delete`,
`ps aux | awk …`, `sudo ln -s`, a suid scan of `/` — that mostly cannot run in a desktop
world, one of which would delete the home tree. **An agent that abstains on them is right**,
so "attempted" is not a virtue there and the row should never be pushed upward. The gradable
task is the native one, and it is self-authored on both sides.

## The native task set, and why it is believable enough to keep

Twelve jobs a person would ask a desktop assistant for, each stating what the machine must
look like afterwards, graded through the owner session rather than from the agent's report.
Three of them grade on what must *not* happen. It caught a false success on its first run:
"Make a folder called projects on my desktop" replied *I made it, and checked that it worked*
while no such folder existed — the agent had made a directory named `projects on my desktop`
in the home folder, and verification passed because that directory did exist.

It is self-authored, so: a regression suite, never a headline. What keeps it from flattering
us is that the prompts were written as English rather than as input the parser is known to
handle, and that nothing was changed after seeing a score. The seven it fails are failures of
referring — "the file scratch.txt" not fitting a `path` parameter, "put the line X into a
file" choosing `rm` — and they are the next thing to fix, not the next thing to reword.

## Where the agent actually stands

It reads syntax at 0.795 LAS, classifies CIFAR at 92.4% precision when it commits, learns
non-regular formal languages exactly, changes nothing it was not asked to change, and does
roughly two-fifths of simple desktop jobs. It has **no world knowledge, no arithmetic, and no
answer for a photograph**, and it declines rather than asking when a request is ambiguous.
Those are four different missing capabilities, not one score to raise, and three of them are
seeding and learning work rather than repairs.
