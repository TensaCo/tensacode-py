# 60 — Lexical goal interpretations

*2026-09-19. This follows [59](59-semantic-preservation-at-the-goal-boundary.md): preserving
qualifications is necessary, but an adapter must also retain competing goal interpretations
instead of silently choosing one by lexical frequency or resource order.*

## A selected reading does not uniquely determine a lexical goal

The language workspace may contain an explicitly selected frame whose verb has several
result-bearing lexical classes, constructions, or compatible preposition-role bindings.
Those alternatives describe different possible goals. A tagged-corpus class frequency or
the first matching resource frame cannot by itself authorize one of them for execution.

`verbnet.goal_candidates` now retains result-state proposals from all loaded result-bearing
resource frames and enumerates maximal injective assignments between supplied roles and
compatible thematic slots. It does not pick the largest matching at the expense of other
maximal matchings. Exact typed duplicate goals merge derivation provenance; this is not a
claim of semantic equivalence between merely similar conditions. Merged class labels list
their contributors rather than presenting the first class as the chosen interpretation.

Role correspondence and result-state extraction remain authored adapters over the supplied
VerbNet inventory. Slot mismatches and incomplete projection become obligations, not hard
evidence that a construction is impossible. The loaded representation still loses upstream
`SYNRESTRS` constraints; this checkpoint cannot recover that discarded construction evidence.

## A separate, explicit goal decision

The agent retains a goal-projection source and comparison group linked to the parent
interpretation dependency. `goal_selector` is a separate explicit policy and defaults to
none. Even a singleton candidate set defers without that policy. Selecting a language
reading does not automatically select its lexical goal.

Successful selection captures a second task dependency: one on the selected input reading
and one on the selected goal interpretation. Changes to either comparison invalidate the
old task commitment. Callback and evidence-copy guards check the precise comparison basis
before extracting and using a selected goal. The domain refinement that follows remains
authored; these dependencies record authorization, not a proof of semantic entailment.

Incomplete enumeration always blocks goal selection, even when an early proposal looks
usable. The selected proposal needs at least one derivation without outstanding structural
obligations. Unmapped input roles may remain explicit obligations on a selected proposal
so an authored domain refiner can represent them—for example, a supplied refinement can
encode a location. Without that explicit refinement, the existing goal-boundary guards
still prohibit execution that drops the role. Selection is not discharge of an obligation.
An unresolved search entry is not an executable goal, and qualification guards remain in force.

`goal_of` remains a convenience utility: it returns a goal only for a unique, completely
enumerated, obligation-free result; otherwise it returns `Unknown`. It does not provide
the agent's execution policy or restore a singleton default.

## Search and evidence limits

The derivation budget counts explored binding-search states, including partial assignments.
Exhaustion remains explicit. There is currently no resumable goal-enumeration cursor;
requesting more work requires another enumeration. Class or frame permutations can change
which partial results are reached under a small budget. Those incomplete results cannot
authorize execution. Complete-set invariance is measured separately below.

The audit uses installed VerbNet on explicitly supplied `make(object)` and
`move(object, destination)` frames with labeled reference values. It compares complete
semantic proposal sets after reversing class/frame order and injecting extreme mock implementations of the removed class-prior helpers,
retains all derivations, and fingerprints the actual resource files. These inputs are
authored frame fixtures, not a demonstration of learned parsing, grounding, or user intent.

## Installed-resource result

[The report](../../eval/results/goal_alternatives.json) is reproduced with:

```sh
.venv/bin/python -m eval.parsing.evaluate_goal_alternatives
```

The installed inventory contains 329 XML files. Their filename/content-hash manifest has
SHA256 `de2f2c264c20831cbfacc24ed6e2cbcc7b9e69f18670aafd9d75efb8b32c9710`.
The report stores every file fingerprint and verifies the resource manifest after the run;
all seven recorded source hashes also agree with the final runtime.

| Authored frame | Loaded classes | Distinct goal proposals | Retained derivations | Fully obligation-free proposals |
| --- | ---: | ---: | ---: | ---: |
| `make(object=Ref)` | 6 | 8 | 22 | 4 |
| `move(object=Ref, destination=Ref)` | 5 | 12 | 30 | 1 |

The last column excludes every obligation, including unmapped roles. It is stricter than
workspace selection eligibility: an explicitly selected partial goal may still require a
domain refiner to discharge unmapped-role obligations. Neither that column nor a unique
clean result supplies an execution policy.

With a 100,000-state enumeration bound, both audits complete. Reversing every class and
frame order preserves the exact typed goal set, including each aggregate class label; only derivation
provenance is excluded from that comparison. Production prior helpers
have been removed; the audit injects extreme mock helpers and verifies that neither is
consulted and the proposal set stays unchanged. This is an invariance check, not evidence
that arbitrary prior knowledge was learned or evaluated. Provenance frame indices naturally
change under permutation and are not compared as semantic content.

A one-state bound explicitly produces incomplete search in both cases, with unresolved
entries identifying 22 and 30 remaining resource frames respectively. These entries are
not a count of all possible remaining meanings. No incomplete candidate set is used to
select or execute a goal. The measured enumeration/permutation/mock-prior/truncation work
for the two fixtures took 4.08 and 6.68 milliseconds; those timings do not include learned
language interpretation, grounding, or action execution.

All twelve audit checks pass. The focused verification reported 15 lexical, 12 workspace,
and seven actual-Agent goal-boundary tests, plus a broader 39-test migration run in 35.25
seconds. These sets may overlap. The repository-wide run completed with 2,124
passing tests, five skips, and seven failures in older integration fixtures that
had not supplied lexical goal choices (316.23 seconds). Explicit named derivation
choices were added to those fixtures; all 31 tests across the four affected modules
then passed in 7.57 seconds. Their actual project, observation-driven, and scene-driven
filesystem assertions remain intact. No runtime code changed after the repository
run started; the other tests and those corrected modules cover the final runtime.
The resource audit itself does not execute an Agent or prove the policy tests' assertions.

The default remains **no goal selection policy**, even with one candidate. Goal-search
truncation is explicit and non-resumable, and outstanding structural restrictions do not
become supported merely because alternatives are retained. General understanding and
justified goal selection remain open; this checkpoint removes implicit authority from
lexical frequency and order while preserving the alternatives for explicit consideration.

[Deferred goal adoption](61-deferred-goal-adoption.md) addresses the next lifecycle step:
using a later explicit caller choice to revise the task already associated with a retained
goal comparison. It does not require reparsing or execute during adoption, and does not
claim to infer that choice from a natural-language clarification.
