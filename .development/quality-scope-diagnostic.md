# Frozen-weight whole-proposition diagnostic

## Decision and scope

The augmented native-foundation run retains 49 known-good development candidates
but admits five known failures. It rejects all 24 reviewed missing/swapped-source
controls. The remaining failures concern category narrowing, exact entity names,
predicate attachment and unsupported qualifiers inherited from the question.
This motivates testing whether the judgment instructions adequately specify the
whole proposition. It does not establish that foundation capacity is inadequate.

This protocol is fixed before running the new instructions. It is explicitly
development-derived prompt adaptation: the development failures have already
been inspected. Neither the cases nor the new instructions are an untouched
final test. No labels, training weights or numerical gates change.

Use the saved GB10 `foundation-only-augmented-xl/model` artifact, with foundation
digest `9a2a03d04c00c2cf41e861134d51795e23eb2eda59ed6dffff6021e834f4b85b`
and weights-file SHA256
`63040d019d9e4bc2d3d3d141e6f3c0085e1857b5135e1a78176185dc926a1561`.
Use its archived runtime, fp32 parameters with bf16 autocast, deterministic math
SDPA, and the same first-token conditional yes/no scores. Run only on GB10.

## Single comparison

Compare the exact original three instructions with this one replacement set.
Keep completeness unchanged. Use the following support instruction verbatim:

> Using only the supplied evidence, is every factual assertion in the proposed answer supported? Check the exact entities, relationships, categories, quantities and qualifiers. The question is a request, not factual evidence. Do not silently correct names, substitute related categories or transfer a fact from one entity to another. Answer yes only if every assertion is supported; otherwise answer no.

Use this constraints instruction verbatim:

> Using only the supplied evidence, does the proposed answer correctly satisfy every restriction in the question? Treat claims made by the question as unverified unless the evidence establishes them. Check the exact entities, relationships, categories, quantities and qualifiers. Answer yes only if every restriction is satisfied; otherwise answer no.

Evaluate all 92 calibration and 91 development candidates from the unchanged
augmented corpus and all 36 reviewed evidence-control records. Include both native
bypass and active workspace paths, with identical weights. Require original-prompt
receipts to reproduce the prior report exactly before interpreting differences.
Retain all rows in coverage denominators. Report every newly truncated input:
longer instructions must not silently displace evidence. Separately compare the
common eligible subset so overflow rejection cannot masquerade as improved
reasoning. Save complete receipts and exact artifact/data/script hashes.

## Interpretation and stopping rule

Apply the existing .5 threshold, per-axis retention/rejection criteria, and
combined zero-known-failure gate. Report known-good retention, known failures,
unresolved approvals, unchanged/new truncations and source interventions for each
condition. Do not choose a threshold or a second instruction after viewing scores.
Keep the original receipts and labels immutable, including typo/metonymy ambiguity
in two reviewed failures. No reserved final questions (304:336) are accessed.

A gain establishes sensitivity to an authored instruction change on development
cases, not newly learned cognition, calibrated truth or workspace usefulness.
A failure would not prove a capacity limit: these weights were trained under the
original instructions. Even a numerical pass requires independent source-grounded
review and complete-tool qualification before integration or publication.

## Verification

Test inference-schema isolation, exact original-prompt scoring parity with the
existing probe on a tiny foundation, overflow rejection, unchanged model weights
and fixed labels. Run the experiment helper suite before launch. Record results
beside the existing development reports and update the pinned execution notes.

Implementation: `.development/experiments/probe_quality_scope.py`, restricted to
the pinned augmented checkpoint and original prompt/precision settings. It checks
the 183 original candidate receipts and 36 original control receipts before
evaluating the replacement instructions. Both condition reports retain full
denominator gates; the final comparison also includes common eligible gates.
Independent review found and corrected a common-subset coverage counter error
before any real inference. Six focused tests and 74 experiment-helper tests pass;
wheel and sdist builds pass. No production scoring path is changed.


## Completed diagnostic: replacement rejected

The archived production runtime was 8e7c15a with diagnostic script 29390f4
(SHA256 `a5edc3ac704500a1547a1d2ab7640e9d950e5157ff9a26e7498c9d39aeddbf49`).
The runtime-source archive SHA256 is
`a8e01ca7dfdfa13a0a42e2fe297a88ada97d009189c53998439cef5985507fe5`.
The process exited successfully. Original instructions exactly reproduced all
183 candidate and 36 control receipts; before/after model-state digests match.
No labels, weights or thresholds changed. Full receipts remain on GB10; compact
provenance and both-path gates are in `docs/results/quality-scope-development.json`.

Native-bypass all-denominator calibration admission changes from 33 good / 5
known-failure / 1 unresolved to 31 / 8 / 3; exclusions rise from 8 to 14 of 92.
On the 78 commonly eligible rows, the comparison is 31 / 5 / 1 versus 31 / 8 / 3.
Development changes from 49 / 5 / 5 to 50 / 7 / 6, with three new overflows among
91 candidates. On the common 88, original is 49 / 3 / 4 versus replacement
50 / 7 / 6. Two formerly accepted known failures are among the new overflows;
that exclusion is not improved reasoning. All full-denominator and common-subset
combined gates fail. Exact new/unchanged overflow IDs and gate denominators are
preserved in the compact record.

All 36 source controls fit under both instruction sets. Both support-only and
joint three-axis thresholds retain 12 positive anchors and reject 24 negative
variants on both active and bypass paths. This preserves the prior source-use
result but does not offset worse admission on unchanged ordinary development
cases. The original instructions remain the incumbent unqualified experiment.

This is a negative result for one development-derived instruction adaptation,
not a training result, final evaluation, or proof of a foundation capacity limit.
The checkpoint was trained with the original instructions. No second prompt or
new run has been launched; no integration or promotion follows. Any next change
requires a separate fine-grained TRAIN-supervision or representation protocol,
with existing labels (including typo/metonymy caveats), thresholds and held-out
bytes preserved. Independent complete-tool qualification is still required.

The full calibration gate retains a denominator of 40 reviewed known-good
candidates (33/40 original, 31/40 replacement); the common eligible denominator
is 34 (31/34 for both). Eligible-only descriptive metrics must not replace these
full gates. All three new development overflows concern Darabont, removing two
accepted known failures and one unresolved case by coverage loss only. All three
original common-subset failures persist; four new failures concern wrong band
restrictions, reversed tennis ranking, acid versus a requested historical name,
and a racing team in place of the requested series. One reviewed-good comparison
answer and two unresolved candidates are also newly admitted. Active and bypass
paths make identical threshold decisions within each prompt condition.
