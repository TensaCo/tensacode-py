# 47 — Learned interpretation alternatives

*2026-09-19. This checkpoint advances candidate generation from real text under the
[structured cognitive workspace objective](36-structured-cognitive-workspace.md).
It does not establish general semantic interpretation or scene understanding.*

## The problem and the active change

The trained reader previously committed to one part-of-speech sequence and one dependency
tree before the workspace could consider alternatives. Retaining a singleton interpretation
made that choice inspectable but could not recover the readings discarded inside perception.
A later evidence policy cannot choose an interpretation the reader never proposed.

The learned reader now uses bounded candidate search over the existing trained perceptron
weights. It retains competing tag sequences and complete dependency trees, then translates
each through the existing dependency-to-meaning adapter. These proposals enter the active
interpretation workspace. Reader ordering remains a search priority, not permission to
execute a request or assert a proposition. No first-reader execution fallback is restored.

The exact learned greedy tag path is retained alongside tag-beam proposals because its
lexical table contains training evidence skipped by the perceptron updates. Its dependency
path is retained only if every transition and the resulting tree validate without repair.
It remains a candidate, not a selected interpretation. An illegal historical transition
produces a diagnostic instead of invented root attachment.

Each proposal carries the source sentence and token character anchors, tag sequence,
lemmas, dependency heads and labels, transition sequence, separate uncalibrated tag/parser
scores, artifact identity, and search-budget metadata. Budget exhaustion and beam pruning
remain visible. A missing complete candidate produces unresolved input rather than a
fabricated attachment. Keeping syntax alongside the proposed meaning allows later
investigation to inspect what the adapter assumed.

The reader's default budgets are four tag hypotheses from a beam of four, four parse
hypotheses per tag sequence from a beam of eight, at most 100,000 expansions per search,
a shared 600,000-expansion sentence budget, and at most sixteen retained alternatives. It retains proposals across tag hypotheses
rather than summing incomparable tag and parser scores into a supposed probability.
These bounds control computation; they do not characterize the input's true ambiguity.

## Supplied knowledge and remaining semantic assumptions

The parser weights were learned from the externally supplied UD English EWT training
annotations. Search now exposes alternatives inferred from unfamiliar token sequences;
test fixtures do not supply their dependency trees. The current training method was
originally optimized for greedy decoding, so beam search is not assumed to improve its
first-ranked answer.

The dependency-to-meaning adapter remains authored software. Its role alignments, mood
rules, and lexical-resource use do not become learned semantics merely because their
input trees came from a learned parser. The former first-role/default-location preposition collapse is replaced by bounded
per-occurrence alternatives, with unknown roles retained explicitly. Corpus role priors
now use the STREUSLE training split only; they remain priors, not context-sensitive learned
meaning. Candidate syntax does not
prove that quantities, negation, scope, references, or preservation requirements reach a
grounded task correctly. The parser scores are uncalibrated margins, not probabilities of
user intent. The implementation does not automatically resolve grounded goals.

This checkpoint changes language candidate generation. It makes no improvement claim for
pixel-derived scene understanding. The current cached image classifier remains limited to
whole-image categories; the owner requires relational and holistic scene interpretation.

## Reproducible empirical evaluation

Run the cached-model evaluation without training, network access, or paid inference:

```sh
.venv/bin/python -m eval.parsing.evaluate_candidates \
  --sample 250 --seed 20260919 --min-tokens 2 --max-tokens 15 \
  --beam-width 8 --max-candidates 4 --max-expansions 10000 \
  --output eval/results/parsing_candidates.json
```

The evaluator samples without replacement from the official EWT test split. It records
artifact and dataset SHA-256 hashes, sentence identifiers, eligibility limits, seed,
search budgets, per-sentence scores, candidate counts, truncation, and latency. Both the
old greedy dependency decoder and the new candidate decoder receive the same greedily
predicted tags, isolating dependency search. It does not evaluate the additional tag
alternatives used by the active reader.

UAS is the fraction of non-punctuation tokens with the correct head; LAS also requires
the correct dependency label. Exact labeled tree recall is reported separately with and
without punctuation. For oracle scores, gold annotations choose the best candidate per
sentence independently for UAS and LAS. This is a candidate-set upper bound, **not an
implemented selection policy**. Empty candidate sets count as zero, rather than silently
falling back to the old decoder.

The historical repaired decoder now lives only in
[`eval/parsing/legacy_baseline.py`](../../eval/parsing/legacy_baseline.py); production
inference cannot call that fallback. Reproduction uses this evaluation-only copy.

The comparison also changes validity requirements: the candidate decoder requires complete
legal single-root trees, while the historical greedy path can attach unfinished material
to the root. Differences therefore cannot be attributed solely to beam width.

The sample includes 250 sentences of 2–15 tokens (including punctuation), selected from
1,347 eligible sentences among 2,077 test sentences (punctuation-only sentences are excluded).
It scores 1,492 non-punctuation tokens.
This short-sentence restriction biases results away from longer constructions. The sample
is now exposed to development; later confirmation should use a fresh predeclared sample.
No number here establishes open-ended cognition or the running acceptance episode in doc36.

## Initial measured result and correction

The first run of the command above produced:

| Decoder / selection | UAS | LAS | Exact tree, excluding punctuation | Exact tree, all tokens |
|---|---:|---:|---:|---:|
| Historical greedy baseline | 83.18% | 78.82% | 57.2% | 56.4% |
| First learned candidate | 49.93% | 47.86% | 42.0% | 40.8% |
| Gold-selected candidate oracle | 54.62% | 53.15% | 58.0% | 57.6% |

At these budgets, 46 of 250 sentences produced no complete candidate, including 45
expansion-budget exhaustions. Every search reported truncation. The mean candidate count
was 3.236. Median candidate decoding took 59.26 ms per sentence, compared with 1.31 ms for
the greedy baseline, excluding tag inference and model loading. Candidate decoding took
14.51 seconds total; the whole evaluation took 14.96 seconds after model loading.

The small increase in oracle exact-tree recall does **not** offset the large token-level
accuracy regression. This measurement establishes that the mechanism returns alternatives
from learned scores, but does not establish better language understanding or even better
syntactic parsing. The budget and complete-tree requirements need investigation before this
configuration can be described as an improvement in interpretation quality. A subsequent
measurement after inspecting these results must be labeled development-informed.

The artifact SHA-256 is
`987003c325014e486a23d861c7942a4d5caf12b769556037c09ecdbdd9eeabc5`.
The test-file SHA-256 is
`fa024f43dc5da3c5ac02563bc9bd0e974f46cbb1560823976a8f342a37dc494a`.
Full per-sentence measurements are in
[`eval/results/parsing_candidates_initial.json`](../../eval/results/parsing_candidates_initial.json).


After inspecting the initial result, two changes were evaluated on the same sample as
**development-informed diagnostics**. Local-margin ranking sums the chosen transition's
score minus the highest legal transition score at that state. This avoids comparing
absolute perceptron offsets across different states. Raw cumulative scores are retained
separately, and neither value is presented as a calibrated probability. Increasing the
expansion budget to 100,000 lets more partial searches reach complete trees.

| Configuration | First-candidate LAS | Candidate-oracle LAS | Exact-tree oracle, non-punctuation | Empty sets |
|---|---:|---:|---:|---:|
| Raw ranking, 10,000 expansions | 47.86% | 53.15% | 58.0% | 46 |
| Local margin, 10,000 expansions | 50.87% | 55.03% | 61.6% | 47 |
| Local margin, 100,000 expansions | 79.36% | 85.92% | 69.6% | 0 |

The final diagnostic retained an unrepaired greedy proposal on 229/250 sentences. Unioning
these with the beam candidates did not change oracle scores on this sample. The baseline
LAS was 78.82%; the final first-candidate UAS was 82.71%, slightly below the baseline's
83.18%. Thus this is evidence for more useful available syntactic alternatives, not a
uniformly improved selected parse. All searches were still truncated by beam/output limits.
The final fixed-tag evaluation took about 18.9 seconds; median beam latency was 63.6 ms.

These adaptive results are preserved in
[the margin diagnostic](../../eval/results/parsing_candidates_margin_diagnostic.json),
[the larger-budget diagnostic](../../eval/results/parsing_candidates_margin_wide_diagnostic.json),
and [the validated-greedy union diagnostic](../../eval/results/parsing_candidates.json).

## Fresh confirmation and the actual reader

**Status: the final frozen-source confirmation is running. No confirmation quality
results are claimed until its complete report is recorded below.**

The confirmation sample is predeclared as seed `20260920`, 250 sentences of 2–30 tokens,
excluding **all** 250 initial diagnostic sentence IDs. This includes longer constructions
while still excluding sentences above 30 tokens; it is not a full-test estimate. No model
is retrained. The final ranking and search bounds are held fixed for this measurement.

```sh
.venv/bin/python -m eval.parsing.evaluate_candidates \
  --active-reader --sample 250 --seed 20260920 --max-tokens 30 \
  --ranking local_margin --max-expansions 100000 \
  --exclude-results eval/results/parsing_candidates_initial.json \
  --output eval/results/parsing_candidates_confirmation.json
```

This run additionally exercises `LearnedReader.read`, including tokenization, learned tag
alternatives, learned parse alternatives, semantic projection, and the sixteen-proposal
retention limit. Inputs are the actual annotated UD word forms joined with spaces; this
preserves linguistic content but does not reproduce original typography. Reader sentence
or token segmentation mismatches count as uncovered and score zero. The fixed-token
syntactic decoder comparison is reported separately, so tokenization losses cannot hide
behind its scores. Retained candidate-oracle tag and attachment accuracy remain upper
bounds using gold annotations; neither measures which meaning the agent actually selects.


The confirmation was first attempted while the semantic adapter and learned reader were
being updated concurrently. It aborted with `ValueError: semantic reading unresolved or
ambiguous; use read_candidates`, before producing a results file. The failing sentence ID
was not retained by that attempt. This was an integration failure, not a model-quality
measurement; no quality-based resampling followed it. The evaluator now records per-sentence
reader exceptions as errors and uncovered input. Confirmation waits for the integrated
reader rather than silently calling the old semantic path.

Semantic alternatives may repeat a dependency tree. Active-reader syntax metrics deduplicate
by tags, heads, and labels, count semantic candidates separately, and report sentences with
retention discards. The final sixteen-alternative cap includes semantic alternatives, so
semantic branching can displace other syntax candidates. That is a measured representation
budget, not evidence that discarded readings are wrong. Current semantic bounds are four
alternatives and 64 expansions per tree, with a 2,048-expansion sentence total.

A subsequent confirmation process was stopped when source timestamps showed that the
semantic adapter changed eight seconds after process launch. That partial attempt produced
no quality report. The final run restarts the same predeclared sample with stable source
files; source hashes are recorded to distinguish these integration attempts.

An active integration example is `Birds fly.`: the learned reader retains a declarative
`fly(subject=Birds)` proposal alongside a noun-compound fragment. Its availability can be
asserted without assuming it ranks first. The grounding integration test explicitly supplies
a scene identity binding, then verifies source/model/tree provenance and that no selection
occurs automatically. This is learned proposal availability plus a supplied grounding
mechanism; it does not establish inferred identity or independent semantic selection.

## Implementation verification

The complete repository suite passed **1,705 tests, with 5 skipped**, in 173.66 seconds.
Focused checks cover legal trees, bounded searches, occurrence-specific semantic choices,
unresolved projections, source anchoring, detached grounding evidence, and absence of
automatic interpretation selection. Historical decoder migration preserved all trees and
aggregate metrics on a separate 250-sentence comparison. These checks establish the stated
mechanisms; the pending confirmation run supplies the separate candidate-quality evidence.
