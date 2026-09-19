# 49 — Source-faithful language evaluation

*2026-09-19. This checkpoint corrects the evaluation boundary exposed by
[47 — Learned interpretation alternatives](47-learned-interpretation-alternatives.md).
It changes measurement, not the active language model or its interpretation policy.*

## The problem

The earlier confirmation joined UD word forms with spaces and required the entire reader
sentence to have identical tokens before awarding any attachment credit. Reconstruction
changed source typography: `I'm` became `I 'm`, and separate apostrophes could be interpreted
as quotation delimiters. A disagreement around one quoted phrase also made unrelated words
unscoreable. Those measurements could not distinguish annotation conventions from actual
interpretation failures.

The local test data already provides original `# text` and multiword-token surface rows.
A read-only corpus audit found exact surface alignment for all 2,077 sentences, including
354 multiword forms whose child forms concatenate exactly. Two empty nodes are excluded
from the basic dependency task. On the earlier 250-sentence sample, merely switching to
original typography changes strict whole-sentence alignment from 190 to 197 matches:
18 recover, 11 become mismatches, and 42 remain mismatches. That inexpensive tokenizer
comparison is a protocol diagnostic, not another model-quality experiment.

## What the new evaluator does

[`eval/parsing/span_evaluation.py`](../../eval/parsing/span_evaluation.py) reads exact original
text and builds character spans for basic annotated words. Multiword surface forms align
first; concatenative children then receive their own exact spans. Only intervening source
whitespace may be skipped. There is no lowercasing, apostrophe normalization, lemma
substitution, spelling repair, or fabricated alignment.

Missing source text, inconsistent surface forms, non-concatenative multiword forms, invalid
ranges, and unresolved alignment are explicit dataset errors. Their IDs and reasons survive
in the report. Alignment failures remain eligible for sampling even if their word count
cannot be trusted; they do not silently disappear through the length filter. Empty nodes
and enhanced dependencies are outside this basic dependency measurement.

The active reader receives original text. Before scoring, the evaluator validates that:

- Sentence and token spans lie inside the source and reproduce the reported text.
- Local token indices are exact integers, contiguous, and agree with the candidate's
  own token sequence. After [learned segmentation](51-learned-source-segmentation.md),
  this may differ from the sentence's first-proposal display tokens.
- Token anchors do not duplicate or overlap, and dependency keys cover every token.
- Heads refer to anchored tokens or ROOT, and trees have one root and no cycles.
- Predicted sentence groups do not overlap.

Malformed candidate metadata is reported and receives no candidate credit. Python boolean
aliases for integer indices or heads are rejected. Duplicate semantic proposals carrying
the same dependency tree cannot multiply syntax credit. The low-level `score` helper takes
trusted `Group` fixtures; `reader_groups` is the validation boundary for actual reader data.

## Metric semantics

An attachment is represented by its dependent character span, its head character span or
ROOT, and its relation label. Exact span matching makes local token indices irrelevant
across representations while preserving source identity. A wrong sentence split does not
make a false ROOT attachment correct.

The report separates:

- **Word alignment precision and recall:** exact word-span matches, including punctuation.
- **Full-denominator UAS/LAS:** correct attachments divided by all gold non-punctuation
  words. UAS checks the head; LAS also checks the label. Unaligned endpoints cannot receive
  credit, but unrelated aligned arcs remain scoreable.
- **Conditional attachment accuracy:** restrict the denominator to dependents and gold
  heads representable by the chosen prediction. This easier score always carries its
  denominator and must not be presented as full input coverage.
- **Whole-candidate oracle:** gold annotations choose one complete candidate per disjoint
  predicted sentence. UAS and LAS optimize separately. It never chooses one word's head
  from one candidate and another word's head from an incompatible candidate.
- **Exact tree recall:** require a single matching sentence organization and a complete
  matching labeled tree, including punctuation.

The `first` row follows retention order and does not represent an agent commitment. The
`oracle` row uses unavailable gold information. Oracle word-alignment and conditional-LAS
statistics describe its LAS-selected whole candidates; conditional UAS reports its own
chosen denominator. A zero denominator produces JSON `null`, never a perfect score.

Quotation grouping, hyphenated forms, and paths may differ legitimately from UD word
segmentation. This evaluator records that difference without assigning a semantic failure
label. It also cannot prove that every proposed frame faithfully expresses the syntax.
It evaluates available syntax, not grounding, user intent, action correctness, or cognition.

## Verification and bounded real-input smoke run

The focused tests use authored prediction fixtures to verify measurement correctness:
original contractions and repeated words, partial credit around merged quotations,
whole-candidate oracle limits, spurious sentence roots, inconsistent multiword records,
empty nodes, zero denominators, malformed metadata, duplicate anchors, and duplicate
semantic proposals. They are evaluator tests, not evidence of learned understanding.
All 16 focused tests pass. The implemented corpus loader independently aligns all 2,077
local test records with zero errors and explicitly counts the two excluded empty nodes.
The complete repository suite passed **1,778 tests, with 5 skipped**, in 178.51 seconds.

The small actual-reader run is predeclared as 12 randomly sampled test inputs, seed
`20260921`, 2–20 basic words, using existing cached weights and the frozen reader defaults:

```sh
.venv/bin/python -m eval.parsing.span_evaluation \
  --sample 12 --seed 20260921 --max-tokens 20 \
  --output eval/results/parsing_spans_smoke.json
```

No training, downloads, or paid inference occur. The report records source hashes for the
evaluator, tokenizer, treebank support, parser, semantic adapter, and active reader, as well
as model/data hashes and post-run equality checks. Reader exceptions, malformed candidates,
retention discards, no-act inputs, search truncation, and latency remain visible.

**Smoke status: completed on all 12 predeclared inputs.** Every evaluator/runtime source,
model, and dataset hash matched after the run. The report is
[`eval/results/parsing_spans_smoke.json`](../../eval/results/parsing_spans_smoke.json).

| Measurement | Result |
|---|---:|
| Gold basic words, including punctuation | 120 |
| Gold non-punctuation words | 103 |
| Exactly aligned words / predicted words | 115 / 123 |
| Word alignment recall / precision | 95.83% / 93.50% |
| First-proposal UAS / LAS, full denominator | 73.79% / 68.93% |
| Whole-candidate oracle UAS / LAS, full denominator | 81.55% / 77.67% |
| Oracle conditional LAS, with denominator | 80 / 98 = 81.63% |
| Oracle exact trees | 5 / 12 |
| Dataset alignment / reader / malformed-candidate errors | 0 / 0 / 0 |
| Inputs with no acts / retention discards / search truncation | 0 / 6 / 12 |
| Median reader latency / total timed reader work | 2.65 s / 27.20 s |

The full-denominator oracle credits 80 correct labeled arcs out of 103, while its
conditional score excludes five unrepresentable endpoints. Keeping both numbers prevents
segmentation differences from disappearing behind a conditional accuracy percentage.
Zero no-act inputs does not mean all emitted acts are understood correctly. The timing
comes from a shared development host running concurrent verification, not an isolated
performance benchmark.

A 12-input smoke cannot support generalization claims or meaningful numerical comparison
with the prior 250-input experiment. No expensive replacement benchmark is claimed in this
checkpoint. The measured advance is a source-faithful, validated evaluation path capable of
retaining partial attachment credit; the active model and its semantic limitations remain.
