# 50 — Quotation boundaries and source evidence

2026-09-19. This checkpoint addresses a preprocessing failure beneath the
[structured cognitive workspace](36-structured-cognitive-workspace.md).

## The observed failure

The reader previously treated a sentence as wholly quoted when its first and last
characters looked like matching quotation marks. For `"delete" then "move"`, that
test removed the outer characters, parsed a different token sequence, and marked
every resulting act as a mention. Separate quotations had acquired an unsupported
whole-sentence interpretation before the interpretation workspace saw them.

Repeated trailing quotation marks could also disappear through `rstrip`, and an
empty quoted interior followed a truthiness fallback instead of being represented
as empty evidence. These are source handling and interpretation errors, not simply
dependency attachment errors.

## Required behavior

A whole-quotation interpretation requires an actual enclosing delimiter pair.
The first applicable closing delimiter must close that envelope; additional text
or unmatched delimiters must not be stripped to manufacture one. Interior content
and its original character offsets must survive. Multiple or malformed quotations
remain available in their original form. An empty quotation remains an observed
input with an unresolved interpretation rather than vanishing from the turn.

The use/mention convention is still authored. Identifying a structural envelope
does not establish the speaker's communicative intention, and marking an act as a
mention is not a learned conclusion. Interpretation selection remains explicit;
correcting quote handling does not authorize executing a candidate.

Both the grammar reader and the learned reader now carry quotation metadata with
the convention's name, whether it was applied, delimiter/content spans, and the
original message offsets. The grammar reader also retains the raw sentence text.
The actual cached learned reader is exercised by regressions for separate quoted
phrases, a trailing unmatched quote, empty content, and repeated quoted sentences.
These tests check retained evidence and preprocessing behavior; they do not label
the resulting semantic interpretations as correct merely because parsing returned.

Focused verification passed 36 quotation/reader tests and 13 existing general-agent
tests, including the regression that quoted language is mentioned rather than obeyed.
The full repository suite passed **1,778 tests, with 5 skipped**, in 178.51 seconds.

## Scope and next work

This correction does not solve general tokenization or sentence segmentation.
The existing tokenizer still supplies authored lexical conventions, treats quoted
phrases as single tokens in some contexts, and has limitations around apostrophes
and ellipses. Learned segmentation with preserved alternatives remains a separate
capability to implement and measure. Adding more special-case phrase repairs would
not meet that objective.

Character-span evaluation must use original text, retain annotation alignment
failures separately, and score unaffected relations when another span is unmatched.
The [companion evaluation checkpoint](49-source-faithful-language-evaluation.md)
documents that protocol. Neither a quote
regression test nor a gold-selected syntax oracle demonstrates general understanding.
