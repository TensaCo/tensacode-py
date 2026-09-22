# Train-only near-correct review pack

These are 32 explicitly authored variants from 32 distinct questions in the
existing training partition. Every preserved original anchor had all three
reviewed targets true. There are eight candidates in each intended category:
supported facts answering the wrong restriction, predicate attachment, unsupported
category narrowing, and an unsupported qualifier inherited from an edited
question. Category names describe authoring intentions, not reviewed labels. No variant was
authored as a typo, metonymy puzzle, indiscriminate name swap, or vague plausibility
judgment; review should focus on the literal source distinctions.

Only the final category edits questions. Evidence objects and source identities
are unchanged in every case. Original anchor lines remain exact. None of the
prepared development, calibration, or reserved final data was read; no model
predictions were consulted and no model was loaded or trained. This pack does not
change existing labels or current training inputs. It is reserved for a separate
future experiment after the native-only control, subject to independent review.

## Frozen files

- `candidates.jsonl`: question, candidate, exact evidence, identifiers, explicit
  authored origin, and all-null target placeholders.
- `anchors.jsonl`: exact original positive training rows in variant order.
- `authorship.jsonl`: original IDs, question-edit flags, original text, categories,
  intended distinctions, and source provenance. These are metadata, not inputs.
- `review-template.jsonl`: all-null labels for separate reviewer output files.
- `authoring-spec.json`: frozen manual authoring specification. Each entry is
  `[positive_anchor_catalog_index, category, candidate, edited_question_or_null,
  intended_distinction]`. Build the catalog by retaining the first row with all
  three targets exactly true per distinct question in prepared training order.
- `manifest.json`: exact input and output hashes, ordered IDs, selection rules,
  category counts, and data-access limitations.

Model inputs, if used in a later experiment, are only the question, candidate,
and evidence source identifiers/text. Never feed category, intended distinction,
reference answers, old targets, or reviewer explanations to the model. These
files must not become implicit production knowledge or automatic semantic policy.

## Two independent reviews

Reviewers should form initial judgments from each candidate's literal question
and exact evidence before reading the author's intended distinction. Use the
existing response-quality definitions consistently:

- **Support:** whether the candidate's claims are licensed by the supplied
  evidence. An unsupported coherent added claim can fail even if the central
  answer is correct. A fact answering the wrong question can still be supported.
- **Completeness:** whether the candidate supplies the requested answer slots and
  types, independently of whether the supplied values are correct. A wrong value
  of the requested type may be complete; a supported restatement missing the
  requested information is incomplete.
- **Constraints:** whether the requested values and explicit restrictions are
  satisfied in light of the evidence. This is distinct from grammatical fluency
  and from merely mentioning a source-supported fact.

Preserve genuine ambiguity as null. Correct brief answers need not repeat every
qualifier. Do not require every axis to be false because the pack was designed to
probe failures. Do not silently repair entity/predicate substitutions or treat an
unsupported question presupposition as source evidence. Conversely, an author's
intended distinction is not sufficient reason to reject a candidate.

Each reviewer writes a separate new JSONL containing exactly one record per ID,
three bool/null targets, assistant authorship, and concise per-axis rationales.
Record concerns about original positive labels separately; do not rewrite frozen
anchors. Reviewers must not consult one another's labels before completing their
independent passes. Preserve both outputs, then adjudicate disagreements openly.
These are assistant judgments, not human gold labels. Frozen candidate/template targets remain null; reviewed supervision is stored
separately in the adjudicated files described below.

No tokenizer was loaded during preparation. Later experiments must measure their
full prompts and report any overflow rather than silently truncate or replace
examples. The categories are intentionally authored and balanced; performance on
this pack alone would not demonstrate natural failure detection or generalization.

## Adjudicated supervision

Two independent reviews are retained in `reviews-first.jsonl` and
`reviews-second.jsonl`. They agree on all three axes for 22 of 32 rows.
`adjudicated.jsonl` and `adjudication.json` preserve exact hashes and all ten
disagreements: four completeness cases concern explicitly answering a different
date field; six qualification/alias constraints remain unknown because the
evidence cannot settle them. A birth year does not fill an explicitly requested
death-year slot merely because both are years. Source support is independently
false for the six unresolved qualification cases.

Final label counts: support 8 true / 24 false; completeness 28 true / 4 false;
constraints 9 true / 17 false / 6 unknown. These labels supervise a proposed separate
training experiment; they are not performance results and do not alter any
existing training, calibration or development labels.

## Token coverage after review

A tokenizer-only check on GB10 used the same pinned foundation and three-axis
instructions as the current native experiments. 28 of 32 variants fit 512 tokens.
Variants 06, 12, 13 and 24 exceed the budget (maximum 530 tokens); their full inputs
and reviewed labels stay in the pack, and the training runner excludes them
explicitly. No sources were shortened or variants replaced. Exact token counts
and provenance are in `token-coverage.json`.
