# Development evidence controls

This frozen review pack contains 12 unchanged, previously reviewed development
anchors and 24 authored evidence interventions. These questions and source texts
are disjoint from the prepared supervised training partition. They are known
development examples, not the reserved final evaluation or new natural failures.
No model was loaded, queried, or trained during preparation.

Selection takes the first 12 distinct question IDs with all three original targets
exactly true in prepared development order. There were 24 eligible questions.
For each anchor, the candidate and question remain unchanged while the evidence is
first removed, then replaced with the next selected question's evidence. The last
anchor uses the first anchor as its donor. Donor evidence objects, source IDs,
and text remain exact; a different question does not guarantee unsupportedness.

- `anchors.jsonl`: exact original prepared lines, including original review metadata.
- `candidates.jsonl`: 24 frozen intervention records, with no transferred targets.
- `review.jsonl`: review pack with all three targets initially null.
- `manifest.json`: selection order, original dataset hashes, provenance, donor map,
  frozen file hashes, and training-disjointness checks.

## Review instructions

Review each intervention's literal question, candidate, and supplied evidence.
Assess **support only**. The original positive labels were a selection rule and
must not be transferred to altered evidence. Do not consult model predictions,
external facts, or reference answers to fill gaps in supplied evidence.

Support is true only when the supplied evidence licenses the candidate as an
answer to the actual question. False means the supplied evidence fails to license
that answer; it does not assert that the answer is false in the world. Use null
when ambiguity prevents a defensible judgment. Empty evidence usually removes
support under this closed-world protocol, but inspect the literal question and
candidate rather than blindly assigning a label by intervention type. Similarly,
a swapped source can coincidentally support an answer and must be read.

Keep completeness and constraints null to isolate the support intervention.
If the original anchor appears mislabeled or ambiguous, record that concern
separately; do not alter frozen anchors or manufacture targets for other axes.

Write separate reviewed-label JSONL files instead of changing these frozen files.
Each label should contain `id`, `targets` (support bool/null; completeness and
constraints null), `review_authorship`, a concise `rationale`, and optional
`anchor_label_concern`. Identify assistant judgments as assistant-authored, not
human gold labels. Independent reviewers should retain their separate outputs so
disagreements can be resolved explicitly.

The manifest records zero overlap with training for question IDs and the union of
source IDs and SHA256 hashes of exact source text. Original development prompts
were reported to fit the existing 512-token protocol. Swapping evidence can change
length: later inference must check every transformed prompt without silent
truncation or replacement of the frozen selection. No length measurement is
claimed for the new variants because no tokenizer was loaded during preparation.

## Reviewed labels

`reviews-first.jsonl` and `reviews-second.jsonl` preserve two independent
assistant reviews. Both assign support=false to all 24 altered-evidence cases;
`adjudicated.jsonl` and `adjudication.json` preserve agreement and file hashes.
Completeness and constraints remain null. These are assistant judgments, not
human gold labels. The original positive anchors are not relabelled.

Reviewers flag two anchor caveats: Orient's population statement omits the
source's 2010-census qualification, and the malformed Thom Yorke question
already mentions Radiohead. The latter still needs supplied evidence for the
candidate's principal-songwriter claim under this protocol. Preserve both
caveats when reporting results; passing these controls alone cannot establish
general evidence-sensitive cognition. No predictions were used for adjudication.
