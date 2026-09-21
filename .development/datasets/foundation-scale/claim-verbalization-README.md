# Claim verbalization development review

This diagnostic rewrites a question and proposed answer into a declarative claim,
then applies the existing NLI screening policy to the original and rewritten
candidates. Source evidence does not enter the verbalizer. The complete inputs,
outputs, prompts, and NLI receipts are in `claim-verbalization-records.jsonl`.
Assistant fidelity judgments are in `claim-verbalization-reviews.jsonl`, joined
one-to-one by `id` (records use `input.id`). The manifest records the exact script,
model fingerprint, input digest, artifact digests, and original review shard hashes.

All 123 pairs are known development material: 32 historical XL first-beam proposals
and 91 natural response-quality development candidates. This is not final validation.
The assistant reviews are authored judgments, not human-validated ground truth.
Fidelity means preserving the proposed answer and every question restriction;
correcting an answer is not faithful transformation. A faithful transformation can
retain a wrong answer. Missing restrictions and malformed output are separate
failure categories, even when the answer string remains present.

The review found 58 faithful, 40 incomplete, 13 malformed, seven changed-meaning,
and five ambiguous transformations. NLI accepted 63 rewritten candidates, including
25 nonfaithful transformations (20 natural candidates and five historical cases).
It accepted 53 original candidates. Increased acceptance does not establish improved
answer correctness. Literal answer retention was measured for 47 outputs and is
not a semantic fidelity test. None of the verbalizer inputs were truncated.

No checkpoint, production routing, or acceptance threshold is promoted. The compact
report is `docs/results/claim-verbalization-development.json`. Joint-evidence NLI
uses the existing source-pair calibration temperature; joint-domain calibration has
not been established. These results do not measure complete Chatbot usefulness.
