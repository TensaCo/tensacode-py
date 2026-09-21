# Response-quality pilot supervision

`response-quality-candidates.jsonl` preserves all 88 natural proposals and exact
supporting passages from the previously inspected 32-question cognitive HotpotQA
run (`docs/results/cognition-hotpot.json`). These are development/adaptation data,
not a fresh benchmark. Reserved final questions have not been accessed.

The three label shards were authored by assistant reviewers on disjoint question
ranges (0–10, 11–21, 22–31), then audited across shards. They are not human ground
truth. Reference answers inform review but source-supported alternatives are
allowed. Source text, not existing NLI/QNLI scores, grounds judgments.

Each target is true, false, or null (masked):

- **support:** candidate factual assertions follow from supplied evidence.
- **completeness:** candidate supplies the requested answer slot/type; a wrong
  value of the right type can still be complete.
- **constraints:** candidate satisfies question restrictions and the requested
  values established by evidence; a wrong count/date fails this axis.

A supported statement can be incomplete or fail a question qualifier. Ambiguous
self-comparisons and incoherent person-to-property assertions have support and
constraints masked; missing answer slots still fail completeness. The cross-shard
review harmonized the repeated board-game comparison and person-as-birthday rows
with their analogous cases. No refusal examples occur in this small corpus.

Only question, candidate, and supplied evidence enter inference. References,
labels, rationales, reviewer identity and dataset provenance remain supervision
or metadata. The preparation script groups all variants and connected documents
before splitting and records exact hashes. Do not regenerate splits to improve
reported metrics or describe this narrow pilot as general response correctness.
