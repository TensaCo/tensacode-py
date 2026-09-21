# Foundation and proposal-prompt development review

Three assistant-reviewed runs over the same 32 historical questions. Each JSONL
reviews the first beam in direct answering, workspace-bypassed proposals and
workspace-active proposals. The raw exact evidence, all beams and model hashes
remain in GB10 artifacts/foundation-scale/{base,xl,xl-evidence-qa}.

`base` is the existing QA2D-adapted generator; `xl` imports FLAN-T5-XL at revision
7d6315df2c2fb742f0f5b556879d730926ca9001. `xl-evidence-qa` changes only the proposal
instruction to the version-2 evidence-QA instruction. No weights are fitted here.
References are review metadata; supplied passages determine support. The reviewers
allow clipped trailing prose when the requested answer remains grounded and
unambiguous, so correctness counts are not well-formedness counts.

Base direct correctness: 18/32. XL direct correctness: 29/32. XL proposal first-beam
correctness: 17/32 with the old instruction, 30/32 with version 2. One v2 proposal
answers a place instead of the requested distance; one question is ambiguous.
No XL first-beam output changed with workspace bypass. This is an untrained
workspace and demonstrates no learned workspace benefit. These are known
development cases, not final validation or complete-tool correctness.
