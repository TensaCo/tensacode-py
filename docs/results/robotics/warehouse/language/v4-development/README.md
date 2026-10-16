# Recursive prompt schema correction (development reuse)

The v2/v3 prompt described composition children as MOVE-only arrays, while the implementation supports recursive instruction nodes. V4 corrects that contract, adds two nested-tree examples, and requires preservation of every color/relation/anchor modifier.

On the 24 reused requests from the v3 fresh challenge, v4 produced 23/24 exact interpretations, compared with the earlier 20/24. Both dropped-color failures were corrected and one nested failure was corrected. The remaining n4 output has a single-child sequence nested under a parallel node; strict decoding rejects its invalid arity. All raw outputs remain available.

The independently authored grammar now reports version 2 and abstains on unpunctuated mixed concurrency/sequence scopes instead of imposing a precedence. It certified 11/24 candidates here, all semantically correct in this sample. Earlier source snapshots and certificate counts remain unchanged.

This run used the reusable cache-only PinnedTextLanguageModel adapter with the same pinned 4B weights and bounded inference settings. Prompt improvement was evaluated on development reuse; these numbers are not heldout generalization evidence. A separate bounded model-only schema-repair experiment retains the original invalid proposal and gives the model its schema error, never the expected answer.
