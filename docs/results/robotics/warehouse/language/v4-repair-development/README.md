# Bounded schema repair: retained negative result

The 24 v4 development first-pass outputs were hash-bound and replayed. Only the single schema-invalid output triggered one additional actual model generation. The retry received the original request, its own rejected candidate, and the strict decoder error; no expected tree or grammar-derived answer was supplied.

The retry failed: it replaced the invalid single-child sequence with duplicate instruction keys. Both raw attempts remain retained and rejected. Accuracy stays 23/24; one repair invocation produced no additional correct interpretation. Schema validity is not semantic admission, and repair never replaces valid-but-wrong candidates with an authored answer.

This run used the same pinned 4B weights on the peer GB10 with its recorded Python3.11/driver environment and verified 32/48-SM allocation. The host first passes used Python3.13. These are development reuse results, not a fresh challenge or a cross-host timing comparison. Source archive, original input binding, protocol, resource receipt, and raw outputs are preserved.
