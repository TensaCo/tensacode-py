# Symbolic language and induction

*2026-09-17. New: `src/tensorcode/language/`, `src/tensorcode/learning/`. Measured by
`eval/language_benchmark.py` → [`eval/results/language_benchmark.json`](../../eval/results/language_benchmark.json).
No model is called anywhere in either package, and neither has a dependency outside the standard library.*

**Open vocabulary (2026-09-17, second round).** The compositional 15/15 below was
measured **with the content words in the lexicon**, which hid a defect that mattered
more than the score: an unknown word could only be a *name*, so a clause with an
unknown verb had no verb. "Anem said the north field failed" lost its whole embedded
clause (coverage 0.33) and "the north field did not fail" was read as a polarity
question about a thing called "fail". Unknown words now take a category from their
morphology and position, enter marked and with an uncalibrated confidence, and the
clause parses. Open-vocabulary results are reported separately throughout, because
in-lexicon and open-vocabulary are different claims. §8 records the defects.

**Integration state (2026-09-17, after the coordinator wired it in).** The assistant
now tries the regexes first and the grammar only where they abstain. With the
coordinator's guard in place the grammar path fires almost only on pronoun cases the
regexes already handle, so **it adds roughly nothing to the assistant today**. Three
defects were found by that integration and are fixed below; the gains listed here
(reported speech, negation, modality) need the answer-form and coordination gaps
closed before a replacement is worth doing. §7 records what was wrong.

**Bottom line.** A real symbolic parser now exists: a feature/unification grammar, an
Earley chart, semantic frames that compose into `Claim`s, and generation from the
*same* grammar. It reads constructions the assistant's regexes cannot represent —
reported speech, negation, modality, quantifiers, comparatives, tense — at 15/15
against the regexes' 0/15, and 13 of those 15 survive being said again and re-read.
On the assistant's *own* 152-utterance benchmark it scores **61.8%** against the
regexes' **100%**, which is the expected result: those patterns were written for
those sentences. The honest summary is that the grammar buys structure and
bidirectionality, not slot accuracy on a tuned benchmark, and it is not yet a
drop-in replacement.

## 1. What came from the three repos

| Idea | Source | Here |
| --- | --- | --- |
| A fitted background unigram instead of a hand-set skip penalty, so a partial parse and a full parse are comparable | `symbolic-ai-models` `symbolic_ai_parsers/parsers/cky_001` | Re-implemented in `chart.cover`. The repo's note that a constant penalty is "a free parameter sitting underneath every classical row at once" is the reason |
| Argument slots carried apart from surface order | `symbolic_ai_parsers/grammar.py` | Re-implemented: a production's roles name daughter positions, so a dialect that moves the verb costs one production, not one predicate per word order |
| Unresolved reference kept as candidates rather than a silent pick (their XOR factor) | `symbolic_ai_core/reader/learned.py` | Re-implemented smaller: candidates live on the `Entity`, and `to_claims` returns `Unknown` |
| Productive shape vs memorised forms (`region_0` generalises, `group` does not) | same file, `Lexicon.induce` | Re-implemented as `learning.induce.role_type`, and as the open-class fallback in `grammar.OpenClass` / `guess_entries` |
| "A temporal envelope is only *learned* when it is constant" | same file | Adopted as a discipline, not code: an inducer that sees variation has learned nothing |
| Decision lists by separate-and-conquer with an MDL stop | `models/ruleinduce_001/dlist.py` | Ported, re-expressed over claims (`learning.induce.decision_list`) |
| The four controls: held-out, same-cardinality random, wrong question, consistent renaming | `symbolic_ai_lean/gate.py` | Ported as `learning.verify`; their phrasing "same-cardinality-but-meaningless … corrected four separate published results" is why |
| Read-set certificates, and "misses are reads" | `symbolic_ai_core/runtime/certificate.py` | Re-implemented as `learning.certificate`; `DecisionList.decide` returns one |
| Content-addressed, versioned artifacts with replayed fixtures and stale propagation | `typed-crystallization-networks` `tcn/library.py` | Re-implemented as `learning.library.Library` |
| Interventional precondition pruning (their precision 0.44 → 0.74) | `synthEX` `perception/induce_rules.py` | Re-implemented as `learning.induce.preconditions(..., intervene=)` |
| Least-general role type from observed fillers | `synthEX` `perception/induce_types.py` | Folded into `role_type` |
| Abstractions that fit a trace fail on held-out cases; group by behaviour, not syntax | TCN `research/FINDINGS.md` §44/§46/§65 | Adopted: `propose_concepts` groups candidates by which cases they cover, and `check_concept` refuses without held-out separation |

Nothing was imported. TCN and synthEX pull in torch/mujoco/pydantic, which would
break this package's zero-dependency rule; `symbolic-ai-models`' parsers are fitted
to its own curriculum rather than to English.

**A caution from the source.** `symbolic-ai-models` states plainly that its own best
exact graph recovery is **0.016** — "nothing in this repo actually turns text into a
graph" (`APPROACH.md`; `RESEARCH_LOG.md` line 1706). That is a different task
(inducing a grammar from scratch over an invented vocabulary), but it is the reason
this work uses a *written* grammar with induced extensions rather than trying to
induce the grammar itself.

## 2. What was built

```python
from tensorcode.language import ENGLISH, understand, realize, Context, resolve, to_claims

got = understand(ENGLISH_plus_words, "Anem said the north field failed")
got.meanings      # (Frame('say', {subject: Anem, content: Frame('fail', {...}, {tense: past})}),)
got.skipped       # words no constituent claimed — reported, never guessed
got.ambiguous     # two readings of equal score stay two readings
realize(grammar, got.meanings[0])        # 'Anem said the north field failed'
to_claims(frame, source=Ref("obs:x"))    # reported content lands in its own scope
```

* `features.py` — feature structures and unification (one mechanism; no side channels).
* `grammar.py` — categories, productions, a lexicon with category-aware morphology,
  and **semantic specs as data** (`Head`, `Build`, `Merge`, `Attach`, `Ent`, `Ask`,
  `Order`, …) which nest. Data rather than callbacks is what makes generation possible.
* `chart.py` — Earley over the feature grammar (no binarisation, left recursion
  allowed, constituents found at every position), then a Viterbi **cover** that reads
  the whole utterance as constituents plus background-priced skipped tokens.
* `semantics.py` — `Entity`, `Frame`, `Request`, `Question`, and the conservative
  conversion to claims.
* `discourse.py` — pronouns against a salience list; ties stay unresolved.
* `generate.py` — inverts the specs to say a meaning with the same grammar.
* `english.py` — the core English grammar (~60 productions, ~180 function words).
* `domains/desktop.py` — the assistant's vocabulary plus a thin projection onto its
  `(act, slots)` vocabulary, so the two parsers can be compared.

Four decisions carry the weight, and each exists to prevent a specific wrong belief:

1. **A question or an imperative converts to `Unknown`, not to a claim.**
2. **Anything qualified — negated, modal, comparative — is reified** (the event gets a
   `Ref` and the qualification becomes claims about it) rather than flattened into a
   triple that would assert the bare proposition. `to_claims` on "the grain did not
   arrive" emits no `arrive` triple at all.
3. **Reported speech gets its own scope, sourced to the speaker**, so "Anem said the
   field failed" never enters the shared world as "the field failed".
4. **An unresolved pronoun blocks conversion.** Two equally recent antecedents are
   kept as candidates; the caller is told, and can ask.

## 3. Measured

### Where these numbers come from

Worth knowing before reading them, because it changes what each one is worth:

| Set | Who wrote it | What that means |
| --- | --- | --- |
| Act benchmark, 152 utterances | The **assistant fork**, in `tests/test_assistant_language.py`, before this grammar existed | Independent. The hardest number here, and the one I cannot tune against without noticing |
| Compositional, 15 cases | **Me**, alongside the code | Written to exercise constructions I had just implemented, so it shares their assumptions — §8 is what that cost |
| Open vocabulary, 19 cases | **Me**, after the first integration report | Same caveat; it also had three of my own defects written into it as expectations (§10) |
| Induction tasks | **Me** | Controls are the check here, not the score |
| 17 sentence shapes, then 193 cases | The **civ-sim fork**, from a running world | Independent. It found five copula failures, a lost quantifier, a mis-stemmed `showed` and a tokenizer bug, none of which were on anyone's list (§11, §12) |

A set written beside the code measures the code's assumptions. Where a number below
rests on one, treat it as a floor on obvious breakage rather than evidence of coverage;
the two independent sets are the ones that have actually found things.

### The assistant's own benchmark (152 utterances)

| | correct | accuracy |
| --- | --- | --- |
| The 105 regexes in `assistant/language.py` | 152/152 | 100% |
| This grammar + desktop vocabulary | 94/152 | **61.8%** |

1.55 ms median per utterance, 8.89 ms p95 on a single pass (see §9 — it was
5.18/28.88 before the parser was profiled). Mean token coverage is 0.954, and 29 of the
152 utterances have two or more readings of equal score — genuine ambiguity, kept
rather than resolved by fiat. The regexes are the ceiling here by construction. The remaining 58 failures, by class: non-clausal answer forms the act
vocabulary treats as `choose` ("the second one", "2nd"); names coordinated inside a
`called …` phrase ("folders called drafts and final"); a few shell-shaped strings
(`rm -r old`, `sudo apt update`); and PP-attachment choices that land a place on the
wrong host. None of them are parser failures in the sense of "no reading" — they are
projection and ranking gaps, and the full list is in the JSON.

Several of those are deliberate. The projection **withholds** an act when the frame
carries a role the act cannot express ("remove the background from photo.png") or
when a changing act has no target ("move on"), because silently dropping part of a
request is how an agent does the wrong thing.

### Compositional constructions

Run twice: once with the content words in the lexicon, once against bare `ENGLISH`
where **every content word is unknown**.

| | in-lexicon (15 cases) | open vocabulary (19 cases) |
| --- | --- | --- |
| Structure correct | **15/15** | **19/19** |
| Whole utterance covered | 15/15 | 19/19 |
| Round-trips (said again, re-read, same meaning) | **15/15** | **18/19** |
| The regexes | 0/15 (all `unknown` or a wrong act) | — |

The open-vocabulary set is checked on what open vocabulary can actually recover, and
the two differences from the in-lexicon set are deliberate:

* the predicate of an unseen verb is a **stem the suffix table can inflect back into
  the word that was heard** — `arrive` for "arrived", `die` for "died", `carry` for
  "carried". It is still an internal identifier rather than a dictionary lemma, but it
  is now a verified one: see §10, where it used to be `arriv`, `di` and `carri`.
* an **irregular comparative** ("better" for "good") and an unseen adjective cannot be
  resolved to a lemma with no lexicon at all, so those two cases are checked on
  structure — that the comparison carries a `standard`, and that the report carries a
  nested clause.

Every guessed word is visible to the caller:

```python
got = understand(ENGLISH, "Anem said the north field failed")
got.guessed      # (('north','Name'), ('field','N'), ('failed','V'))
got.confidence   # Score(0.25, 'uncalibrated') — lower when more of it rested on guesses
```

Covered: past/perfect tense, negation, two modalities, quantifiers (`all`, `some`,
`no one`), comparatives with a standard, reported speech with and without `that`,
reported speech preserving the *inner* negation, an imperative, a yes/no question and
a wh-question.

**Generation round-trips: 15/15 in-lexicon, 18/19 open vocabulary.** Each reading is
said again with the same grammar and re-read. The one that does not is `all the fields
failed`: the grammar can read a quantified definite noun phrase but cannot build one,
so `realize` returns `None` rather than saying something narrower than the meaning.
That is the intended failure direction, but it is a gap.

### Discourse

`single antecedent resolved: True`, `tied antecedents left unresolved: True` with both
candidates recorded (`south field`, `north field`).

### Induction with its controls

A three-way routing task (120 train / 60 held-out) that no single condition can
capture:

| | value |
| --- | --- |
| held-out accuracy | 1.000 |
| train | 1.000 |
| simple floor (majority, or one-line rule) | 0.750 |
| same-cardinality random control | 0.250 |
| wrong-question control (labels re-paired) | 0.317 |
| identical under renaming | yes |
| **adopted** | **yes** |

Induced, and readable:

```
IF ¬urgent=True THEN 'normal'   [55/55]
IF paid=False   THEN 'queued'   [39/39]
ELSE 'fast'
```

And the control that matters — a task where the label *is* a symbol, so any rule can
only be reading vocabulary:

| | value |
| --- | --- |
| held-out accuracy | 1.000 |
| identical under renaming | **no** |
| **adopted** | **no** — "verdicts change under renaming: it is reading vocabulary, not structure" |

A 100%-accurate artifact is refused. That is the whole point of the gate.

Two of my own controls were wrong on the first attempt and the tests caught both: a
single label shuffle can come back as the identity permutation (now averaged over 20
draws), and renaming the *artifact* along with the cases hides exactly the failure the
rename control exists to find (now only the inputs are renamed).

## 4. Tests

47 new tests, 322 in the repo, all passing.

* `test_language_grammar.py` — unification, agreement, morphology in both directions
  from one table, clitic splitting, partial parses priced on one scale, ambiguity kept.
* `test_language_english.py` — every construction in §3, the four refusals, discourse
  ties, and parametrised round-trips.
* `test_language_desktop.py` — acts, chained requests without a clause splitter, and
  the property the assistant's suite also asserts: unclear input never yields a
  destructive act.
* `test_learning_induction.py` — MDL stop, all four controls (including a
  vocabulary-reading artifact being rejected and a majority-only rule hitting the
  floor), interventional pruning, productivity gate, concept proposal and checking.
* `test_learning_library.py` — versions, replayed fixtures, a tampered fixture
  refusing to load, stale propagation and revalidation, provenance kept across
  processes.

## 5. What does not work

* **61.8% on the act benchmark.** Not a replacement for the regexes today.
* **PP attachment is genuinely ambiguous** and the grammar keeps both readings; the
  projection has to look in both places, and sometimes picks wrong.
* **Open vocabulary cannot recover irregular forms.** "better" does not become
  `good`, and an unseen adjective after a copula may read as a nominal complement.
  Structure survives; the lemma does not.
* **Irregular forms of an unseen verb are regularised.** An unknown verb's past is
  formed by rule, so a world whose vocabulary includes an irregular verb the lexicon
  has never seen will hear it and say it back regularly. All 19 open-vocabulary cases
  now round-trip (§10, §11); this is the gap that remains underneath that.
* **No induction of the grammar itself.** The lexicon and productions are written;
  only rules, types and concepts are induced. Given `symbolic-ai-models`' 0.016, this
  is deliberate.
* **Generation is partial**: a production whose spec cannot be inverted is skipped,
  and `realize` returns `None` rather than an approximation.
* **Latency grows with sentence length**, because a constituent may begin anywhere so
  the cover can find fragments: 1.5 ms median, 7.7 ms p95, and 57 ms on a 28-token
  sentence (§9). Fine for chat and, now, for the sim without a cache.
* **Certificates do not pay for cheap answers.** A read-set certificate is worth
  `(cost of recomputing) / (cost of revalidating) × (fraction still valid)`. For an
  answer that is one pass over its own read set, those costs are equal and it buys
  nothing; it pays for model calls, fixpoints and long traces.

## 6. What adopters would need

**The assistant** (`examples/browser_agents/assistant/`): the parser can run *beside*
the regexes rather than replacing them — call `read_request`, and fall back to
`parse_message` when it returns nothing. That is strictly additive: it would add
reported speech, negation and modality to what the assistant can hear, and lose
nothing. A real replacement needs the `choose`/answer forms and the coordination
cases above.

**The civilization sim** (`research/civ_sim/`): `Grammar.extend` is the seam. One
grammar per settlement, sharing `ENGLISH`'s productions and diverging in the lexicon,
gives dialects; `Lexicon.without` drops a word; `Entry.weight` shifts a preference.
Speaking uses `realize`, hearing uses `understand`, and the lossy path is already
there — `Understanding.skipped` is exactly the material a rumour loses. For claim
transfer, `to_claims(..., source=speaker_ref)` puts what was heard in the speaker's
scope with provenance, which is what makes a rumour chain traceable.

**Both**: the pieces are pure functions over immutable grammars, so a caller can hold
several grammars at once without interference.

**Read-set certificates** (`learning.certificate`) are now available to both:

```python
from tensorcode.learning import Reader, revalidate
label, rule, certificate = rules.decide(facts)      # or Reader(facts) around any code
revalidate(certificate, facts_now)                  # Verdict: holds / fails, naming what moved
```

A key that was **absent** is recorded with digest `MISSING`, so an answer that rested
on "there is no such file" is invalidated when the file appears — not only when a value
changes. Revalidation is one digest comparison per key read.

## 7. Defects found by the integration, and what they were

Three, all reported by the coordinator after wiring `read_request` in. Two would have
made the agent act wrongly.

| Symptom | Real cause | Fix |
| --- | --- | --- |
| "make me a sandwich" → `create_folder(name='@it')` | The **unifier treated a missing feature as compatible**, so `V[ditrans=true]` matched *every* verb and the sentence parsed as a double-object verb whose object was "me". The projection then named the folder after that pronoun | A literal feature demand must now be *present* on the daughter (`chart._demands_met`), which is what subcategorisation means. Plus: a first/second-person pronoun can never fill a content slot |
| "could you possibly show me what the notes file says" → `list(place='@it')` | Same dative-pronoun confusion, plus the projection **inventing** a reference | `_needed` now withholds any act whose slot would carry `"@it"` unless the utterance really pointed at something — a third-person pronoun or a demonstrative |
| `inflect("fail", past)` → "faild" | Two suffix rules applied (`-ed` and `-d`) and the shorter surface was preferred | The candidate ordering now prefers a restored stem, then the longer suffix. Parametrised tests cover fail/share/carry/open/folder/box |

The unifier fix was the important one: it was silently distorting parses across the
grammar, and correcting it *raised* the benchmark from 57.9% to 61.8% while the new
withholding removed the wrong acts. "make a folder" (no name) and "move on" (no target)
now also abstain, because a generic noun does not name anything and an act that changes
something must say what.

The coordinator's own guard (reject a reading whose slots contain `"@it"` when the
utterance has no pronoun) is now redundant with `_needed`, but it is a correct
defence-in-depth check and worth keeping: it fails closed if this projection regresses.

## 8. Open-vocabulary defects, and what they were

Reported after the first round, from real transcripts. The benchmark did not catch any
of them because its compositional cases used in-lexicon words — the lesson being that a
benchmark which shares the system's assumptions measures the assumptions.

| Symptom | Cause | Fix |
| --- | --- | --- |
| "Anem said the north field failed" → the embedded clause gone, coverage 0.33 | An unknown word could only enter as a `Name`, and `Name` only feeds `NP`. With no verb there is no clause, so the cover skipped the lot | `OpenClass` + `guess_entries`: an unknown word may also be a noun, verb, adjective or adverb, with **morphology deciding** — `-ed`/`-ing`/`-s` mark a verb, `-s` a plural, `-er`/`-est` a comparative. Marked readings outrank bare ones, which is how the clause finds its verb among three unknown words |
| "the north field did not fail" → `Question(be, subject='fail')` | `QC -> NP` let a bare noun phrase be a whole yes/no clause, so "did not fail" parsed as a question about a thing called "fail" | A bare noun phrase is a question's subject only after a wh-word (`QSUBJ`), so the yes/no question needs a real predicate |
| "Anem said the north field failed" → one long noun phrase, then → a dative reading | A lowercase unknown word was as cheap a *name* as anything, and a reporting verb had no preference for a clause | Lowercase names cost more than a marked verb; `V[reports=true] S` is preferred over the dative `V NP S`; and a noun may modify a noun ("grain store") |
| "the grain has arrived" → `have(object='arrived')` | Nothing preferred an auxiliary supporting a verb over an auxiliary taking a noun | `Aux VP` is preferred, and the perfect has its own production demanding a past-form daughter |

Three of my own mechanisms were wrong along the way and the tests caught each: a
guard against invented auxiliaries also rejected ordinary `V NP` for tenseless frames
(it must only apply to features lifted from a *non-head* daughter); generation compared
candidates by score first, which rewarded piling on structure once productions had
positive weights (it now prefers the fewest words, since `_invert` already guarantees
every candidate is complete); and asking `inflect` to *add* "singular" failed because no
suffix expresses it (a bare guessed form now carries the unmarked features).

21 tests in `tests/test_language_open_vocabulary.py` cover the fallback, negation with
an unknown verb, sentential complements with unknown content words (with and without
"that"), noun-noun compounds, tense/aspect/modality, that skipping still happens for
genuinely unattachable material, and that a known word is never overridden by a guess.


## 9. Performance: the chart was keyed on `repr`

Reported after the second round, with a profile: `build_chart` cost 112 ms per
utterance and 50.0 s of a 52.6 s profiled run, and the sim only stayed usable because
it memoised `hear` — a 94% hit rate, which means its speed depended on villagers
repeating themselves. That is a fair criticism of the parser, not of the cache.

Four causes, none of them a constant factor:

| Cause | Cost | Fix |
| --- | --- | --- |
| Chart and agenda keys built from `repr()` of dataclasses | 520,054 `repr` calls on 171 parses | `Entity`/`Frame` carry a cached structural `key`; `Node` and `_Item` are `__slots__` classes that compute their key once |
| `isinstance(x, Mapping)` on the unification hot path | ~2.2 s of a 4.8 s run | `type(x) is dict` — every feature structure here is a dict |
| Every production seeded at every position | ~90 items per column, nearly all dead | `_startable`: only productions whose first symbol can actually begin there |
| Completion scanned the whole start column and let `_advance` reject the mismatches | `_advance` 332 ms of a profiled run | Items are indexed by the category they wait on |

Measured over 171 utterances (152 assistant + 15 in-lexicon + 19 open-vocabulary),
three repeats, ms per utterance:

| | p50 | p95 | mean | max | total |
| --- | --- | --- | --- | --- | --- |
| Before | 5.18 | 28.88 | 8.66 | 65.11 | 4.83 s |
| After | **1.65** | **7.71** | **2.66** | 27.84 | 1.48 s |

`repr` calls in the parse path: **520,054 → 0**. By grammar, after, over five repeats:
assistant 1.48 ms p50 / 7.32 p95; village in-lexicon 1.31 / 2.71; open-vocabulary 3.82 / 15.55
(an unknown word is several candidate readings, so open vocabulary is the expensive
case — and the sim's case).

**This is ~3.3×, not the order of magnitude that was asked for.** What is left is the
parser doing its own work: after the fixes the profile is `unify` 375 ms, `_unify_into`
279 ms, `Node.__init__` 258 ms, `_advance` 221 ms — spread across the algorithm rather
than sitting in one mistake. Getting another 3× would mean changing what the parser
computes, not how it stores it: interning feature structures so unification can hit a
memo, or capping ambiguity earlier than the per-span cap does. I did not do either,
because both change results and the point of this round was not to. The 112 ms figure
came from the sim's own profile, so the comparable number after this work has to come
from re-running that profile; what I can show is the 3.3× on my own 171 utterances.

### Identical parses, and one honest caveat

The target was "an order of magnitude without changing results", so results were
checked three ways:

* **Seeding filter**: parses with `_startable` versus seeding every production at every
  position — **171/171 identical**. The filter is result-preserving, not a heuristic.
* **Completion index**: parses before and after — **171/171 identical**.
* **Across runs**: two full runs of the snapshot — **identical**. This one mattered:
  `grammar.categories()` returned a *set*, so seed order — and therefore which of two
  equal-scoring readings won — varied between runs. The optimisation work is what
  exposed it. It is fixed by iterating productions in the grammar's own order, sorting
  categories, and breaking cover ties on chart serial numbers.

The caveat: **14 of 171 parses differ from the pre-optimisation snapshot.** 11 are
exact top-two score ties and 3 are beam truncation among ties, so they are attributable
to making the tie-break deterministic rather than to any optimisation — but a tie that
used to resolve one way now resolves the other, and the act benchmark moved 93 → 92 as
a result. It is back to 94 after the fixes in §10 and the nested-reference fix below.
Preferring the cover that filled *more* roles was tried as a tie-break instead of
serial order; it changed nothing on either benchmark, so the cheap rule stands.

### A nested reference the licensing check could not see

Found while auditing which benchmark cases had swapped: "add a folder called logs in
it" produced no act at all. The parse was right — `make(object=folder[name=logs,
location=it])` — but `_needed` refused to license `@it` because `Frame.entities()` did
not descend into an entity's own features, and PP attachment had put the pronoun
there. The utterance plainly said "it".

`Frame.entities()` now walks nested entities, which fixes the same blindness in two
other places: `resolve` never descended either, so the pronoun in "make a folder in it"
could not be resolved at all, and `to_claims` could not see an unresolved one to refuse
on. Act benchmark 92 → **94/152**.

## 10. Generation morphology

Reported after the second round: `realize` said "snow cames", "Miol ought gave food",
and stemmed "owes" to "ow" and "died" to "di" — *"the same class of bug as the earlier
'faild': your stem rules strip more than a suffix."* That is right, and there were two
distinct mistakes under it.

**A stem is now one the suffix table can inflect back.** The stemmer proposes
candidates — the plain strip, the strip with a final "e" restored, the letters a rule
says come back (`-ied` → `y`) — and keeps only those that, run *forward* through
`inflect`, reproduce the word that was heard. "ow" is rejected for "owes" because it
would have been said "ows"; "carr" because it would have been "carred". Stemming and
inflection are one verified pair rather than two rules that drift apart:

| heard | was | now | why |
| --- | --- | --- | --- |
| died | `di` | `die` | a vowel-final strip lost an "e" |
| owes | `ow` | `owe` | `-es` after a non-sibilant would have been said `-s` |
| carried | `carri` | `carry` | the `-ied` rule says a `y` comes back |
| shared | `shar` | `share` | one vowel group, single final consonant |
| arrived | `arriv` | `arrive` | no English stem ends in a bare "v" |
| failed | `fail` | `fail` | two vowel letters before the consonant: nothing restored |
| opened | `open` | `open` | two vowel groups: keeps its own final consonant |
| boxes | `boxe` | `box` | a sibilant stem takes `-es`, so it lost nothing |

**A form carries its own inflection.** "came" was listed as a bare alternate of "come"
with no features, so nothing contradicted the agreement rule and `-s` was appended.
Irregular pasts now carry `tense=past`, and `inflect` answers three ways rather than
two: `None` when contradicted ("came" cannot be made present), the word unchanged when
the language marks the demand with nothing ("came" takes no agreement, because only
present verbs agree; "share" is how a plural subject says it), and a suffixed form
otherwise. The distinction matters in both directions — returning the bare word for
*anything* unexpressible dropped the future off "the field will fail", so a feature no
suffix marks at all is a refusal, which sends the caller to the production that has the
auxiliary.

**A modal takes a bare complement.** `VP[tense=!]` is a new kind of feature demand — an
absence — declared in the grammar so the parser and the generator both honour it, and
"ought" is no longer a word on its own (`Modal -> "ought" "to"`). A modal frame that
carries a tense is said by the modal's own past form, which is what those forms are:
"could" is the past of "can". `should` and `would` have no present partner left in the
language, so each is listed twice, tenseless and past; without that a past modal frame
had no truthful way to be said at all.

Two further things fell out of testing that:

* a tenseless frame was being said with a past form, because both `give` and `gave`
  matched a need that demanded nothing and the tie went alphabetically. Saying "gave"
  for a tenseless meaning *adds* meaning, so a form carrying tense, aspect, polarity or
  modality the meaning lacks is now refused outright, not merely dispreferred.
* a feature demanded *of* a constituent was dropped at the mother, so the perfect said
  "the grain has arrive". A demanded feature is now routed to the daughter that
  expresses it, like any lifted one.
* `did` was listed as a main verb as well as an auxiliary, which read "did the north
  field fail" as an order. And contractions now lose generation ties to the written
  form, which is why it no longer says "the field 'll fail".

Results: in-lexicon round-trips **13/15 → 15/15**, open vocabulary **15/19 → 19/19**
(the last one with the plural fix in §11), act benchmark 94/152 unchanged, suite green. Timings cost ~10% at the median
(1.50 → 1.65 ms p50) for the extra lexicon entries and the round-trip check in
stemming.

**Three of my own open-vocabulary expectations had `arriv` written into them**, and one
of my tests asserted it twice. They failed the moment the stemmer was fixed. That is
the §8 lesson again from the other side: a benchmark written beside the system inherits
its defects as expectations, so a fix shows up first as a red test that looks like a
regression.

32 parametrised cases in `tests/test_language_grammar.py` and
`tests/test_language_english.py` pin each reported failure in the direction it went
wrong: the eight stems above, plural stems, "came" refusing a second suffix, unmarked
versus unmarkable, "snow came", "the elder died", "Miol owes Anem grain", the future
keeping its auxiliary, "ought to give" in both directions, the sim-built modal+past
frame, and that a tensed complement under a modal cannot even be read.

### Known morphology gaps

* **Consonant doubling** is not modelled: "stopped" stems to `stopp`, which round-trips
  to "stopped" correctly but would say "stopps" for the present. Doubling needs a rule
  shape the suffix table does not have (it consumes a letter rather than restoring one).
* **No past participles.** "must have given" is unsayable; the grammar has one past
  form per verb. This is why a past modal frame is said with the modal's past form
  rather than the perfect.
* **Irregular comparatives** ("better") still resolve to no lemma without a lexicon.
* **An unseen irregular verb is regularised** — the table would say "sayed" for a
  "say" it had never met. Listed irregulars are preferred over derived forms, so this
  only bites on vocabulary no lexicon has.

## 11. The copula, and a standing test instead of a lesson

The civ-sim fork ran its world through this grammar and reported 16 of 17 sentence
shapes realizing and reading back correctly, with five failures — all the same
construction, and all of it the most common thing that world says:

| frame | said | should be |
| --- | --- | --- |
| `be(Nise, hungry)` | "Nise am hungry" | "Nise is hungry" |
| `be(wood, dear)` | "wood am dear" | "wood is dear" |
| `be(Kasa, trustworthy)`, negated | "Kasa am never be trustworthy" | "Kasa is not trustworthy" |

Three causes, and none of my own sets contained a single copular sentence with a
third-person subject, which is why none of them caught it:

1. **Agreement never reached the verb.** `NP[number=?n] VP[number=?n]` is the grammar
   saying these agree, and parsing gets the value from the words. Generation has the
   opposite problem: the value can only come from the entity, and a name carries no
   features at all though it is third person singular for every purpose here. So the
   copula was picked alphabetically — "am" before "are" before "is". Agreement now
   travels along the grammar's own agreement variables, as a bundle (person *and*
   number, because that is what subject-verb agreement is), demanded only of the
   categories that can express it.
2. **`is` was not marked third person** in the lexicon, so once agreement *was*
   demanded it still could not be found. Its absence had been invisible because
   parsing never needed it.
3. **A negated copula had to borrow a clause negation**, which is where the second
   copula and the temporal quantifier came from. Copular negation is now its own
   production, and "never" is the second spelling of a negative polarity rather than
   the first, since it means more than that.

Fixing (1) exposed two more, both of which had been hiding behind the same missing
demand: a **guessed noun asserted `number=singular`**, so every plural was unsayable —
which also fixes `all the fields failed`, the gap §10 documented — and a bare verb form
was answering a demand for a past tense it did not express ("Nise be hungry").

`showed` stemmed to `showe`, which they rightly said the round-trip rule should have
caught. It could not: *both* `show` and `showe` inflect back to "showed", so the check
can only reject an inconsistent stem, never choose between two consistent ones. The
preference is still a guess, and it now knows that "w" and "y" after the vowel spell a
diphthong rather than closing a syllable.

### The lesson as a test, not a note

Two rounds in a row, a stale expectation outlived the bug it was written for. So
`tests/test_language_expectations.py` now reads every utterance in my own test files
and benchmark, asks what the current code makes of each word, and fails on any literal
that is that word with a suffix chopped off but is not a stem the code produces. It
reads the real tables, so it cannot drift from them, and a second test hands it the
`arriv` expectation we actually had, so the guard is known to fire.

It found a defect on its first run that neither benchmark exercised: `inflect` turned
"say" into "saies", because the `y → ie` rule fired after a vowel. Nothing round-tripped
to "says", so the stemmer refused the word outright.


## 12. Quantifiers, a tokenizer bug, and generation being the real cost

The civilization swapped its generator onto this grammar, which put *both* directions on
its hot path and turned up three things.

### The partitive, and one grammar read two ways

Their dialects use synonyms for *much* — `plenty`, `heaps` — and those are partitives:
they cannot stand in front of a noun without "of". `realize` returned None for them, so
that world printed "Coralin holds heaps bread" from its fallback. The gap was larger
than the report: **this lexicon had no amount words at all**. `much`, `little`, `many`,
`few` were not in it, which is why their generator passed the amount as a *name* — the
workaround my gap forced.

So the quantifiers are now in it, in the five shapes English allows, and with one
meaning per word rather than a canonical one per group: which word a dialect reaches for
is information its world uses, and a speaker asked for "heaps" must not be handed
"much". Normalising synonyms is the caller's business, and theirs already does it.

| shape | example | who can |
| --- | --- | --- |
| bare | "much food", "all files" | anything not a partitive |
| with a determiner | "all the files" | anything that is not an amount |
| partitive | "heaps of bread" | a partitive, which has no bare form |
| amount to a determiner | "much of the food" | an amount, which cannot reach one bare |
| repaired | "plenty food", "much of food" | read only — see below |

The last row is the interesting one. Their canonicaliser swaps a dialect's "heaps" for
the shared "much" and leaves the "of" behind, so the parser also has to read "much of
food"; and a hurried speaker drops the "of" the other way. Both are understood, and
marked `nonstandard=True` so a hearer can see the reading was repaired — **and neither
can be generated**, because a production may not state a feature the meaning does not
carry. That rule was already there to stop invented auxiliaries; here it buys a grammar
that is tolerant in and strict out, in one direction-neutral description rather than two
code paths that would drift.

Making that rule exact was a change of its own: a production's stated features used to
be satisfied by *absence* (`features.pop(key, value)`), so generation could quietly add
a negation or a perfect the meaning never had. Now only `mood` may be assumed, because
declarative is the unmarked mood; everything else must be in the meaning.

### The last word of every sentence was an unknown word

Chasing why a dialect case still failed turned up a tokenizer bug: the word pattern
allowed a trailing dot, so "food." was **one token**. No lexicon entry and no open-class
pattern matches it, so the final word of every sentence entered as a guessed name.

That was invisible here — a guessed name absorbs anything, and the benchmark's sentences
mostly end without punctuation — and expensive there, because that world rolls for
comprehension when it meets an unfamiliar word. It is the same shape as the finding they
reported from the other side: "Nise am hungry" and "Anem dies" were counted as unknown
words, so four of its commonest sentence types carried a 45% chance of being
misunderstood *because of our morphology*. Bad grammar was not cosmetic there; it was a
spurious misunderstanding mechanism, and a trailing full stop was another one.

A word may now contain a dot but not end with one ("hi.txt" survives, "food." does not).

### Generation cost 40× what parsing did

With both directions on their hot path, they measured 36.65 ms/mind/day un-memoized
against 3.00 memoized. Profiling from my side found the split nobody had looked at:

| | before | after |
| --- | --- | --- |
| saying one of their sentences (p50) | **91.15 ms** | **2.84 ms** |
| hearing one (p50) | 2.21 ms | 2.14 ms |
| my 171 utterances, parse (p50) | 1.59 ms | 1.57 ms |
| their 193-case suite, memos off | 19.73 s | **1.04 s** |

**Saying cost 40× hearing**, and all of the last round's parser work had been on the
cheaper half. Three causes, in order of size:

1. **The search re-explored everything.** One six-word sentence ran 24,103 searches over
   **74** distinct (category, need) pairs. Memoising them for the duration of one
   `realize` call is the whole fix — but only sound if an answer does not depend on where
   it was reached from, which took two steps. First the cycle guard had to go: a set of
   pairs already on the stack made every answer context-dependent, and it was redundant
   anyway, because the depth limit already stops left recursion. Then depth itself had
   to stop being part of the key, which it is not really — an answer that used three
   levels is the same answer at every depth of three or more, so each result now reports
   **how much depth it used** and one computation serves them all.
2. **`Entry.__hash__` built a string**, so putting entries in a cache key cost 148
   `repr` calls per sentence. The same mistake as §9's chart keys, in the same shape,
   found the same way. Now 0.
3. **`inflect` is a pure function** and was called 794 times per sentence; it is cached.

Two things I tried and dropped, because they changed nothing measurable: indexing
lexical entries by meaning as well as category (the scan was not the cost — the number
of calls was), and pre-filtering productions by the kind of meaning they can build (it
cost more than it saved). Both are noted here rather than kept.

**Identity, since a memo that changes answers is a bug:** 255 snapshots — my 171 parses
plus all 84 frames that world speaks, in four dialects, each said and heard back —
**255/255 identical** before and after, and identical across two runs. The act benchmark,
the compositional sets and all 721 tests are unchanged.

This is the order of magnitude §9 asked for and did not get, on the direction that
actually carried the cost: **32× on generation**. Their memo is now an optimisation
rather than a load-bearing part — un-memoized is faster than their memoized number was.
