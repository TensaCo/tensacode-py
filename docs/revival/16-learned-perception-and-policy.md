# 16. A learned tier for perception and action

Before this, the live path held exactly one trained component — a TF-IDF and
logistic-regression intent classifier (`examples/support_router/config.py`) — plus the OCR
and detector models in `examples/browser_agents/vision/`. Everything else deciding what a
label means, what a rectangle is, or which intention to take was hand-written. This chapter
trains four components from supervision we already had, and reports what each one beat and
what it did not.

Two of the four are worth keeping. One is a clear win on the weakest number in the
perception report. One fails, and the arithmetic says why it was never going to work at
this data scale. Every artifact lives in the scratchpad, not the repo; the training scripts
are in `eval/training/` with fixed seeds and a recorded manifest.

| Component | Replaces | Verdict |
| --- | --- | --- |
| Label encoder (§16.1) | trigram string similarity | **mixed**: better on held-out apps, worse across operating systems, 9× faster, a tie in the live path |
| Role model (§16.2) | hand-written role rules | **win**: textbox recall 0.15-0.17 → 0.82-0.83, and it holds across operating systems |
| Intention ranker (§16.3) | the hand-written priority objective | **fails**: 0.79 per decision, which is ~1% of episodes |
| Claim embeddings (§16.4) | exact links for associative recall | **fails**: loses to token overlap on both measures |

## 16.0 Where the supervision came from

Nothing here was hand-annotated.

* **The apps' own label tables.** `web/access.html` renders one of four label variants per
  form field. The agent's hand-written synonym lists (`TEXT_CONCEPTS`) contain three of
  those four. The fourth — `Legal name`, `Badge ID`, `Effective from`, `Supervisor e-mail`
  — is therefore a held-out set that exists because the app's author wrote it, not because
  an evaluator chose it. A guard in the trainer refuses to run if one leaks into training,
  and it fired on the first attempt: the captured frames include access pages.
* **Pixels paired with the DOM at the same instant** (72 frames, `tune` / `test_app` /
  `test_os` splits from `eval/vision_capture.py`). The DOM says the role and the name; the
  recognizer says what the pixels look like. Both labels are mechanically true.
* **Our own graded episodes.** `eval/training/rollout_decisions.py` replays episodes with a
  recording `choose` implementation that writes down *the options that were rejected*, not
  only the one taken, and stamps each decision with whether the app scored that episode
  fully correct. 2,383 multi-option decisions over 221 episodes, all of which the
  hand-written chooser got fully right.
* **Live recordings** (`live_run6.jsonl`): claims that arrived in the same cycle, as
  free co-occurrence data.

## 16.1 A learned label encoder

`examples/browser_agents/learned/text_embed.py`. Hashed character n-grams (3-5) plus whole
words into an embedding bag, mean-pooled, one tanh layer, L2-normalised. 528,448
parameters, trained by InfoNCE over in-batch negatives on 2,046 positive pairs in **1.6 s**
on CPU. No pretrained weights: none were cached, and downloading a sentence encoder would
have turned a comparison of ideas into a comparison of pretraining corpora.

**Form concepts.** Scoring a field label against the four concepts:

| | n | top-1 | mean score of the right concept | clears the 0.3 claim threshold |
| --- | --- | --- | --- | --- |
| rules, phrasings in the synonym lists | 12 | 1.000 | 1.000 | 1.000 |
| learned, same phrasings | 12 | 1.000 | 1.000 | 1.000 |
| **rules, held-out fourth variant** | 4 | **0.750** | **0.215** | **0.250** |
| **learned, held-out fourth variant** | 4 | **0.750** | **0.469** | **0.750** |

Same top-1, different failures, and that is the interesting part:

```
rules    Effective from     -> employee_id  WRONG  score=0.000  margin=0.004
learned  Effective from     -> start_date   OK     score=0.375  margin=0.173
rules    Supervisor e-mail  -> manager      OK     score=0.250  margin=0.195
learned  Supervisor e-mail  -> name         WRONG  score=0.200  margin=0.372
```

The rules get `Effective from` wrong by a margin of 0.004 — a coin flip dressed as a
score. The learned encoder gets it right and gets `Supervisor e-mail` wrong instead,
probably because nothing taught it that a hyphen inside a word is noise (`e-mail` ↔
`email`). I did not add that augmentation: it would have been chosen *after* seeing the
held-out failure, which is how a held-out set stops being held out.

The threshold column matters more than top-1. The rule needs a score of 0.3 before it will
even record `may_mean`, and the field-plan search needs a claim for every concept, so three
of the four held-out variants produce no claim at all from the rules.

**Targeting through a recognizer.** Query with the DOM's name for a control, rank the
recognizer's readings of that frame, and count a hit when the top-ranked reading sits
inside the control's own box. At `find`'s own thresholds (accept above 0.55, refuse a
margin under 0.08) the two scorers are not comparable — the thresholds were tuned for
trigram scores — so the honest comparison is accuracy at matched coverage:

| split | arm | cov 0.3 | cov 0.5 | cov 0.8 | cov 1.0 |
| --- | --- | --- | --- | --- | --- |
| test_app (held-out apps) | rules | 0.690 | 0.660 | 0.427 | 0.345 |
| test_app | **learned** | **0.727** | **0.696** | **0.483** | **0.412** |
| test_os (macOS, Windows) | rules | 0.653 | **0.547** | **0.377** | **0.304** |
| test_os | learned | **0.664** | 0.511 | 0.344 | 0.281 |

So it generalises to unseen *apps* and does not generalise to unseen *operating systems* —
which is what you would expect from an encoder whose OCR pairs all came from Ubuntu frames.
At fixed thresholds it also trades refusals for wrong clicks (test_app: hit 0.364 / wrong
0.245 / refused 0.391, against the rules' 0.287 / 0.079 / 0.635), so adopting it would
require recalibrating `find`'s thresholds, not just swapping the scorer.

It is **9× faster**: 0.020 ms p50 against 0.189 ms, because candidates are encoded once per
frame instead of compared pairwise.

**In the live path it is a tie.** Swapping the encoder in for `label_similarity` inside the
access agent, 30 seeds, 17 of which show a variant the synonym lists do not contain:

| arm | items | episodes fully correct | escalated | duplicates |
| --- | --- | --- | --- | --- |
| rules | 120/120 | 30/30 | 0 | 0 |
| learned | 120/120 | 30/30 | 0 | 0 |

Identical, including on the 17 seeds with an unseen label. The diagnosis is worth more than
the tie: the rules survive those labels not because matching works but because
`_type_evidence` adds 0.6 for a date hint, an email input or an `E12345` placeholder, and
the remaining concept falls out of the permutation search. **Label matching was not the
binding constraint in that task**, so improving it changes nothing there. The place it
would pay is the pixel path, where the numbers above say it helps on new apps and hurts on
new platforms.

## 16.2 A learned role model

`eval/training/train_role_model.py`. Gradient-boosted trees over 16 features a pixel
provider actually has — the box's geometry and position, the recognizer's words inside it
(count, confidence, digit and uppercase fractions, a colon), the detector's best overlap,
and how many elements share its row and column. No DOM feature is an input; the DOM is only
the label. 911 training rows from the `tune` split, **7.9 s** to fit, 0.014-0.028 ms per
element to predict.

The baseline is the weakest number in the vision report: the hand-written rules get the
role right for **0.15-0.17** of matched textboxes on held-out frames.

| split | accuracy | majority baseline | macro recall | textbox recall | textbox precision |
| --- | --- | --- | --- | --- | --- |
| tune | 1.000 | 0.869 | 1.000 | 1.000 | 1.000 |
| test_app | 0.916 | 0.910 | 0.549 | **0.824** | 0.867 |
| test_os | 0.874 | **0.961** | 0.853 | **0.831** | 0.948 |

Textbox recall of 0.82-0.83 against 0.15-0.17 is roughly a fivefold improvement on the
number that mattered, and unlike the label encoder it **holds across operating systems**.

Two honest costs. On `test_os` the overall accuracy (0.874) is *below* the majority
baseline (0.961): the model finds textboxes by being willing to mislabel some buttons
(button recall 0.876, precision 0.995). Whether that trade is worth it depends on the task
— for an agent that must type into fields, it is. And the rare classes it never saw enough
of are simply absent: `tab` (0 in training, 20 in test_app) and `combobox` (4 in training)
both score 0.0 recall. A role model trained on 911 rows from one platform cannot invent a
class it has not met.

**Not integrated.** The vision perceiver is owned by another line of work, so this is
measured offline. The hook it needs is one call: replace the role decision in
`vision/perceive.py` with `model.predict(features(box))` and keep the rules as the fallback
for classes the model abstains on.

## 16.3 An intention ranker, and why behaviour cloning fails here

`eval/training/train_intention_ranker.py` and `examples/browser_agents/learned/chooser.py`.
A linear scorer over 84 features (the intention's kind, what its `why` says through the
label encoder, and the shape of the option set) trained with a softmax-over-options loss,
weighted by whether the episode it came from ended fully correct. 2,383 decisions, **0.26 s**
to fit.

Two things were withheld on purpose. `priority` **is** the hand-written objective, so a
model given it learns to copy one number. And the candidate lists arrive in the order the
task code generated them, which leaked the answer outright:

> With the position feature included, held-out top-1 was **0.7992** — and "always take the
> first option" was also **0.7992**, to four decimal places. The model had learned nothing
> except list order.

With the options shuffled and the index dropped, it does learn something real:

| | held-out top-1 |
| --- | --- |
| learned ranker | **0.7922** |
| always first option | 0.3236 |
| random weights | 0.2971 |

Per task: recon 1.000, chart 0.900, access 0.746, shop 0.375 — and shop's 0.375 is exactly
its random control, so on that task it learned nothing at all.

**Then it collapses in closed loop.** Same agents, same seeds, same perception; only the
`choose` implementation differs, on 20 held-out seeds per task:

| task | arm | items | accuracy | episodes fully correct | escalated | mean actions |
| --- | --- | --- | --- | --- | --- | --- |
| access | hand-written objective | 80/80 | 1.0000 | 20/20 | 0 | 45.5 |
| access | learned ranker | 4/80 | **0.0500** | 0/20 | 20 | 0.4 |
| shop | hand-written objective | 20/20 | 1.0000 | 20/20 | 0 | 5.8 |
| shop | learned ranker | 10/20 | 0.5000 | 10/20 | 0 | 4.8 |
| recon | hand-written objective | 55/55 | 1.0000 | 20/20 | 0 | 23.8 |
| recon | learned ranker | 0/55 | **0.0000** | 0/20 | 20 | 4.1 |
| chart | hand-written objective | 40/40 | 1.0000 | 20/20 | 0 | 5.0 |
| chart | learned ranker | 0/40 | **0.0000** | 0/20 | 20 | 0.0 |

The arithmetic explains it completely. An access episode is 20-45 sequential decisions, and
a single wrong one can end the episode — `0.4` mean actions means the ranker picked a
terminal intention (escalate or finish) on the first cycle. At 0.79 per decision,
0.79²⁰ ≈ 0.009, so about 1% of episodes should survive, which is what we see. To match the
hand-written objective's 98.06% item accuracy over 102,899 episodes, per-decision accuracy
would need to be around 0.999.

**What this does and does not say.** It does not say a learned policy cannot work. It says
behaviour cloning from 2,383 decisions cannot replace an objective whose per-decision
accuracy is effectively 1.0, because errors compound multiplicatively over an episode. The
routes that could work are in-the-loop training (DAgger, or reinforcement learning against
the app's own score, which is available as a reward), or using the ranker only where the
hand-written chooser abstains — a tie-break, which cannot make a confident decision worse.
Neither is measured here.

## 16.4 Claim embeddings, and a near-miss that would have been a false positive

`eval/training/train_claim_embeddings.py`. The awareness core wants to spread activation
from a cue to related claims, and hand-built links (shared subject, shared object,
provenance) are exact but narrow. The recordings offer the association for free: claims
believed in the same cycle were relevant to each other at that moment. 866-token vocabulary,
41,568 parameters, over 4,000 cycles with the last 20% held out.

The first attempt reported that learned similarity beat token overlap on associative recall,
0.201 against 0.106. **It was an artifact.** The sampled objective had diverged and 847 of
866 embedding rows were NaN, so the ranking was arbitrary sort order. I found it by checking
the loss (`nan`) rather than the metric, and the trainer now refuses to save a non-finite
checkpoint. Had the loss not been printed, this would have gone into a report as a win.

With that fixed, and with a closed-form alternative (PPMI over the co-occurrence matrix,
factorised by SVD, which needs no learning rate):

| method | same-cycle p@5 | same-subject p@5 |
| --- | --- | --- |
| token overlap (baseline) | **0.106** | **0.944** |
| sampled objective, norm-clipped | 0.005 | 0.645 |
| PPMI + SVD | 0.005 | 0.741 |

Both lose, on both measures, on a held-out pool of 4,362 claims. The diagnosis is that
these claim strings are short and their tokens are highly diagnostic — a subject id appears
verbatim in every claim about that subject — so exact token overlap is an extremely strong
baseline, and 866 tokens of co-occurrence over 3,500 cycles is not enough signal to beat it.

**Recommendation to the awareness core:** use exact links plus token overlap for recall
now. Embeddings over these strings are not worth wiring in. If associative recall is wanted
later, train it on something richer than a bag of claim tokens — the cycle's full context,
the action taken, and the outcome — and hold the token-overlap baseline to beat.

## 16.5 Where training did not beat the hand-written baseline

Stated plainly, because three of these are the interesting results:

1. **The intention ranker loses catastrophically in closed loop** (0.05, 0.00, 0.00 item
   accuracy against 1.00) despite 0.79 per-decision accuracy. Compounding, not capability.
2. **Claim embeddings lose to token overlap** on both associative and structural recall,
   by both training methods.
3. **The label encoder does not generalise across operating systems** (0.281 against 0.304
   at full coverage on macOS and Windows frames) and is **a tie in the live path**, because
   label matching was not the binding constraint there.
4. **The role model is below the majority baseline on overall accuracy** for held-out
   operating systems, and scores 0.0 recall on the two classes it barely saw in training.

And what did work: role classification from pixels, which turned the weakest number in the
perception report (0.15-0.17) into 0.82-0.83 and held it across platforms; and the label
encoder's calibration on unseen phrasings (0.469 against 0.215 mean score, clearing the
claim threshold three times out of four against once), which matters wherever a downstream
rule needs a score rather than a ranking.

## 16.6 Reproducing

```sh
SP=<scratchpad>
PYTHONPATH=src:. python eval/training/train_label_encoder.py --out $SP/learned --data $SP/vision_data --cache $SP/vision_cache_final --epochs 30
PYTHONPATH=src:. python eval/training/eval_label_encoder.py  --model $SP/learned --data $SP/vision_data --cache $SP/vision_cache_final
PYTHONPATH=src:. python eval/training/train_role_model.py    --data $SP/vision_data --cache $SP/vision_cache_final --out $SP/learned
PYTHONPATH=src:. python eval/training/rollout_decisions.py   --out $SP/learned/decisions_train.jsonl   --episodes 60 --first-seed 1
PYTHONPATH=src:. python eval/training/rollout_decisions.py   --out $SP/learned/decisions_heldout.jsonl --episodes 20 --first-seed 9001
PYTHONPATH=src:. python eval/training/train_intention_ranker.py --train $SP/learned/decisions_train.jsonl --held-out $SP/learned/decisions_heldout.jsonl --out $SP/learned --encoder $SP/learned/label_encoder.npz
PYTHONPATH=src:. python eval/training/eval_intention_ranker.py  --model $SP/learned --episodes 20 --first-seed 9001
PYTHONPATH=src:. python eval/training/eval_access_labels.py     --model $SP/learned --episodes 30 --first-seed 9101
PYTHONPATH=src:. python eval/training/train_claim_embeddings.py --recording $SP/live_run6.jsonl --out $SP/learned --cycles 4000 --method ppmi-svd
```

Results: `eval/results/learned_label_matching.json`, `learned_role_model.report.json`,
`learned_intention_ranker.json` and `.report.json`, `learned_access_labels.json`,
`learned_claim_embeddings.report.json`, `learned_label_encoder.manifest.json`.
Contracts are tested in `tests/test_learned_components.py` (11 tests); the full suite is
814 passed, 4 skipped.
