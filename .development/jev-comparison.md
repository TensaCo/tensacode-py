# Jev (TypeSafe System One) comparison

Surveyed 2026-09-22, one week after Jev's public launch. Sources are the public
docs (docs.typesafe.ai, api.typesafe.ai/openapi.json), the launch blog, the HN
thread, jevals.com, github.com/yibie/awesome-jev and independent write-ups. Vendor
speed/cost figures are self-reported; independent measurements are far smaller.

## What Jev is

A text-only "typed decision" model: `POST /v1/systemone` with `{state, model,
questions}`. Questions are `noul` (yes/no probability), `choice` (≤255 described
options) or `score` (≤10 ordered levels). Answers return full probability
distributions plus a concentration-derived `confidence`. It never generates text.
Many questions against one state are evaluated in isolation in one request.

Reported weaknesses: literal instruction following, counting/dates/math,
option-order sensitivity, always answers (callers must add "other/not stated"),
degraded accuracy with irrelevant state, disputed calibration.

## How people use it

- LLM-as-judge replacement and test assertions (one yes/no claim per question).
- Agent harness gating: allow/ask/deny on side-effecting tool calls, risk scores,
  checking tool calls against the user's actual request (LangChain middleware).
- Routing and moderation with per-action confidence thresholds tuned on labels.
- Composite scores: several single-axis `score` questions weighted in code.
- Guarding generator output: citation checks, RAG passage filtering.
- Open clones and distillations (0.4–0.6B decision heads, LoRA + heads) that
  reproduce the interface: typed questions in, a distribution out, one pass.

## How TensorCode compares

| Aspect | Jev | TensorCode today |
|---|---|---|
| Typed decisions | `noul`/`choice`/`score` | `text.Classify`/`Decide`/`Score`/`Retrieve` with strict validation |
| Distribution source | Scored over options in one pass | Owned native ops *generate* JSON including numbers; tools' NLI verifier uses logits |
| Invalid outputs | Impossible by construction | Possible (`InvalidModelOutput`); no constrained decoding |
| Many questions / one state | One request | No primitive; `batch` is one question over many inputs |
| Option descriptions | Per option | Labels only (adapter sends `None` descriptions) |
| Owned, trainable, persisted | No fine-tuning | Yes: weights, experience, checkpoints |
| Evidence provenance, revision, memory | No | Yes (cognitive sessions) |
| Multimodal | Text only | Text + images (VLM path, ViT encoders) |

`integrations.JevModel` maps Choice and Score, one question per request, and
has only been tested against local servers; hosted quality is unevaluated.

## What we can do like that (proposals, not implemented)

1. **Likelihood-scored structured ops.** Give owned `Classify`/`Decide`/`Score` a
   scoring mode: encode once, score each configured alternative with the decoder
   (or a trained head), softmax. Always-valid distributions, one encoder pass,
   trainable with cross-entropy, calibratable with `fit_threshold`. This directly
   targets the pinned blocker (question-conditioned verification).
2. **Multi-question evaluation of one state.** A public way to ask several typed
   operations about the same messages, sharing encoding for owned ops and fusing
   into one request for Jev. Needed for composite rubrics (the response-quality
   assessor's support/relevance/completeness axes are exactly this shape).
3. **Jev as an external baseline/teacher.** Run the frozen development set through
   the same axes with `JevModel` as a direct comparison (needs an API key and the
   owner's consent to send data). Its labels could seed distillation, but must be
   reported as supplied supervision, not learned capability.
4. **Adapter completeness.** Map `noul` for two-label boolean classification and
   pass option descriptions once operations carry them.

Acceptance for (1)/(2) follows the big-picture gates: measured change in
known-failure admission and good-answer retention on frozen data, with the
generated-JSON path as the baseline.
