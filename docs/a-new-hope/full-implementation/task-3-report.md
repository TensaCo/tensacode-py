# Task 3 — message decisions and provider integrations

Implemented on 2026-09-20 against the replacement architecture and the provider documentation cited below.

## Public API

`tensorcode.ops.llm` now exports:

- `Message`, retaining `Message(role, str)` compatibility.
- Frozen `TextPart(text, source_ref=None)` and `ImagePart(data=... | url=..., media_type=None, source_ref=None, detail=None)` values. An image has exactly one byte or URL source. Construction and encoding never download URLs.
- `TextEncoder`, `ImageEncoder`, `TextDecoder`, and `Transform`. `ImageEncoder` accepts raw bytes, an HTTP(S)/data URL, or an already configured `ImagePart`; an explicit part is preserved without applying encoder defaults over it.
- `ModelRequest`, `ModelOutput`, and the runtime-checkable `Model`, `AsyncModel`, and `BatchModel` protocols. Provider-specific serialization is outside the operation vocabulary.
- `Classify`, `Score`, `Decide`, and `Retrieve`, with `ClassificationResult`, `ScoreResult`, `DecisionResult`, and `RetrievalResult`.
- `InvalidModelOutput` for semantic response-contract violations.

Classifications and decisions validate selected values against their configured alternatives. Supplied distributions must contain exactly those alternatives, contain finite values in `[0, 1]`, and sum to one within an absolute tolerance of `0.001`. Scores validate their numeric rubric range and canonical distribution keys. Retrieval returns only configured stable keys and their original items; arbitrary values require explicit text descriptions rather than implicit `repr` conversion. Retrieval scores remain named scores and are never presented as a probability distribution.

Every structured result requires an explicit `abstained` boolean. A valid abstention carries no selected label, choice, score, or retrieved key. There is no default threshold. `distribution` and `confidence` remain `None` when the backend omits them; TensorCode does not synthesize either value.

Operations have explicit `acall`. Structured operations also have `batch` and `abatch`. `batch` uses `model.complete_batch` when available outside tracing and verifies one result per request. Inside an active trace it uses ordinary per-item operation calls so `OutputRef` resolution, failures, and replay boundaries remain accurate. `abatch` runs explicit asynchronous calls concurrently and works with an async-only model. Sync calls always return values rather than awaitables.

## Integrations

`tensorcode.integrations.OpenAICompatibleModel` supports two explicit HTTP modes:

- `api="chat_completions"` sends `POST /chat/completions` with documented text/image content parts and `response_format.type="json_schema"` for structured operations.
- `api="responses"` sends `POST /responses`, sets `store=false`, uses input text/image parts, and uses `text.format.type="json_schema"` when requested.

Image bytes become base64 data URLs using their supplied media type; image URLs remain URLs. Provider serialization does not fetch either. Structured response text is parsed once and then validated by the operation. There is no repair prompt or implicit malformed-output retry. Chat truncation, non-stop completion, and refusal are errors. Responses output must be completed, error-free, and non-refusal; all output text blocks are retained in order. Redirects are rejected before credentials can be forwarded to another origin.

`tensorcode.integrations.JevModel` sends the documented `POST /v1/systemone` request and maps TensorCode classification/decision schemas to Jev Choice and score schemas to Jev Score. Jev's returned probabilities and confidence are retained with their provider semantics. TensorCode sets `abstained=false` only when a documented answer is present. The adapter rejects plain chat, image inputs, and Retrieve because the primary API material reviewed did not establish those capabilities. It does not approximate them through prompts or a fallback provider.

Both HTTP adapters use one request per `complete` call, with no retries or fallback. Provider failures remain operation failures. Their `configuration()` values are JSON-safe and omit API keys; exceptions and `repr` also exclude configured secrets. HTTP-backed operations retain the base operation default `replayable=false`.

`tensorcode.integrations.LocalModel` is the separately supplied Transformers adapter used by the local evaluation work. It implements the same `ModelRequest`/`ModelOutput` contract and does not change the remote provider claims in this report.

## Primary provider sources

- OpenAI Chat API reference: <https://developers.openai.com/api/reference/cli/resources/chat>
- OpenAI Responses create reference: <https://developers.openai.com/api/reference/cli/resources/beta/subresources/responses/methods/create>
- OpenAI schema-name constraint in the official Python API types: <https://github.com/openai/openai-python/blob/main/src/openai/types/responses/response_format_text_json_schema_config.py>
- TypeSafe's official Python SDK: <https://github.com/typesafe-ai/typesafe-sdk-python>
- TypeSafe's generated models from `https://api.typesafe.ai/openapi.json`, including System One, Choice, Score, and Noul request/response fields: <https://github.com/typesafe-ai/typesafe-sdk-python/blob/main/src/typesafe_sdk/_schemas/models.py>
- TypeSafe's official endpoint builder for `POST /v1/systemone`: <https://github.com/typesafe-ai/typesafe-sdk-python/blob/main/src/typesafe_sdk/_core/endpoints.py>

## Verification and scope

The focused tests are:

- `tests/test_llm_messages.py`
- `tests/test_llm_decisions.py`
- `tests/test_provider_http.py`
- Parent-owned `tests/test_local_model.py` and tool integration tests consume the shared interface.

Local `ThreadingHTTPServer` fixtures exercise real JSON serialization, HTTP headers, endpoint paths, multimodal data URLs and URL references, structured outputs, HTTP errors, redirect rejection, timeouts, malformed envelopes, truncation/refusal handling, and secret exclusion. These fixtures verify transport behavior only. They are not reported as model-quality evaluation. No OpenAI or TypeSafe API key was configured, so no live remote inference was run. The full implementation report records the separate real local pretrained-model evaluation.

No runtime dependency was added for the HTTP adapters; they use the Python standard library. The local adapter's optional dependencies and model evaluation are recorded by the parent integration work.

Known limits are explicit: OpenAI-compatible servers must implement the selected wire mode; source references remain in TensorCode messages but the OpenAI wire has no corresponding field, so they are not inserted into prompt text; Jev support is limited to the typed operations established by its official schema; HTTP calls are buffered and non-streaming; and provider-side batch job APIs are not treated as synchronous operation batching.
