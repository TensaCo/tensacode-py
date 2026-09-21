"""TypeSafe Jev System One HTTP adapter based on its published OpenAPI schema."""
from __future__ import annotations

import asyncio

from ..ops.llm.messages import ImagePart, TextPart
from ..ops.llm.model import ModelOutput, ModelRequest
from ._http import ProviderProtocolError, endpoint, post_json


class JevModel:
    """Adapt TensorCode decision requests to ``POST /v1/systemone``.

    Jev is a typed evaluation model rather than a chat model. This adapter
    supports classification, decisions and rubric scores; unsupported request
    shapes fail before making an HTTP request.
    """

    def __init__(self, *, api_key, base_url="https://api.typesafe.ai", model="jev-latest", timeout=30.0):
        if not isinstance(api_key, str) or not api_key:
            raise ValueError("api_key must be a nonempty string")
        if not isinstance(model, str) or not model:
            raise ValueError("model must be a nonempty string")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        endpoint(self.base_url, "v1/systemone")

    def __repr__(self):
        return f"JevModel(base_url={self.base_url!r}, model={self.model!r}, timeout={self.timeout!r})"

    def configuration(self):
        return {
            "type": "jev",
            "base_url": self.base_url,
            "model": self.model,
            "timeout": self.timeout,
        }

    def complete(self, request: ModelRequest):
        if not isinstance(request, ModelRequest):
            raise TypeError("complete expects ModelRequest")
        question, result_kind = _question(request)
        payload = {
            "state": _state(request),
            "model": self.model,
            "questions": {"result": question},
        }
        response = post_json(
            endpoint(self.base_url, "v1/systemone"),
            payload,
            api_key=self.api_key,
            timeout=self.timeout,
        )
        try:
            answer = response["answers"]["result"]
        except (KeyError, TypeError) as exc:
            raise ProviderProtocolError("Jev response has no result answer") from exc
        if not isinstance(answer, dict) or answer.get("type") != result_kind:
            raise ProviderProtocolError("Jev result answer has the wrong type")
        structured = _canonical(request.schema_name, answer)
        metadata = {
            key: response[key] for key in ("model", "usage") if key in response
        }
        return ModelOutput(structured=structured, provider_metadata=metadata or None)

    async def acomplete(self, request: ModelRequest):
        return await asyncio.to_thread(self.complete, request)


def _question(request):
    schema = request.response_schema
    if not isinstance(schema, dict) and schema is not None:
        schema = dict(schema)
    if request.schema_name in ("tensorcode.classify", "tensorcode.decide"):
        field = "label" if request.schema_name == "tensorcode.classify" else "choice"
        try:
            alternatives = [item for item in schema["properties"][field]["enum"] if item is not None]
        except (KeyError, TypeError) as exc:
            raise ProviderProtocolError("Selection schema is not compatible with Jev Choice") from exc
        return {
            "type": "choice",
            "instructions": request.instructions,
            "criteria": {alternative: None for alternative in alternatives},
        }, "choice"
    if request.schema_name == "tensorcode.score":
        try:
            properties = schema["properties"]["distribution"]["properties"]
            rubric = [properties[str(index)]["description"] for index in range(len(properties))]
        except (KeyError, TypeError) as exc:
            raise ProviderProtocolError("Score schema is not compatible with Jev Score") from exc
        return {
            "type": "score",
            "instructions": request.instructions,
            "criteria": rubric,
        }, "score"
    if request.schema_name == "tensorcode.retrieve":
        raise ProviderProtocolError("Jev adapter does not map multi-item retrieve requests")
    raise ProviderProtocolError("Jev requires a supported structured decision request")


def _state(request):
    state = []
    for message in request.messages:
        if isinstance(message.content, str):
            content = message.content
        else:
            content = []
            for part in message.content:
                if isinstance(part, ImagePart):
                    raise ProviderProtocolError("Jev documented state does not establish image input support")
                if not isinstance(part, TextPart):
                    raise ProviderProtocolError("Unsupported Jev message part")
                entry = {"type": "text", "text": part.text}
                if part.source_ref is not None:
                    entry["source_ref"] = part.source_ref
                content.append(entry)
        state.append({"role": message.role, "content": content})
    return state


def _canonical(schema_name, answer):
    if schema_name in ("tensorcode.classify", "tensorcode.decide"):
        field = "label" if schema_name == "tensorcode.classify" else "choice"
        try:
            return {
                field: answer["choice"],
                "distribution": answer["probabilities"],
                "confidence": answer["confidence"],
                "abstained": False,
            }
        except KeyError as exc:
            raise ProviderProtocolError("Jev Choice answer is missing required fields") from exc
    if schema_name == "tensorcode.score":
        try:
            return {
                "score": answer["score"],
                "distribution": answer["probabilities"],
                "confidence": answer["confidence"],
                "abstained": False,
            }
        except KeyError as exc:
            raise ProviderProtocolError("Jev Score answer is missing required fields") from exc
    raise ProviderProtocolError("Unsupported Jev result")
