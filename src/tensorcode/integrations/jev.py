"""TypeSafe Jev System One HTTP adapter based on its published OpenAPI schema."""
from __future__ import annotations

import asyncio
from collections.abc import Mapping

from ..ops.text.messages import ImagePart, TextPart
from ..ops.text.model import ModelOutput, ModelRequest
from ._http import ProviderProtocolError, endpoint, post_json


class JevModel:
    """Adapt TensorCode decision requests to ``POST /v1/systemone``.

    Jev is a typed evaluation model rather than a chat model. This adapter
    supports classification, decisions and rubric scores; labels exactly
    ``true``/``false`` use Jev's yes/no question. ``complete_questions`` sends
    several questions about the same messages in one request. Unsupported
    request shapes fail before making an HTTP request.
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
        """JSON description of this adapter; never includes credentials."""
        return {
            "type": "jev",
            "base_url": self.base_url,
            "model": self.model,
            "timeout": self.timeout,
        }

    def complete(self, request: ModelRequest):
        """Send one request and return a ``ModelOutput``."""
        return self.complete_questions({"result": request})["result"]

    async def acomplete(self, request: ModelRequest):
        """Asynchronous ``complete`` (runs the blocking call in a thread)."""
        return await asyncio.to_thread(self.complete, request)

    def complete_questions(self, requests):
        """Send named decision requests about identical messages in one call."""
        if not isinstance(requests, Mapping) or not requests:
            raise TypeError("complete_questions expects a nonempty mapping of ModelRequest")
        if not all(isinstance(request, ModelRequest) for request in requests.values()):
            raise TypeError("complete_questions expects ModelRequest values")
        states = [request.messages for request in requests.values()]
        if any(state != states[0] for state in states[1:]):
            raise ProviderProtocolError("Jev questions in one request must share the same messages")
        questions = {name: _question(request) for name, request in requests.items()}
        payload = {
            "state": _state(next(iter(requests.values()))),
            "model": self.model,
            "questions": {name: question for name, (question, _kind) in questions.items()},
        }
        response = post_json(
            endpoint(self.base_url, "v1/systemone"),
            payload,
            api_key=self.api_key,
            timeout=self.timeout,
        )
        try:
            answers = response["answers"]
        except (KeyError, TypeError) as exc:
            raise ProviderProtocolError("Jev response has no answers") from exc
        if not isinstance(answers, Mapping) or set(answers) != set(requests):
            raise ProviderProtocolError("Jev answers do not match the requested questions")
        metadata = {key: response[key] for key in ("model", "usage") if key in response}
        outputs = {}
        for name, request in requests.items():
            answer = answers[name]
            if not isinstance(answer, dict) or answer.get("type") != questions[name][1]:
                raise ProviderProtocolError("Jev answer has the wrong type")
            outputs[name] = ModelOutput(
                structured=_canonical(request, answer), provider_metadata=metadata or None
            )
        return outputs

    async def acomplete_questions(self, requests):
        """Asynchronous ``complete_questions``."""
        return await asyncio.to_thread(self.complete_questions, requests)


_SELECTION = {"tensorcode.classify": "label", "tensorcode.decide": "choice"}


def _alternatives(request):
    """Configured alternatives and optional descriptions from a selection schema."""
    schema = request.response_schema
    field = _SELECTION[request.schema_name]
    try:
        alternatives = [item for item in schema["properties"][field]["enum"] if item is not None]
        described = schema["properties"]["distribution"]["properties"]
        descriptions = {alternative: described[alternative].get("description") for alternative in alternatives}
    except (KeyError, TypeError, AttributeError) as exc:
        raise ProviderProtocolError("Selection schema is not compatible with Jev Choice") from exc
    return alternatives, descriptions


def _question(request):
    if request.schema_name in _SELECTION:
        alternatives, descriptions = _alternatives(request)
        if set(alternatives) == {"true", "false"}:
            # Exactly the labels true/false map to Jev's yes/no probability.
            question = {"type": "noul", "instructions": request.instructions}
            if any(descriptions.values()):
                question["criteria"] = {"true": descriptions["true"], "false": descriptions["false"]}
            return question, "noul"
        return {
            "type": "choice",
            "instructions": request.instructions,
            "criteria": descriptions,
        }, "choice"
    if request.schema_name == "tensorcode.score":
        try:
            properties = request.response_schema["properties"]["distribution"]["properties"]
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


def _canonical(request, answer):
    schema_name = request.schema_name
    if schema_name in _SELECTION:
        field = _SELECTION[schema_name]
        if answer["type"] == "noul":
            probability = answer.get("noul")
            if isinstance(probability, bool) or not isinstance(probability, (int, float)) or not 0 <= probability <= 1:
                raise ProviderProtocolError("Jev noul answer is not a probability")
            return {
                field: "true" if probability >= 0.5 else "false",
                "distribution": {"true": probability, "false": 1 - probability},
                "confidence": None,
                "abstained": False,
            }
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
