"""OpenAI-compatible Chat Completions and Responses HTTP model adapter."""
from __future__ import annotations

import asyncio
import base64
import json
import re

from ..ops.text.messages import ImagePart, TextPart
from ..ops.text.model import ModelOutput, ModelRequest
from ._http import ProviderProtocolError, endpoint, post_json


class OpenAICompatibleModel:
    """Call an explicitly configured OpenAI-compatible JSON endpoint.

    ``api='chat_completions'`` targets the broadly implemented
    ``/chat/completions`` contract. ``api='responses'`` targets ``/responses``.
    Requests are never retried or silently routed to another provider.
    """

    def __init__(self, *, base_url, model, api_key=None, timeout=30.0, api="chat_completions"):
        if not isinstance(model, str) or not model:
            raise ValueError("model must be a nonempty string")
        if api not in ("chat_completions", "responses"):
            raise ValueError("api must be 'chat_completions' or 'responses'")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.api = api
        # Validate before a request is attempted.
        endpoint(self.base_url, "chat/completions" if api == "chat_completions" else "responses")

    def __repr__(self):
        return (
            f"OpenAICompatibleModel(base_url={self.base_url!r}, model={self.model!r}, "
            f"api={self.api!r}, timeout={self.timeout!r})"
        )

    def configuration(self):
        return {
            "type": "openai_compatible",
            "base_url": self.base_url,
            "model": self.model,
            "api": self.api,
            "timeout": self.timeout,
        }

    def complete(self, request: ModelRequest):
        if not isinstance(request, ModelRequest):
            raise TypeError("complete expects ModelRequest")
        if self.api == "chat_completions":
            payload = self._chat_payload(request)
            response = post_json(
                endpoint(self.base_url, "chat/completions"),
                payload,
                api_key=self.api_key,
                timeout=self.timeout,
            )
            text = _chat_text(response)
        else:
            payload = self._responses_payload(request)
            response = post_json(
                endpoint(self.base_url, "responses"),
                payload,
                api_key=self.api_key,
                timeout=self.timeout,
            )
            text = _responses_text(response)
        metadata = {
            key: response[key]
            for key in ("id", "model", "usage", "status")
            if key in response
        }
        if request.response_schema is None:
            return ModelOutput(text=text, provider_metadata=metadata or None)
        try:
            structured = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ProviderProtocolError("Structured provider response is not valid JSON") from exc
        if not isinstance(structured, dict):
            raise ProviderProtocolError("Structured provider response must be a JSON object")
        return ModelOutput(structured=structured, provider_metadata=metadata or None)

    async def acomplete(self, request: ModelRequest):
        return await asyncio.to_thread(self.complete, request)

    def _chat_payload(self, request):
        messages = []
        if request.instructions:
            messages.append({"role": "system", "content": request.instructions})
        messages.extend(_chat_message(message) for message in request.messages)
        payload = {"model": self.model, "messages": messages, "stream": False}
        if request.response_schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": _schema_name(request.schema_name),
                    "strict": True,
                    "schema": dict(request.response_schema),
                },
            }
        return payload

    def _responses_payload(self, request):
        payload = {
            "model": self.model,
            "input": [_responses_message(message) for message in request.messages],
            "store": False,
        }
        if request.instructions:
            payload["instructions"] = request.instructions
        if request.response_schema is not None:
            payload["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": _schema_name(request.schema_name),
                    "strict": True,
                    "schema": dict(request.response_schema),
                }
            }
        return payload


def _parts(content):
    return (TextPart(content),) if isinstance(content, str) else content


def _schema_name(value):
    normalized = re.sub(r"[^A-Za-z0-9_-]", "_", value or "tensorcode_response")
    return normalized[:64] or "tensorcode_response"


def _image_url(part):
    if part.url is not None:
        return part.url
    media_type = part.media_type or "application/octet-stream"
    encoded = base64.b64encode(part.data).decode("ascii")
    return f"data:{media_type};base64,{encoded}"


def _chat_message(message):
    if isinstance(message.content, str):
        return {"role": message.role, "content": message.content}
    content = []
    for part in _parts(message.content):
        if isinstance(part, TextPart):
            content.append({"type": "text", "text": part.text})
        elif isinstance(part, ImagePart):
            image_url = {"url": _image_url(part)}
            if part.detail is not None:
                image_url["detail"] = part.detail
            content.append({"type": "image_url", "image_url": image_url})
        else:
            raise TypeError("Unsupported message part")
    return {"role": message.role, "content": content}


def _responses_message(message):
    content = []
    for part in _parts(message.content):
        if isinstance(part, TextPart):
            content.append({"type": "input_text", "text": part.text})
        elif isinstance(part, ImagePart):
            item = {"type": "input_image", "image_url": _image_url(part)}
            if part.detail is not None:
                item["detail"] = part.detail
            content.append(item)
        else:
            raise TypeError("Unsupported message part")
    return {"role": message.role, "content": content}


def _chat_text(response):
    try:
        choice = response["choices"][0]
        finish_reason = choice["finish_reason"]
        message = choice["message"]
        text = message["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ProviderProtocolError("Chat completion response has no assistant content") from exc
    if finish_reason != "stop":
        raise ProviderProtocolError(f"Chat completion finish reason is {finish_reason!r}, not 'stop'")
    if message.get("refusal"):
        raise ProviderProtocolError("Chat completion returned a refusal")
    if not isinstance(text, str):
        raise ProviderProtocolError("Chat completion assistant content must be text")
    return text


def _responses_text(response):
    if response.get("status") != "completed":
        raise ProviderProtocolError(f"Responses status is {response.get('status')!r}, not 'completed'")
    if response.get("error") is not None:
        raise ProviderProtocolError("Responses response contains an error")
    if isinstance(response.get("output_text"), str):
        return response["output_text"]
    texts = []
    try:
        for output in response["output"]:
            if output.get("type") == "message":
                for part in output.get("content", ()):
                    if part.get("type") == "refusal":
                        raise ProviderProtocolError("Responses response contains a refusal")
                    if part.get("type") == "output_text" and isinstance(part.get("text"), str):
                        texts.append(part["text"])
    except (TypeError, KeyError) as exc:
        raise ProviderProtocolError("Responses response has invalid output content") from exc
    if texts:
        return "".join(texts)
    raise ProviderProtocolError("Responses response has no output text")
