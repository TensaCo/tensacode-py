import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from tensorcode.integrations import (
    JevModel,
    OpenAICompatibleModel,
    ProviderHTTPError,
    ProviderProtocolError,
    ProviderTimeout,
)
from tensorcode.ops import text as text_ops


class Server:
    def __init__(self, responder):
        self.requests = []
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("content-length", "0"))
                body = self.rfile.read(length)
                owner.requests.append(
                    {
                        "path": self.path,
                        "headers": {key.lower(): value for key, value in self.headers.items()},
                        "json": json.loads(body),
                    }
                )
                response = responder(owner.requests[-1])
                status, payload, delay = response[:3]
                extra_headers = response[3] if len(response) == 4 else {}
                if delay:
                    time.sleep(delay)
                encoded = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(encoded)))
                for key, value in extra_headers.items():
                    self.send_header(key, value)
                self.end_headers()
                try:
                    self.wfile.write(encoded)
                except BrokenPipeError:
                    pass

            def log_message(self, format, *args):
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    @property
    def url(self):
        host, port = self.httpd.server_address
        return f"http://{host}:{port}"

    def close(self):
        self.httpd.shutdown()
        self.thread.join()
        self.httpd.server_close()


@pytest.fixture
def server_factory():
    servers = []

    def make(responder):
        server = Server(responder)
        servers.append(server)
        return server

    yield make
    for server in servers:
        server.close()


def test_openai_compatible_chat_wire_preserves_multimodal_parts(server_factory):
    def respond(_request):
        content = json.dumps(
            {
                "label": "cat",
                "distribution": {"cat": 0.75, "dog": 0.25},
                "confidence": None,
                "abstained": False,
            }
        )
        return 200, {"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": content}}]}, 0

    server = server_factory(respond)
    model = OpenAICompatibleModel(
        base_url=server.url + "/v1", model="vision-test", api_key="secret-key"
    )
    operation = text_ops.Classify(model, labels=("cat", "dog"), instructions="Identify it")
    message = text_ops.Message(
        "user",
        (
            text_ops.TextPart("What animal?", source_ref="prompt:1"),
            text_ops.ImagePart(data=b"image-bytes", media_type="image/png", source_ref="upload:1"),
            text_ops.ImagePart(url="https://example.test/cat.jpg", detail="low"),
        ),
    )

    result = operation((message,))

    assert result.label == "cat"
    request = server.requests[0]
    assert request["path"] == "/v1/chat/completions"
    assert request["headers"]["authorization"] == "Bearer secret-key"
    assert request["json"]["model"] == "vision-test"
    assert request["json"]["messages"][0] == {"role": "system", "content": "Identify it"}
    content = request["json"]["messages"][1]["content"]
    assert content[0] == {"type": "text", "text": "What animal?"}
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert content[2] == {
        "type": "image_url",
        "image_url": {"url": "https://example.test/cat.jpg", "detail": "low"},
    }
    assert request["json"]["response_format"]["json_schema"]["strict"] is True
    assert request["json"]["response_format"]["json_schema"]["name"] == "tensorcode_classify"
    operation_configuration = operation.configuration()
    assert operation_configuration["model"]["type"] == "openai_compatible"
    assert "api_key" not in json.dumps(operation_configuration)


def test_openai_compatible_plain_text_and_responses_api(server_factory):
    def respond(request):
        assert request["json"]["store"] is False
        return 200, {
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {"type": "output_text", "text": "ans"},
                        {"type": "output_text", "text": "wer"},
                    ],
                }
            ]
        }, 0

    server = server_factory(respond)
    model = OpenAICompatibleModel(
        base_url=server.url + "/v1", model="text-test", api="responses"
    )
    output = model.complete(text_ops.ModelRequest((text_ops.Message("user", "hello"),)))
    assert output.text == "answer"
    assert server.requests[0]["path"] == "/v1/responses"
    assert server.requests[0]["json"]["input"] == [
        {"role": "user", "content": [{"type": "input_text", "text": "hello"}]}
    ]


def test_http_provider_errors_timeout_and_secret_safe_configuration(server_factory):
    error_server = server_factory(lambda request: (503, {"error": "key=secret-key"}, 0))
    model = OpenAICompatibleModel(
        base_url=error_server.url + "/v1",
        model="test",
        api_key="secret-key",
        timeout=0.2,
    )
    with pytest.raises(ProviderHTTPError) as caught:
        model.complete(text_ops.ModelRequest((text_ops.Message("user", "hello"),)))
    assert caught.value.status == 503
    assert "secret-key" not in str(caught.value)
    assert "secret-key" not in repr(model)
    assert "api_key" not in model.configuration()
    json.dumps(model.configuration())

    slow_server = server_factory(lambda request: (200, {"choices": []}, 0.2))
    slow = OpenAICompatibleModel(
        base_url=slow_server.url + "/v1", model="test", timeout=0.03
    )
    with pytest.raises(ProviderTimeout):
        slow.complete(text_ops.ModelRequest((text_ops.Message("user", "hello"),)))


@pytest.mark.parametrize(
    "payload",
    [
        {"choices": []},
        {"choices": [{"finish_reason": "stop", "message": {"content": 7}}]},
        {"choices": [{"finish_reason": "stop", "message": {"content": "not json"}}]},
    ],
)
def test_openai_rejects_malformed_structured_responses(server_factory, payload):
    server = server_factory(lambda request: (200, payload, 0))
    model = OpenAICompatibleModel(base_url=server.url + "/v1", model="test")
    classify = text_ops.Classify(model, labels=("yes", "no"))
    with pytest.raises(ProviderProtocolError):
        classify((text_ops.Message("user", "question"),))


def test_openai_rejects_redirect_without_forwarding_authorization(server_factory):
    target = server_factory(
        lambda request: (
            200,
            {"choices": [{"finish_reason": "stop", "message": {"content": "unexpected"}}]},
            0,
        )
    )
    redirect = server_factory(
        lambda request: (302, {}, 0, {"Location": target.url + "/stolen"})
    )
    model = OpenAICompatibleModel(
        base_url=redirect.url + "/v1", model="test", api_key="do-not-forward"
    )
    with pytest.raises(ProviderHTTPError) as caught:
        model.complete(text_ops.ModelRequest((text_ops.Message("user", "hello"),)))
    assert caught.value.status == 302
    assert target.requests == []


def test_openai_rejects_truncation_refusal_and_incomplete_responses(server_factory):
    chat = server_factory(
        lambda request: (
            200,
            {"choices": [{"finish_reason": "length", "message": {"content": "partial"}}]},
            0,
        )
    )
    with pytest.raises(ProviderProtocolError, match="finish"):
        OpenAICompatibleModel(base_url=chat.url + "/v1", model="test").complete(
            text_ops.ModelRequest((text_ops.Message("user", "hello"),))
        )

    responses = server_factory(
        lambda request: (
            200,
            {
                "status": "incomplete",
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "refusal", "refusal": "no"}],
                    }
                ],
            },
            0,
        )
    )
    with pytest.raises(ProviderProtocolError, match="status"):
        OpenAICompatibleModel(
            base_url=responses.url + "/v1", model="test", api="responses"
        ).complete(text_ops.ModelRequest((text_ops.Message("user", "hello"),)))


def test_jev_wire_maps_documented_choice_and_score_answers(server_factory):
    answers = [
        {
            "model": "jev-latest",
            "answers": {
                "result": {
                    "type": "choice",
                    "choice": "billing",
                    "confidence": 0.8,
                    "probabilities": {"billing": 0.8, "technical": 0.2},
                }
            },
            "usage": {"input_tokens": 10, "output_tokens": 2},
        },
        {
            "model": "jev-latest",
            "answers": {
                "result": {
                    "type": "score",
                    "score": 1.7,
                    "confidence": 0.9,
                    "legend": {"0": "low", "1": "medium", "2": "high"},
                    "probabilities": {"0": 0.1, "1": 0.1, "2": 0.8},
                }
            },
            "usage": {"input_tokens": 10, "output_tokens": 2},
        },
    ]
    server = server_factory(lambda request: (200, answers.pop(0), 0))
    model = JevModel(base_url=server.url, api_key="jev-secret")

    classification = text_ops.Classify(
        model, labels=("billing", "technical"), instructions="Route ticket"
    )((text_ops.Message("user", "charged twice"),))
    score = text_ops.Score(model, rubric=("low", "medium", "high"), instructions="Urgency")(
        (text_ops.Message("user", "help now"),)
    )

    assert classification.distribution == {"billing": 0.8, "technical": 0.2}
    assert classification.confidence == 0.8
    assert score.value == 1.7
    first = server.requests[0]
    assert first["path"] == "/v1/systemone"
    assert first["headers"]["authorization"] == "Bearer jev-secret"
    assert first["json"]["questions"] == {
        "result": {
            "type": "choice",
            "instructions": "Route ticket",
            "criteria": {"billing": None, "technical": None},
        }
    }
    second = server.requests[1]["json"]
    assert second["questions"]["result"]["criteria"] == ["low", "medium", "high"]


def test_jev_rejects_unsupported_multimodal_and_retrieval_requests(server_factory):
    server = server_factory(lambda request: (200, {}, 0))
    model = JevModel(base_url=server.url, api_key="key")
    with pytest.raises(ProviderProtocolError, match="image"):
        text_ops.Classify(model, labels=("a", "b"))(
            (text_ops.Message("user", (text_ops.ImagePart(data=b"x", media_type="image/png"),)),)
        )

    retrieve = text_ops.Retrieve(
        model,
        items={"a": "first", "b": "second"},
        descriptions={"a": "first", "b": "second"},
    )
    with pytest.raises(ProviderProtocolError, match="retrieve"):
        retrieve((text_ops.Message("user", "which"),))
