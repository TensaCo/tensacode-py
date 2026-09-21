import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from threading import Thread

from tensorcode.integrations import OpenAICompatibleModel
from tensorcode.ops import llm
from tensorcode.tools.agents import Chatbot, JsonMemory


def test_chatbot_image_bytes_reach_provider_through_public_message_operations():
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers["content-length"])
            self.server.payload = json.loads(self.rfile.read(length))
            body = json.dumps(
                {
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {"content": "a small PNG"},
                        }
                    ]
                }
            ).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever)
    thread.start()
    try:
        model = OpenAICompatibleModel(
            base_url=f"http://127.0.0.1:{server.server_port}/v1",
            model="local-vlm-fixture",
        )
        image_encoder = llm.ImageEncoder(
            media_type="image/png", source_ref="upload:7", detail="high"
        )
        bot = Chatbot(model=model, encode_image=image_encoder)
        image = b"\x89PNG\r\n\x1a\nfixture"

        assert bot("What is in this image?", images=[image]) == "a small PNG"

        payload = server.payload
        assert payload["model"] == "local-vlm-fixture"
        assert payload["messages"][0] == {
            "role": "user",
            "content": "What is in this image?",
        }
        image_wire = payload["messages"][1]["content"][0]
        assert image_wire["type"] == "image_url"
        assert image_wire["image_url"] == {
            "url": "data:image/png;base64," + base64.b64encode(image).decode(),
            "detail": "high",
        }
        image_message = bot.history[1]
        assert image_message.content[0].source_ref == "upload:7"
        assert image_message.content[0].data == image
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def test_message_memory_preserves_multimodal_parts_across_restart(tmp_path):
    memory = JsonMemory.for_messages(
        tmp_path / "messages.json",
        retrieve=lambda request: request.candidates[-request.limit :],
    )
    respond = lambda messages, *, context=None: messages + (
        llm.Message("assistant", (llm.TextPart("answer", source_ref="model:1"),)),
    )
    bot = Chatbot(
        respond=respond,
        encode_image=llm.ImageEncoder(
            media_type="image/png", source_ref="upload:8", detail="low"
        ),
        memory=memory,
    )

    bot("look", images=[b"\x89PNG data"])
    live_history = bot.history

    restarted = Chatbot(
        respond=respond,
        memory=JsonMemory.for_messages(
            tmp_path / "messages.json",
            retrieve=lambda request: request.candidates[-request.limit :],
        ),
    )
    assert restarted.history == live_history
    image = restarted.history[1].content[0]
    assert image.data == b"\x89PNG data"
    assert image.source_ref == "upload:8"
    assert image.detail == "low"
