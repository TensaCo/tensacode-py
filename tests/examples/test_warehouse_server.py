"""Actual local preview HTTP boundary, independent of physics or model weights."""

import json
import sys
import threading
from functools import partial
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from http.server import ThreadingHTTPServer

from examples.robotics.warehouse.representation import Entity, SceneGraph
from examples.robotics.warehouse.server import RecordedInterpreter, make_handler


@pytest.fixture
def interpreter():
    scene = SceneGraph()
    for i in range(2):
        scene.upsert(
            Entity(
                f"parcel_{i}",
                "parcel",
                {"color": "orange"},
                (i, 0, 0.8),
                3.0,
                "rgbd:frame1",
            )
        )
        scene.upsert(
            Entity(
                f"receiver_{i}",
                "station",
                {"color": "magenta"},
                (i, 0.3, 0.9),
                3.0,
                "rgbd:frame1",
            )
        )
    return RecordedInterpreter(scene, 3.0)


def test_language_changes_dependency_plan_without_execution(interpreter):
    text = "move parcel_0 to receiver_0 then move parcel_1 to receiver_1"
    sequential = interpreter.interpret({"text": text, "mode": "instruction"})
    parallel = interpreter.interpret(
        {"text": text.replace(" then ", " while "), "mode": "instruction"}
    )
    assert sequential["status"] == parallel["status"] == "supported"
    assert sequential["tasks"][1]["dependencies"] == ("task-1",)
    assert parallel["tasks"][1]["dependencies"] == ()
    assert sequential["execution"] is False
    assert sequential["scope"] == "recorded_observation_preview"


def test_unknown_and_ambiguous_requests_do_not_make_tasks(interpreter):
    for text, status in [
        ("move parcel to receiver_0", "ambiguous"),
        ("move parcel_99 to receiver_0", "unknown"),
    ]:
        result = interpreter.interpret({"text": text, "mode": "instruction"})
        assert result["status"] == status
        assert not result.get("tasks")
    answer = interpreter.interpret({"text": "Where is parcel_0?", "mode": "question"})
    assert answer["answer"]["sources"] == ("rgbd:frame1",)


def test_malformed_or_oversized_input_rejected(interpreter):
    for data in (
        {"text": "x"},
        {"text": "x", "mode": "execute"},
        {"text": "x" * 1501, "mode": "question"},
    ):
        with pytest.raises(ValueError):
            interpreter.interpret(data)


def test_http_serves_preview_and_rejects_cross_origin(interpreter, tmp_path):
    (tmp_path / "index.html").write_text("warehouse")
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(interpreter, tmp_path))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}"
    request = partial(
        Request,
        url + "/api/warehouse/interpret",
        data=json.dumps({"text": "Where is parcel_0?", "mode": "question"}).encode(),
        method="POST",
    )
    try:
        with urlopen(url) as response:
            assert response.read() == b"warehouse"
        with urlopen(request()) as response:
            assert json.load(response)["status"] == "supported"
        with pytest.raises(HTTPError) as error:
            urlopen(request(headers={"Origin": "https://untrusted.example"}))
        assert error.value.code == 403
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_model_wrong_order_is_rejected_before_grounding(interpreter):
    from examples.robotics.warehouse.model_language import ModelProposal

    class Model:
        def propose(self, text, prompt):
            return ModelProposal(
                '{"instruction":["sequence",["move","parcel_0","receiver_0"],["move","parcel_1","receiver_1"]]}',
                30,
                0.01,
                "fixture-prompt",
            )

    interpreter.model = Model()
    result = interpreter.interpret(
        {
            "text": "Move parcel_0 to receiver_0 while moving parcel_1 to receiver_1",
            "mode": "instruction",
        }
    )
    assert result["status"] == "rejected"
    assert result["semantic_admission"]["status"] == "rejected"
    assert not result.get("tasks")
