"""Serve the diagnosis lab with grounded language previews of recorded evidence.

This service does not execute robot actions. The physics runner separately consumes
validated instruction trees. Bind only to loopback; no model downloads occur.
"""

from __future__ import annotations

import argparse
import json
import threading
from dataclasses import asdict
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .language import (
    GroundingError,
    answer_question,
    ground_instruction,
    parse_instruction,
)
from .representation import SceneGraph

RESULTS = Path(__file__).resolve().parents[3] / "docs/results/robotics"


class RecordedInterpreter:
    def __init__(self, scene: SceneGraph, observed_at: float, model=None):
        self.scene = scene
        self.observed_at = observed_at
        self.model = model
        self.lock = threading.Lock()

    def interpret(self, payload: dict) -> dict:
        if not isinstance(payload, dict) or set(payload) != {"text", "mode"}:
            raise ValueError("Expected text and mode")
        text, mode = payload["text"], payload["mode"]
        if not isinstance(text, str) or not 0 < len(text.strip()) <= 1500:
            raise ValueError("Text must contain 1–1500 characters")
        if mode not in {"instruction", "question"}:
            raise ValueError("Mode must be instruction or question")
        result = {
            "scope": "recorded_observation_preview",
            "execution": False,
            "motion_feasibility": "not_checked",
            "parser": "frozen_model" if self.model else "restricted_grammar",
            "scene_revision": self.scene.revision,
            "observation_time_s": self.observed_at,
        }
        try:
            if self.model is not None:
                from .model_language import (
                    MODEL_ID,
                    MODEL_REVISION,
                    V2_SYSTEM_PROMPT,
                    check_semantic_admission,
                    decode_request,
                    ground_request,
                )

                result["model"] = {
                    "id": MODEL_ID,
                    "revision": MODEL_REVISION,
                    "prompt_version": "concise_v2",
                }
                proposal = self.model.propose(text, V2_SYSTEM_PROMPT)
                request = decode_request(proposal.raw_text)
                admission = check_semantic_admission(text, request)
                result["semantic_admission"] = admission
                if admission["status"] != "admitted":
                    return {
                        **result,
                        "status": "rejected",
                        "reason": "; ".join(admission["reasons"]),
                        "model_proposal": asdict(proposal),
                    }
                expected = "instruction" if mode == "instruction" else "query"
                if request["kind"] not in {expected, "unsupported"}:
                    return {
                        **result,
                        "status": "unsupported",
                        "reason": "Parsed request does not match selected mode",
                    }
                result.update(ground_request(request, self.scene, self.observed_at))
                if request["kind"] == "instruction":
                    result["instruction_tree"] = request["tree"].to_dict()
                result["model_proposal"] = asdict(proposal)
            elif mode == "question":
                answer = asdict(answer_question(text, self.scene, self.observed_at))
                result.update(status=answer["status"], answer=answer)
            else:
                tree = parse_instruction(text)
                result["instruction_tree"] = tree.to_dict()
                tasks = ground_instruction(tree, self.scene, self.observed_at)
                result.update(status="supported", tasks=[asdict(t) for t in tasks])
        except GroundingError as exc:
            result.update(
                status=exc.status, reason=str(exc), candidates=list(exc.candidates)
            )
        except (ValueError, KeyError, TypeError) as exc:
            result.update(status="unsupported", reason=str(exc))
        return result


def make_handler(interpreter: RecordedInterpreter, root: Path):
    root = root.resolve()

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(root), **kwargs)

        def translate_path(self, path):
            translated = Path(super().translate_path(path)).resolve()
            if not translated.is_relative_to(root):
                return str(root / "__denied_path__")
            return str(translated)

        def send_json(self, status, body):
            content = json.dumps(body, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(content)

        def do_POST(self):
            if self.path != "/api/warehouse/interpret":
                self.send_json(404, {"status": "error", "reason": "Unknown endpoint"})
                return
            origin = self.headers.get("Origin")
            if origin and origin != "http://" + self.headers.get("Host", ""):
                self.send_json(
                    403, {"status": "error", "reason": "Same-origin requests required"}
                )
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 8192:
                    raise ValueError("Invalid request size")
                payload = json.loads(self.rfile.read(length))
            except (ValueError, UnicodeError) as exc:
                self.send_json(400, {"status": "error", "reason": str(exc)})
                return
            if not interpreter.lock.acquire(blocking=False):
                self.send_json(
                    429,
                    {"status": "busy", "reason": "Another interpretation is running"},
                )
                return
            try:
                self.send_json(200, interpreter.interpret(payload))
            except ValueError as exc:
                self.send_json(400, {"status": "error", "reason": str(exc)})
            finally:
                interpreter.lock.release()

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--parser", choices=("grammar", "model"), default="grammar")
    args = parser.parse_args()
    payload = json.loads(args.scene.read_text())
    scene = SceneGraph.from_dict(payload.get("scene", payload))
    observed_at = max((e.observed_at for e in scene.entities.values()), default=0.0)
    model = None
    if args.parser == "model":
        from .model_language import FrozenLanguageModel

        model = FrozenLanguageModel()
    interpreter = RecordedInterpreter(scene, observed_at, model)
    server = ThreadingHTTPServer(
        ("127.0.0.1", args.port), make_handler(interpreter, RESULTS)
    )
    print(
        f"http://127.0.0.1:{args.port}/ — recorded-scene preview, no robot execution",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
