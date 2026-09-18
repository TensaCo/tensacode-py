"""Chat with an Ubuntu assistant and watch it work, live.

    python -m examples.browser_agents.assistant.server [--port 8770]

The machine is the computerworld engine, running in this process: no browser and no
separate server. Frames for the viewer are the engine's own rasterization. See
docs/revival/10-computerworld.md for what its shell does and does not cover.

The page shows the desktop, the conversation, how each message was understood, and the
agent's intentions and thoughts.
"""

from __future__ import annotations

import argparse
import collections
import json
import multiprocessing as mp
import queue
import threading
import time
import traceback
import webbrowser
from pathlib import Path

from .. import harness

PAGE = Path(__file__).parent / "chat.html"


def engine_body(events: mp.Queue, fps: float):
    """A machine from the computerworld engine, plus a way to send the viewer a frame.

    The engine is not thread-safe (its Python handle must stay on the thread that made it),
    so frames are rendered here, on the worker thread: after every action and while idle,
    at most ``fps`` times a second.
    """
    import base64

    from ..perception.computerworld import CwProvider
    from ..perception.cw_body import CwBody
    from ..worlds import desktop_world
    from ..worlds.runtime import CwWorld

    world = CwWorld(desktop_world(), 0)
    last = [0.0]

    def frame(force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - last[0] < 1.0 / fps:
            return
        last[0] = now
        events.put({"type": "frame", "t": time.time(), "data": base64.b64encode(ui.surface.png()).decode()})

    ui = CwBody(world.actor(), CwProvider(), episode="assistant", on_step=frame)
    return ui, world, frame


def worker(inbox: mp.Queue, stop: mp.Event, events: mp.Queue, fps: float) -> None:
    from ..mind import describe
    from . import agent
    from .language import parse_message

    from ..mind import scene_graph

    mind = agent.new_mind()
    runtime = harness.runtime_for(harness.Task("assistant", lambda b, s: "", agent.SPEC, lambda: [scene_graph]))
    ui, _world, frame = engine_body(events, fps)
    frame(force=True)
    events.put({"type": "ready", "t": time.time(), "computer": "computerworld:dev"})

    def wait() -> None:
        frame()
        time.sleep(0.06)

    converse(ui, mind, runtime, inbox, stop, events, wait=wait)


def converse(ui, mind, runtime, inbox: mp.Queue, stop: mp.Event, events: mp.Queue, *, wait) -> None:
    """Answer messages until the process ends. The body decides what "waiting" costs."""
    import traceback

    from ..mind import describe
    from . import agent
    from .language import parse_message

    while True:
        try:
            text = inbox.get_nowait()
        except queue.Empty:
            wait()
            continue
        stop.clear()
        frames = parse_message(text)
        events.put({"type": "understood", "t": time.time(), "text": text, "frames": [{"act": f.act, "words": f.words, "slots": {k: str(v) for k, v in f.slots.items()}} for f in frames]})
        events.put({"type": "busy", "t": time.time(), "busy": True})
        started, actions0 = time.perf_counter(), ui.stats.actions

        def on_cycle(m, thought, intention) -> None:
            if stop.is_set():
                raise KeyboardInterrupt("stopped by user")
            events.put({"type": "cycle", "t": time.time(), "intention": getattr(intention, "why", None) or getattr(intention, "reason", ""),
                        "thoughts": describe(thought, limit=12), "beliefs": len(m._claims), "actions": ui.stats.actions})

        def on_say(reply: str) -> None:
            events.put({"type": "chat", "t": time.time(), "from": "agent", "text": reply, "seconds": round(time.perf_counter() - started, 2)})

        try:
            with __import__("tensorcode").use(runtime):
                outcome = agent.respond(ui, mind, text, on_say=on_say, on_cycle=on_cycle)
            status, reason = getattr(outcome, "status", "?"), getattr(outcome, "reason", "")
        except KeyboardInterrupt:
            for req in agent.open_requests(mind):
                agent.set_state(mind, req, "status", "dropped", "user:stop")
            agent._clear_awaiting(mind, 0)
            status, reason = "stopped", "stopped by you"
            on_say("Stopped.")
        except Exception as exc:  # noqa: BLE001
            status, reason = "error", f"{type(exc).__name__}: {exc}"
            on_say(f"Something broke on my side: {reason}")
            traceback.print_exc()
        if status == "escalated":
            on_say(f"I got stuck ({reason}) and stopped.")
        events.put({"type": "busy", "t": time.time(), "busy": False, "status": status, "reason": reason,
                    "seconds": round(time.perf_counter() - started, 2), "actions": ui.stats.actions - actions0,
                    "model_calls": agent.BODY.model_calls, "model_libraries": harness.model_libraries_loaded()})


class Hub:
    """Fan events out to viewers: every non-frame event in order, but only the newest frame."""

    def __init__(self) -> None:
        self.lock = threading.Condition()
        self.clients: list[dict] = []
        self.history: collections.deque = collections.deque(maxlen=600)
        self.last_frame: dict | None = None

    def publish(self, ev: dict) -> None:
        with self.lock:
            if ev["type"] == "frame":
                self.last_frame = ev
                for c in self.clients:
                    c["frame"] = ev
            else:
                self.history.append(ev)
                for c in self.clients:
                    c["events"].append(ev)
            self.lock.notify_all()

    def serve(self, handler) -> None:
        client = {"events": collections.deque(self.history), "frame": self.last_frame}
        with self.lock:
            self.clients.append(client)
        handler.send_response(200)
        handler.send_header("Content-Type", "text/event-stream")
        handler.send_header("Cache-Control", "no-store")
        handler.end_headers()
        try:
            while True:
                with self.lock:
                    while not client["events"] and client["frame"] is None:
                        if not self.lock.wait(timeout=15):
                            break
                    batch = list(client["events"])
                    client["events"].clear()
                    frame, client["frame"] = client["frame"], None
                out = "".join(f"data: {json.dumps(ev)}\n\n" for ev in batch)
                if frame is not None:
                    out += f"data: {json.dumps(frame)}\n\n"
                handler.wfile.write((out or ": keepalive\n\n").encode())
                handler.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            with self.lock:
                self.clients.remove(client)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args()
    ctx = mp.get_context("spawn")
    inbox, events, stop = ctx.Queue(), ctx.Queue(), ctx.Event()
    hub = Hub()

    def routes(handler) -> bool:
        path = handler.path.split("?")[0]
        if path == "/" and handler.command == "GET":
            body = PAGE.read_bytes()
            handler.send_response(200)
            handler.send_header("Content-Type", "text/html; charset=utf-8")
            handler.end_headers()
            handler.wfile.write(body)
            return True
        if path == "/events":
            hub.serve(handler)
            return True
        if path in ("/say", "/stop") and handler.command == "POST":
            raw = handler.rfile.read(int(handler.headers.get("Content-Length") or 0)) or b"{}"
            code, reply = 200, {"ok": True}
            if path == "/stop":
                stop.set()
            else:
                text = str(json.loads(raw).get("text", "")).strip()[:2000]
                if text:
                    hub.publish({"type": "chat", "t": time.time(), "from": "user", "text": text})
                    inbox.put(text)
                else:
                    code, reply = 400, {"ok": False, "error": "empty message"}
            handler.send_response(code)
            handler.send_header("Content-Type", "application/json")
            handler.end_headers()
            handler.wfile.write(json.dumps(reply).encode())
            return True
        return False

    base, server = harness.serve(routes, port=args.port)
    proc = ctx.Process(target=worker, args=(inbox, stop, events, args.fps), daemon=True)
    proc.start()
    print(f"assistant: {base}/  (computerworld engine)", flush=True)
    if not args.no_open:
        webbrowser.open(f"{base}/")
    while proc.is_alive():
        try:
            hub.publish(events.get(timeout=1.0))
        except queue.Empty:
            continue
    print("worker exited", flush=True)
    server.shutdown()


if __name__ == "__main__":
    main()
