"""Chat with an Ubuntu assistant and watch it work, live.

    python -m examples.browser_agents.assistant.server [--port 8770] [--computer ubuntu-2]

Two engines. ``--engine seed`` (the default) drives the separate simulator over HTTP through
a Chromium page, and needs that server running (SEED_URL, default http://127.0.0.1:4391).
``--engine computerworld`` runs the deterministic engine in this process: no browser and no
server, with the engine's own rasterization for the viewer. See
docs/revival/10-computerworld.md for what its shell does and does not cover.

The page shows the desktop, the conversation, how each message was understood, and the
agent's intentions and thoughts.
"""

from __future__ import annotations

import argparse
import collections
import json
import multiprocessing as mp
import os
import queue
import threading
import time
import traceback
import webbrowser
from pathlib import Path

from .. import harness

PAGE = Path(__file__).parent / "chat.html"
SEED = os.environ.get("SEED_URL", "http://127.0.0.1:4391")


def ensure_computer(computer: str) -> str:
    """Use the named Seed computer, or create a fresh Ubuntu machine called 'assistant'."""
    import urllib.request

    state = json.loads(urllib.request.urlopen(f"{SEED}/api/state", timeout=10).read())
    ids = {c["spec"]["id"]: c["spec"] for c in state["computers"]}
    if computer in ids:
        return computer
    named = [cid for cid, spec in ids.items() if spec.get("hostname") == "assistant"]
    if named:
        return named[0]
    req = urllib.request.Request(f"{SEED}/api/computers", data=json.dumps({"os": "ubuntu", "hostname": "assistant"}).encode(), headers={"content-type": "application/json"}, method="POST")
    return json.loads(urllib.request.urlopen(req, timeout=10).read())["id"]


def engine_body(events: mp.Queue, fps: float):
    """A machine from the computerworld engine: no browser, no simulator server.

    Frames for the viewer are the engine's own rasterization, produced on a timer rather
    than pushed by a screencast.
    """
    import base64
    import threading

    from ..perception.computerworld import CwProvider
    from ..perception.cw_body import CwBody
    from ..worlds import desktop_world
    from ..worlds.runtime import CwWorld

    world = CwWorld(desktop_world(), 0)
    ui = CwBody(world.actor(), CwProvider(), episode="assistant")

    def stream() -> None:
        while True:
            time.sleep(1.0 / fps)
            try:
                events.put({"type": "frame", "t": time.time(), "data": base64.b64encode(ui.surface.png()).decode()})
            except Exception:  # noqa: BLE001 - the viewer going away must not stop the agent
                return

    threading.Thread(target=stream, daemon=True).start()
    return ui, world


def worker(computer: str, inbox: mp.Queue, stop: mp.Event, events: mp.Queue, fps: float, engine: str = "seed") -> None:
    from ..mind import describe
    from . import agent
    from .language import parse_message

    from ..mind import scene_graph

    mind = agent.new_mind()
    runtime = harness.runtime_for(harness.Task("assistant", lambda b, s: "", agent.SPEC, lambda: [scene_graph]))
    if engine == "computerworld":
        ui, _world = engine_body(events, fps)
        events.put({"type": "ready", "t": time.time(), "computer": "computerworld:dev"})
        converse(ui, mind, runtime, inbox, stop, events, wait=lambda: time.sleep(0.06))
        return
    from playwright.sync_api import sync_playwright

    from ..browser import Browser

    with sync_playwright() as p:
        browser = p.chromium.launch()
        context = browser.new_context(viewport={"width": 1280, "height": 800}, device_scale_factor=1)
        harness.block_outside_network(context, SEED)
        page = context.new_page()
        cdp = context.new_cdp_session(page)
        last = [0.0]

        def on_frame(params: dict) -> None:
            cdp.send("Page.screencastFrameAck", {"sessionId": params["sessionId"]})
            now = time.monotonic()
            if now - last[0] >= 1.0 / fps:
                last[0] = now
                events.put({"type": "frame", "t": time.time(), "data": params["data"]})

        cdp.on("Page.screencastFrame", on_frame)
        page.goto(f"{SEED}/?computer={computer}")
        page.wait_for_timeout(1500)
        cdp.send("Page.startScreencast", {"format": "jpeg", "quality": 60, "maxWidth": 1280, "maxHeight": 800, "everyNthFrame": 1})
        ui = Browser(page, episode="assistant")
        events.put({"type": "ready", "t": time.time(), "computer": computer})
        converse(ui, mind, runtime, inbox, stop, events, wait=lambda: page.wait_for_timeout(60))


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
    ap.add_argument("--computer", default="ubuntu-2")
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--no-open", action="store_true")
    ap.add_argument("--engine", default=os.environ.get("TENSORCODE_ENGINE", "seed"), choices=("seed", "computerworld"),
                    help="seed: the separate simulator over HTTP; computerworld: the in-process deterministic engine")
    args = ap.parse_args()
    computer = args.computer if args.engine == "computerworld" else ensure_computer(args.computer)
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
    proc = ctx.Process(target=worker, args=(computer, inbox, stop, events, args.fps, args.engine), daemon=True)
    proc.start()
    print(f"assistant: {base}/  ({'computerworld engine' if args.engine == 'computerworld' else f'Seed computer {computer}'})", flush=True)
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
