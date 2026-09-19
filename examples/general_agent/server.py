"""Chat with the general agent while it works a computerworld desktop.

    python -m examples.general_agent.server [--port 8770] [--plugin desktop] [--plugin vision]

A dev server: it runs until you stop it and installs nothing. Which plugins the agent has is
chosen on the command line — no computer, one, or several — and the page shows a monitor for
each one that can be looked at. The plugins live in one worker process (the engine's handle
cannot cross threads);
the page receives everything the agent emits — how each sentence parsed, the goal it
became, the plan, the action, the check, the reply — and draws it generically. It
has no components for particular requests.
"""

from __future__ import annotations

import argparse
import base64
import collections
import json
import multiprocessing as mp
import queue
import threading
import time
import traceback
import webbrowser
from pathlib import Path

from examples.browser_agents import harness

PAGE = Path(__file__).parent / "chat.html"


class Hub:
    """Fan events out to viewers: every other event in order, and the newest frame per source.

    Frames are dropped rather than queued — a viewer that falls behind should see the current
    picture, not a backlog of old ones — and there is one current picture *per plugin*, since
    any number of machines may be mounted.
    """

    def __init__(self) -> None:
        self.lock = threading.Condition()
        self.clients: list[dict] = []
        self.history: collections.deque = collections.deque(maxlen=600)
        self.frames: dict[str, dict] = {}

    def publish(self, ev: dict) -> None:
        with self.lock:
            if ev["type"] == "frame":
                source = ev.get("source", "")
                self.frames[source] = ev
                for c in self.clients:
                    c["frames"][source] = ev
            else:
                self.history.append(ev)
                for c in self.clients:
                    c["events"].append(ev)
            self.lock.notify_all()

    def serve(self, handler) -> None:
        client = {"events": collections.deque(self.history), "frames": dict(self.frames)}
        with self.lock:
            self.clients.append(client)
        handler.send_response(200)
        handler.send_header("Content-Type", "text/event-stream")
        handler.send_header("Cache-Control", "no-store")
        handler.end_headers()
        try:
            while True:
                with self.lock:
                    while not client["events"] and not client["frames"]:
                        if not self.lock.wait(timeout=15):
                            break
                    batch = list(client["events"])
                    client["events"].clear()
                    frames, client["frames"] = list(client["frames"].values()), {}
                out = "".join(f"data: {json.dumps(ev)}\n\n" for ev in batch + frames)
                handler.wfile.write((out or ": keepalive\n\n").encode())
                handler.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            with self.lock:
                self.clients.remove(client)


def worker(inbox: mp.Queue, events: mp.Queue, fps: float, reader: str | None = None,
           specs: tuple[str, ...] = ("desktop", "vision")) -> None:
    from examples.general_agent.plugins import attach_agent, mount_all
    from tensorcode.agent import Agent
    from tensorcode.agent.plugin import describe_capabilities

    last: dict[str, float] = {}
    mounted: list = []
    quiet: set[str] = set()

    def frames(force: bool = False) -> None:
        """Send each plugin that can be looked at its newest picture, at most fps per second."""
        now = time.monotonic()
        for m in mounted:
            if not force and now - last.get(m.name, 0.0) < 1.0 / fps:
                continue
            try:
                picture = m.view()
            except Exception as exc:  # noqa: BLE001
                # no encoder here, or the engine could not draw: the conversation is the point,
                # and this used to take the whole server down with it
                if m.name not in quiet:
                    quiet.add(m.name)
                    events.put({"type": "note", "t": time.time(),
                                "text": f"no picture from {m.name} ({type(exc).__name__}); chat and events still work"})
                continue
            if picture is None:
                continue
            last[m.name] = now
            events.put({"type": "frame", "t": time.time(), "source": m.name,
                        "data": base64.b64encode(picture).decode()})

    mounted.extend(mount_all(list(specs), on_step=frames))
    agent = Agent([m.plugin for m in mounted], reader=reader)
    attach_agent(mounted, agent)
    frames(force=True)
    events.put({"type": "ready", "t": time.time(), "capabilities": describe_capabilities(agent.plugins),
                "mounted": [{"name": m.name, "about": m.about, "view": m.has_view} for m in mounted]})
    while True:
        try:
            message = inbox.get(timeout=1.0 / fps)
        except queue.Empty:
            frames()
            continue
        text, images = message["text"], [base64.b64decode(b) for b in message.get("images", [])]
        events.put({"type": "busy", "t": time.time(), "busy": True})
        started = time.perf_counter()
        try:
            turn = agent.turn(text, images=images)
            for ev in turn.events:
                events.put({**ev, "t": time.time()})
            if turn.reply:
                events.put({"type": "chat", "t": time.time(), "from": "agent", "text": turn.reply,
                            "seconds": turn.seconds})
        except Exception as exc:  # noqa: BLE001 - a crash is shown, and the agent keeps listening
            traceback.print_exc()
            events.put({"type": "chat", "t": time.time(), "from": "agent", "text": f"Something broke on my side: {type(exc).__name__}: {exc}"})
        frames(force=True)
        events.put({"type": "busy", "t": time.time(), "busy": False, "seconds": round(time.perf_counter() - started, 2)})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--no-open", action="store_true")
    ap.add_argument("--reader", default="learned", choices=["learned", "grammar"],
                    help="which registered parse implementation to prefer")
    ap.add_argument("--plugin", action="append", default=None, metavar="SPEC",
                    help="mount a plugin; repeatable. 'desktop', 'desktop:note', 'vision'. "
                         "Pass --plugin none for a conversation with no tools at all.")
    args = ap.parse_args()
    specs = tuple(s for s in (args.plugin or ["desktop", "vision", "self"]) if s and s != "none")
    ctx = mp.get_context("spawn")
    inbox, events = ctx.Queue(), ctx.Queue()
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
        if path == "/say" and handler.command == "POST":
            raw = handler.rfile.read(int(handler.headers.get("Content-Length") or 0)) or b"{}"
            body = json.loads(raw)
            text = str(body.get("text", "")).strip()[:20000]
            images = [str(b) for b in body.get("images", [])][:4]
            ok = bool(text or images)
            if ok:
                hub.publish({"type": "chat", "t": time.time(), "from": "user", "text": text, "images": len(images)})
                inbox.put({"text": text, "images": images})
            handler.send_response(200 if ok else 400)
            handler.send_header("Content-Type", "application/json")
            handler.end_headers()
            handler.wfile.write(json.dumps({"ok": ok}).encode())
            return True
        return False

    base, _server = harness.serve(routes, port=args.port)
    proc = ctx.Process(target=worker, args=(inbox, events, args.fps, None if args.reader == 'grammar' else args.reader, specs), daemon=True)
    proc.start()
    print(f"general agent: {base}/", flush=True)
    if not args.no_open:
        webbrowser.open(f"{base}/")
    while proc.is_alive():
        try:
            hub.publish(events.get(timeout=1.0))
        except queue.Empty:
            continue


if __name__ == "__main__":
    main()
