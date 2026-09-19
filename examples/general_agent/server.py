"""Chat with the general agent while it works a computerworld desktop.

    python -m examples.general_agent.server [--port 8770] [--no-open]

A dev server: it runs until you stop it and installs nothing. The agent and the
desktop plugin live in one worker process (the engine's handle cannot cross threads);
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


def worker(inbox: mp.Queue, events: mp.Queue, fps: float, reader: str | None = None) -> None:
    from examples.browser_agents.worlds import desktop_world
    from examples.browser_agents.worlds.runtime import CwWorld
    from examples.general_agent.desktop import DesktopPlugin
    from tensorcode.agent import Agent
    from tensorcode.agent.plugin import describe_capabilities
    from tensorcode.agent.vision_plugin import VisionPlugin

    world = CwWorld(desktop_world(), 0)
    last = [0.0]
    holder: dict = {}

    def frame(force: bool = False) -> None:
        now = time.monotonic()
        if "plugin" not in holder or (not force and now - last[0] < 1.0 / fps):
            return
        last[0] = now
        try:
            picture = holder["plugin"].surface.png()
        except Exception as exc:  # noqa: BLE001
            # no encoder on this host, or the engine could not draw: the conversation is the
            # point, and it used to take the whole server down with it
            if not holder.get("said_no_frames"):
                holder["said_no_frames"] = True
                events.put({"type": "note", "t": time.time(),
                            "text": f"no desktop picture here ({type(exc).__name__}); the chat and the events still work"})
            return
        events.put({"type": "frame", "t": time.time(), "data": base64.b64encode(picture).decode()})

    plugin = DesktopPlugin(world, on_step=frame)
    holder["plugin"] = plugin
    agent = Agent([plugin, VisionPlugin()], reader=reader)
    frame(force=True)
    events.put({"type": "ready", "t": time.time(), "capabilities": describe_capabilities(agent.plugins),
                "places": plugin.places, "apps": sorted(plugin.apps)})
    while True:
        try:
            message = inbox.get(timeout=1.0 / fps)
        except queue.Empty:
            frame()
            continue
        text, images = message["text"], [base64.b64decode(b) for b in message.get("images", [])]
        events.put({"type": "busy", "t": time.time(), "busy": True})
        started = time.perf_counter()
        try:
            turn = agent.turn(text, images=images)
            for ev in turn.events:
                events.put({**ev, "t": time.time()})
            events.put({"type": "chat", "t": time.time(), "from": "agent", "text": turn.reply, "seconds": turn.seconds})
        except Exception as exc:  # noqa: BLE001 - a crash is shown, and the agent keeps listening
            traceback.print_exc()
            events.put({"type": "chat", "t": time.time(), "from": "agent", "text": f"Something broke on my side: {type(exc).__name__}: {exc}"})
        frame(force=True)
        events.put({"type": "busy", "t": time.time(), "busy": False, "seconds": round(time.perf_counter() - started, 2)})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--no-open", action="store_true")
    ap.add_argument("--reader", default="learned", choices=["learned", "grammar"],
                    help="which registered parse implementation to prefer")
    args = ap.parse_args()
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
    proc = ctx.Process(target=worker, args=(inbox, events, args.fps, None if args.reader == 'grammar' else args.reader), daemon=True)
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
