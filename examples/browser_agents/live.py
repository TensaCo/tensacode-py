"""Watch cognitive browser agents work, live, with no model calls.

    python -m examples.browser_agents.live [--tasks access,shop,recon,chart,inbox] [--record run.jsonl] [--minutes 2]

Opens a local page with one video window per agent (Chromium screencast frames),
each agent's current intention and newest thoughts, and running scores. The page can
command the agents: pause, slow down, rerun a seed, leave a note on the simulated
desktop, or send the inbox a customer message (POST /command).
"""

from __future__ import annotations

import argparse
import dataclasses
import io
import json
import multiprocessing as mp
import queue
import threading
import time
import webbrowser
from collections import deque
from pathlib import Path

import tensorcode as tc

from . import harness
from .mind import describe, run_mind

VIEWER = Path(__file__).parent / "viewer.html"
LOW_SIZE = (660, 460)  # wall thumbnails
HIGH_FPS = 20.0  # a focused agent's frame cap
FOCUS_LEASE_S = 15.0  # a focusing viewer heartbeats every 5 s; without it the agent drops back to low resolution


class Controls:
    """What the viewer has asked of one agent. Drained between cycles and between episodes."""

    def __init__(self, inbox: mp.Queue, events: mp.Queue, index: int, on_resolution=lambda high: None) -> None:
        self.inbox, self.events, self.index = inbox, events, index
        self.paused, self.pace, self.queue = False, 0.0, []  # pace: seconds between actions (0 = full speed)
        self.on_resolution, self.high = on_resolution, False

    def drain(self) -> None:
        while True:
            try:
                cmd = self.inbox.get_nowait()
            except queue.Empty:
                return
            kind = cmd.get("kind")
            if kind == "resolution":  # a viewer focused (or unfocused) this agent: re-encode the screencast
                if bool(cmd.get("high")) != self.high:
                    self.high = bool(cmd.get("high"))
                    self.on_resolution(self.high)
                continue
            if kind == "pause":
                self.paused = True
            elif kind == "resume":
                self.paused = False
            elif kind == "pace":
                self.pace = max(0.0, min(5.0, float(cmd.get("seconds") or 0)))
            elif kind in ("seed", "note", "message"):
                self.queue.append(cmd)
            self.status(f"{kind} received")

    def status(self, said: str) -> None:
        self.events.put({"type": "control", "agent": self.index, "t": time.time(), "said": said, "paused": self.paused, "pace": self.pace,
                         "queued": [c.get("text", c.get("seed")) for c in self.queue]})

    def hold(self, page: object) -> None:
        """Between cycles: honour pause and pace while keeping the page (and its screencast) running."""
        self.drain()
        if self.pace:
            page.wait_for_timeout(self.pace * 1000)
        while self.paused:
            page.wait_for_timeout(100)
            self.drain()


def engine_worker(index: int, task_name: str, events: mp.Queue, commands: mp.Queue, stop_at: float, fps: float, first_seed: int) -> None:
    """An agent inside the computerworld engine: frames are the engine's own rasterization.

    There is no browser here, so nothing pushes frames; they are rendered on a timer at the
    size the wall currently wants.
    """
    import base64

    task = harness.tasks([task_name])[task_name]
    low_gap, high_gap = 1.0 / fps, 1.0 / HIGH_FPS
    state = {"gap": low_gap, "last": 0.0, "size": LOW_SIZE, "body": None}

    def resolution(high: bool) -> None:
        state["gap"] = high_gap if high else low_gap
        state["size"] = task.viewport if high else LOW_SIZE
        events.put({"type": "resolution", "agent": index, "t": time.time(), "high": high, "size": list(state["size"])})

    controls = Controls(commands, events, index, on_resolution=resolution)
    resolution(False)

    def frame() -> None:
        body = state["body"]
        now = time.monotonic()
        if body is None or now - state["last"] < state["gap"]:
            return
        state["last"] = now
        try:
            from PIL import Image

            picture = Image.fromarray(body.surface.picture())
            picture.thumbnail(state["size"])
            buffer = io.BytesIO()
            picture.save(buffer, format="JPEG", quality=80 if state["size"] != LOW_SIZE else 55)
            events.put({"type": "frame", "agent": index, "t": time.time(), "data": base64.b64encode(buffer.getvalue()).decode()})
        except Exception:  # noqa: BLE001 - a dropped frame must not stop the agent
            pass

    seed, custom_n = first_seed, 0
    while time.time() < stop_at:
        controls.drain()
        while controls.paused:
            frame()
            time.sleep(0.1)
            controls.drain()
        episode_task, episode_seed, command = task, seed, None
        if controls.queue:
            command = controls.queue.pop(0)
            if command["kind"] == "seed":
                episode_seed = int(command["seed"])
            elif command["kind"] == "note" and task_name == "desktop":
                from .tasks import desktop

                custom_n += 1
                episode_seed = 900000 + 1000 * index + custom_n
                desktop.NOTES[episode_seed] = command["text"]
        events.put({"type": "start", "agent": index, "task": task_name, "seed": episode_seed, "t": time.time(),
                    "command": command and {k: v for k, v in command.items() if k != "agent"}})
        ui, world = harness.body_for(episode_task, episode_seed)
        state["body"] = ui
        cycle_n = [0]
        runtime = harness.runtime_for(episode_task)
        started = time.perf_counter()

        def on_cycle(mind, thought, intention) -> None:
            cycle_n[0] += 1
            frame()
            controls.drain()
            if controls.pace:
                time.sleep(controls.pace)
            while controls.paused:
                frame()
                time.sleep(0.1)
                controls.drain()
            events.put({"type": "cycle", "agent": index, "t": time.time(), "cycle": cycle_n[0],
                        "intention": getattr(intention, "why", None) or getattr(intention, "reason", repr(intention)),
                        "thoughts": describe(thought), "actions": ui.stats.actions, "beliefs": len(mind._claims)})

        error, outcome = None, None
        with tc.use(runtime):
            try:
                outcome = run_mind(ui, episode_task.spec, on_cycle=on_cycle)
            except Exception as exc:  # noqa: BLE001
                error = f"{type(exc).__name__}: {exc}"
        score = episode_task.score(world, episode_seed) if episode_task.score else {}
        frame()
        events.put({
            "type": "episode", "agent": index, "task": task_name, "seed": episode_seed, "t": time.time(),
            "correct": score.get("correct", 0), "items": score.get("items", 0),
            "seconds": round(time.perf_counter() - started, 3), "actions": ui.stats.actions, "model_calls": 0,
            "outcome": getattr(outcome, "status", None), "reason": getattr(outcome, "reason", error),
            "model_libraries": harness.model_libraries_loaded(), "commanded": command is not None,
            "typed": score.get("want") if (command and score.get("want", {}).get("custom")) else None,
            "checks": score.get("checks") if command else None,
        })
        if command is None:
            seed += 1
    events.put({"type": "stopped", "agent": index})


def worker(index: int, task_name: str, base: str, events: mp.Queue, commands: mp.Queue, stop_at: float, fps: float, first_seed: int) -> None:
    from playwright.sync_api import sync_playwright

    task = harness.tasks([task_name])[task_name]
    if task.simulated:  # no browser to screencast: the engine renders its own frames
        engine_worker(index, task_name, events, commands, stop_at, fps, first_seed)
        return
    low_gap, high_gap = 1.0 / fps, 1.0 / HIGH_FPS
    frame_gap, last_frame = [low_gap], [0.0]
    with sync_playwright() as p:
        browser = p.chromium.launch()
        context = browser.new_context(viewport={"width": task.viewport[0], "height": task.viewport[1]}, device_scale_factor=1)
        harness.block_outside_network(context, base)
        page = context.new_page()
        cdp = context.new_cdp_session(page)

        def on_frame(params: dict) -> None:
            cdp.send("Page.screencastFrameAck", {"sessionId": params["sessionId"]})
            now = time.monotonic()
            if now - last_frame[0] >= frame_gap[0]:
                last_frame[0] = now
                events.put({"type": "frame", "agent": index, "t": time.time(), "data": params["data"]})

        def screencast(high: bool) -> None:
            """Low: small thumbnails for the wall. High (someone focused this agent): the page's full viewport."""
            width, height = task.viewport if high else LOW_SIZE
            try:
                cdp.send("Page.stopScreencast")
            except Exception:  # noqa: BLE001 - not started yet
                pass
            frame_gap[0] = high_gap if high else low_gap
            cdp.send("Page.startScreencast", {"format": "jpeg", "quality": 80 if high else 55, "maxWidth": width, "maxHeight": height, "everyNthFrame": 1})
            events.put({"type": "resolution", "agent": index, "t": time.time(), "high": high, "size": [width, height]})

        controls = Controls(commands, events, index, on_resolution=screencast)
        cdp.on("Page.screencastFrame", on_frame)
        screencast(False)
        seed, custom_n = first_seed, 0
        while time.time() < stop_at:
            controls.hold(page)
            episode_task, episode_seed, command = task, seed, None
            if controls.queue:
                command = controls.queue.pop(0)
                if command["kind"] == "seed":
                    episode_seed = int(command["seed"])
                elif command["kind"] == "note" and task_name == "desktop":
                    from .tasks import desktop

                    custom_n += 1
                    episode_seed = 900000 + 1000 * index + custom_n
                    desktop.NOTES[episode_seed] = command["text"]
                elif command["kind"] == "message" and task_name == "inbox":
                    from urllib.parse import quote

                    url = f"{base}/inbox.html?seed={seed}&msg={quote(command['text'])}"
                    episode_task = dataclasses.replace(task, url=lambda _b, _s, url=url: url)
            events.put({"type": "start", "agent": index, "task": task_name, "seed": episode_seed, "t": time.time(),
                        "command": command and {k: v for k, v in command.items() if k != "agent"}})
            cycle_n = [0]

            def on_cycle(ui, runtime, mind, thought, intention) -> None:
                cycle_n[0] += 1
                events.put({
                    "type": "cycle", "agent": index, "t": time.time(), "cycle": cycle_n[0],
                    "intention": getattr(intention, "why", None) or getattr(intention, "reason", repr(intention)),
                    "thoughts": describe(thought), "actions": ui.stats.actions,
                    "beliefs": len(mind._claims),
                })
                controls.hold(page)

            result = harness.run_episode(context, base, episode_task, episode_seed, on_cycle=on_cycle, page=page)
            typed = result.score.get("typed") or (result.score.get("want") if result.score.get("want", {}).get("custom") else None)
            events.put({
                "type": "episode", "agent": index, "task": task_name, "seed": episode_seed, "t": time.time(),
                "correct": result.score.get("correct", 0), "items": result.score.get("items", 0),
                "seconds": round(result.seconds, 3), "actions": result.actions, "model_calls": result.model_calls,
                "outcome": getattr(result.outcome, "status", None), "reason": getattr(result.outcome, "reason", result.error),
                "model_libraries": harness.model_libraries_loaded(), "commanded": command is not None,
                "typed": typed, "checks": result.score.get("checks") if command else None,
            })
            if command is None:
                seed += 1
        browser.close()
    events.put({"type": "stopped", "agent": index})


class Client:
    """One viewer connection. Non-frame events queue in order; frames coalesce to the newest per agent.

    A browser that paints slower than the agents produce frames therefore never falls behind:
    it skips frames instead of replaying a backlog (which made clicks look ignored). Each agent's
    frames are rate-capped separately, with a higher cap for agents someone has focused.
    """

    def __init__(self, low_gap: float, high_gap: float, high: set[int]) -> None:
        self.cond = threading.Condition()
        self.events: deque[dict] = deque(maxlen=2000)
        self.frames: dict[int, dict] = {}
        self.low_gap, self.high_gap, self.high = low_gap, high_gap, set(high)
        self.last: dict[int, float] = {}

    def gap(self, agent: int) -> float:
        return self.high_gap if agent in self.high else self.low_gap

    def set_high(self, agent: int, high: bool) -> None:
        with self.cond:
            (self.high.add if high else self.high.discard)(agent)
            self.cond.notify()

    def push(self, ev: dict) -> None:
        with self.cond:
            if ev["type"] == "frame":
                self.frames[ev["agent"]] = ev
            else:
                self.events.append(ev)
            self.cond.notify()

    def take(self) -> list[dict]:
        """Block until something is due; return events first, then the newest frames that are due."""
        with self.cond:
            while True:
                now = time.monotonic()
                due = [a for a in self.frames if now - self.last.get(a, 0.0) >= self.gap(a)]
                if self.events or due:
                    out = list(self.events)
                    self.events.clear()
                    for a in due:
                        out.append(self.frames.pop(a))
                        self.last[a] = now
                    return out
                wait = min((self.gap(a) - (now - self.last.get(a, 0.0)) for a in self.frames), default=15.0)
                self.cond.wait(timeout=max(0.005, wait))
                if not self.events and not self.frames:
                    return []  # idle: let the writer send a keep-alive


class _Handled(Exception):
    """A command fully handled in the server (no worker round trip needed for the reply)."""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="access,shop,recon,chart,inbox,desktop")
    ap.add_argument("--minutes", type=float, default=3.0)
    ap.add_argument("--fps", type=float, default=12.0)
    ap.add_argument("--record", type=Path, default=None)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args()
    names = [t for t in args.tasks.split(",") if t]

    clients: list[Client] = []
    lock = threading.Lock()
    history: list[dict] = []  # latest non-frame events for late joiners
    latest_frames: dict[int, dict] = {}
    latest_control: dict[int, dict] = {}  # pause/pace state survives the history window
    leases: dict[int, dict[str, float]] = {}  # agent -> viewer token -> lease expiry (monotonic)
    high_state: dict[int, bool] = {}

    def refresh_focus(agent: int) -> None:
        """High resolution while any viewer holds a live focus lease on the agent (several tabs may)."""
        with lock:
            now = time.monotonic()
            live = {tok: exp for tok, exp in leases.get(agent, {}).items() if exp > now}
            leases[agent] = live
            high = bool(live)
            if high == high_state.get(agent, False):
                return
            high_state[agent] = high
            for client in clients:
                client.set_high(agent, high)
        inboxes[agent].put({"kind": "resolution", "high": high})

    def expire_leases() -> None:
        while True:
            time.sleep(2.0)
            for agent in range(len(names)):
                refresh_focus(agent)

    def routes(handler) -> bool:
        if handler.path in ("/", "/viewer"):
            body = VIEWER.read_bytes()
            handler.send_response(200)
            handler.send_header("Content-Type", "text/html; charset=utf-8")
            handler.end_headers()
            handler.wfile.write(body)
            return True
        if handler.path == "/command" and handler.command == "POST":
            try:
                cmd = json.loads(handler.rfile.read(int(handler.headers.get("Content-Length") or 0)) or b"{}")
                agent = int(cmd["agent"])
                if not 0 <= agent < len(names) or cmd.get("kind") not in ("pause", "resume", "pace", "seed", "note", "message", "resolution"):
                    raise ValueError("unknown agent or command")
                if cmd["kind"] == "resolution":
                    token = str(cmd.get("token") or "")[:64]
                    if not token:
                        raise ValueError("resolution needs a viewer token")
                    with lock:
                        if cmd.get("high"):
                            leases.setdefault(agent, {})[token] = time.monotonic() + FOCUS_LEASE_S
                        else:
                            leases.setdefault(agent, {}).pop(token, None)
                    refresh_focus(agent)
                    raise _Handled
                if cmd["kind"] in ("note", "message") and not str(cmd.get("text", "")).strip():
                    raise ValueError("empty text")
                if cmd["kind"] == "note" and names[agent] != "desktop" or cmd["kind"] == "message" and names[agent] != "inbox":
                    raise ValueError(f"{names[agent]} does not take a {cmd['kind']}")
                cmd["text"] = str(cmd.get("text", ""))[:1200]
                inboxes[agent].put(cmd)
                code, body = 200, {"ok": True}
            except _Handled:
                code, body = 200, {"ok": True, "high": high_state.get(agent, False)}
            except (ValueError, KeyError, json.JSONDecodeError) as exc:
                code, body = 400, {"ok": False, "error": str(exc)}
            handler.send_response(code)
            handler.send_header("Content-Type", "application/json")
            handler.end_headers()
            handler.wfile.write(json.dumps(body).encode())
            return True
        if handler.path == "/events":
            with lock:
                client = Client(low_gap=1.0 / min(args.fps, 10.0), high_gap=1.0 / HIGH_FPS, high={a for a, on in high_state.items() if on})
                clients.append(client)
                backlog = list(history) + list(latest_control.values())
                for ev in latest_frames.values():
                    client.push(ev)
            handler.send_response(200)
            handler.send_header("Content-Type", "text/event-stream")
            handler.send_header("Cache-Control", "no-store")
            handler.end_headers()
            try:
                handler.wfile.write(f"data: {json.dumps({'type': 'hello', 'agents': names})}\n\n".encode())
                for ev in backlog:
                    handler.wfile.write(f"data: {json.dumps(ev)}\n\n".encode())
                handler.wfile.flush()
                while True:
                    batch = client.take()
                    handler.wfile.write(b"".join(f"data: {json.dumps(ev)}\n\n".encode() for ev in batch) if batch else b": keep-alive\n\n")
                    handler.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                with lock:
                    clients.remove(client)
            return True
        return False

    ctx = mp.get_context("fork")
    inboxes = [ctx.Queue() for _ in names]
    base, server = harness.serve(routes, port=args.port)
    events: mp.Queue = ctx.Queue()
    stop_at = time.time() + args.minutes * 60
    procs = [
        # the desktop's world is rebuilt per episode, so a run may reuse seeds freely
        ctx.Process(target=worker, args=(i, name, base, events, inboxes[i], stop_at, args.fps, 1000 * (i + 1)), daemon=True)
        for i, name in enumerate(names)
    ]
    for p in procs:
        p.start()
    threading.Thread(target=expire_leases, daemon=True).start()
    print(f"viewer: {base}/  ({len(names)} agents for {args.minutes} min; no model calls)")
    if not args.no_open:
        webbrowser.open(f"{base}/")

    record = args.record.open("w") if args.record else None
    stopped = 0
    started = time.time()
    while stopped < len(procs):
        try:
            ev = events.get(timeout=1.0)
        except queue.Empty:
            if not any(p.is_alive() for p in procs):
                break
            continue
        ev["t_rel"] = round(ev["t"] - started, 3) if "t" in ev else None
        if ev["type"] == "stopped":
            stopped += 1
        if record:
            record.write(json.dumps(ev) + "\n")
        with lock:
            if ev["type"] == "frame":
                latest_frames[ev["agent"]] = ev
            else:
                if ev["type"] == "control":
                    latest_control[ev["agent"]] = ev
                history.append(ev)
                del history[:-400]
            for client in clients:
                client.push(ev)
    if record:
        record.close()
    print("all agents stopped")
    time.sleep(1)
    server.shutdown()


if __name__ == "__main__":
    main()
