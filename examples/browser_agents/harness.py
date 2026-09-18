"""Serve the demo apps, run agent episodes in Chromium, and score them.

Scoring reads ``window.__score()``, a hook the harness calls after an episode.
Agent programs only see what ``Browser.observe`` and screenshots expose.
"""

from __future__ import annotations

import functools
import importlib
import os
import http.server
import socketserver
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import tensorcode as tc
from tensorcode.backends.builtin import UtilityChooser

from .browser import Browser, LabelMatcher, classify_feedback
from .mind import run_mind

WEB = Path(__file__).parent / "web"
MODEL_LIBRARIES = ("torch", "transformers", "openai", "anthropic", "vllm", "llama_cpp", "google.generativeai", "litellm")


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args: object) -> None:
        pass

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


class _Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def serve(extra_routes: Callable[[http.server.BaseHTTPRequestHandler], bool] | None = None, port: int = 0) -> tuple[str, _Server]:
    """Static server for web/, optionally with extra routes (used by the live viewer)."""

    class Handler(_QuietHandler):
        def do_GET(self) -> None:  # noqa: N802
            if extra_routes and extra_routes(self):
                return
            super().do_GET()

        def do_POST(self) -> None:  # noqa: N802
            if not (extra_routes and extra_routes(self)):
                self.send_error(404)

    server = _Server(("127.0.0.1", port), functools.partial(Handler, directory=str(WEB)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{server.server_address[1]}", server


@dataclass(frozen=True)
class Task:
    name: str
    url: Callable[[str, int], str]  # (demo server base, seed) -> page to open
    spec: object  # MindSpec
    bindings: Callable[[], list]
    setup: Callable[[int], None] | None = None  # prepare the world (e.g. write a note on a simulated desktop)
    score: Callable[[object, int], dict] | None = None  # default: window.__score() in the page
    viewport: tuple[int, int] = (1100, 760)
    world: Callable[[int], dict] | None = None  # a computerworld definition for this episode: no browser is used

    @property
    def simulated(self) -> bool:
        """Does this task run inside the computerworld engine rather than a real web page?"""
        return self.world is not None


TASK_PAGES = {"access": "access.html", "shop": "shop.html", "recon": "recon.html", "chart": "chart.html", "inbox": "inbox.html", "desktop": None}


def tasks(names: Iterable[str] = TASK_PAGES) -> dict[str, Task]:
    from .mind import scene_graph

    found = {}
    for name in names:
        mod = importlib.import_module(f"{__package__}.tasks.{name}")
        extra = getattr(mod, "bindings", None) or (lambda mod=mod: list(mod.BINDINGS))
        page = TASK_PAGES[name]
        url = (lambda base, seed, page=page: f"{base}/{page}?seed={seed}") if page else (lambda base, seed: "")
        found[name] = Task(name, url, mod.SPEC, lambda extra=extra: [scene_graph, *extra()], getattr(mod, "setup", None), getattr(mod, "score", None),
                           (1280, 800) if page is None else (1100, 760), getattr(mod, "world", None))
    return found


def runtime_for(task: Task) -> tc.Runtime:
    policy = tc.Policy(localities=frozenset({"in_process"}), allow_egress=False, available=frozenset({"sklearn"}), cache=False)
    return tc.Runtime([LabelMatcher(), UtilityChooser(), classify_feedback, *task.bindings()], policy=policy)


@dataclass
class EpisodeResult:
    task: str
    seed: int
    score: dict
    outcome: object
    seconds: float
    actions: int
    observations: int
    browser_s: float
    spans: int
    tensacode_ms: float  # summed span time of inferential ops (parse/classify/choose/rank/check)
    model_calls: int
    blocked_requests: int
    error: str | None = None
    trace_lines: list[str] = field(default_factory=list)


def body_for(task: Task, seed: int, *, context: object = None, base: str = "", page: object = None):
    """The body this task acts through: a computerworld machine, or a Playwright page."""
    if task.simulated:
        from .perception.computerworld import CwProvider
        from .perception.cw_body import CwBody
        from .worlds.runtime import CwWorld

        world = CwWorld(task.world(seed), seed, width=task.viewport[0], height=task.viewport[1])
        return CwBody(world.actor(), CwProvider(), episode=f"{task.name}-{seed}"), world
    page = page or context.new_page()
    if task.setup:
        task.setup(seed)
    page.goto(task.url(base, seed))
    return Browser(page, episode=f"{task.name}-{seed}"), page


def run_episode(context: object = None, base: str = "", task: Task = None, seed: int = 1, *, on_cycle: Callable[..., None] | None = None, page: object | None = None) -> EpisodeResult:
    ui, world = body_for(task, seed, context=context, base=base, page=page)
    runtime = runtime_for(task)
    t0 = time.perf_counter()
    error, outcome = None, None
    with tc.use(runtime):
        try:
            outcome = run_mind(ui, task.spec, on_cycle=(lambda m, t, i: on_cycle(ui, runtime, m, t, i)) if on_cycle else None)
        except Exception as exc:  # noqa: BLE001 - an agent crash is a scored failure, not a harness crash
            error = f"{type(exc).__name__}: {exc}"
    seconds = time.perf_counter() - t0
    score = task.score(world, seed) if task.score else (world.evaluate("() => window.__score ? window.__score() : null") or {})
    spans = runtime.trace.spans
    infer = [s for s in spans if s.op in ("parse", "classify", "choose", "rank", "check")]
    model_calls = sum(1 for s in spans for a in s.attempts if a.outcome != "skipped" and (a.implementation.startswith("chat:") or "model" in a.implementation))
    return EpisodeResult(
        task.name, seed, score, outcome, seconds, ui.stats.actions, ui.stats.observations, ui.stats.browser_s,
        len(spans), sum(s.total_ms for s in infer), model_calls, 0, error,
    )


def block_outside_network(context: object, base: str) -> list[str]:
    blocked: list[str] = []

    def route(r: object) -> None:
        if r.request.url.startswith((base, "data:")):
            r.continue_()
        else:
            blocked.append(r.request.url)
            r.abort()

    context.route("**/*", route)
    return blocked


def model_libraries_loaded() -> list[str]:
    return [m for m in MODEL_LIBRARIES if m in sys.modules]
