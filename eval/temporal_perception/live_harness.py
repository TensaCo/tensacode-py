"""A headless assistant on its own simulated computer, driven in-process.

The HTTP route (``change_live.py``) can only read replies. Permanence and memory dynamics need to
be measured *inside* the agent — how many object files survived, how many claims the store holds,
how long a turn took — so this harness runs the same assistant in-process against a computer it
creates for itself, on a headless Chromium. It never touches the user's session or display.

Ground truth stays independent: the caller drives the desktop with plain-language commands (so it
knows what it asked for) and this harness observes the screen through its *own* perception pass
into its *own* store, never by asking the agent what it thinks happened.
"""

from __future__ import annotations

import json
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Any

SEED = "http://127.0.0.1:4391"


def make_computer(hostname: str) -> str:
    """A fresh ubuntu machine, so nothing we do disturbs the live assistant's own desktop."""
    state = json.loads(urllib.request.urlopen(f"{SEED}/api/state", timeout=10).read())
    for c in state["computers"]:
        if c["spec"].get("hostname") == hostname:
            return c["spec"]["id"]
    req = urllib.request.Request(f"{SEED}/api/computers", method="POST",
                                 data=json.dumps({"os": "ubuntu", "hostname": hostname}).encode(),
                                 headers={"content-type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=15).read())["id"]


@dataclass
class Turn:
    text: str
    replies: list[str]
    seconds: float
    claims: int      # records held, retracted ones included: what the store costs
    live: int        # claims still standing: what the mind believes
    cycles: int

    @property
    def reply(self) -> str:
        return "\n".join(self.replies)


@dataclass
class Session:
    """A live assistant, in this process, plus an independent eye on the same screen."""

    hostname: str = "perception-eval"
    viewport: tuple[int, int] = (1280, 800)
    turns: list[Turn] = field(default_factory=list)
    computer: str = ""
    _stack: Any = None

    def __enter__(self) -> "Session":
        import tensacode as tc
        from playwright.sync_api import sync_playwright

        from examples.browser_agents import harness
        from examples.browser_agents.assistant import agent
        from examples.browser_agents.browser import Browser
        from examples.browser_agents.mind import scene_graph

        self.tc = tc
        self.agent = agent
        self.computer = make_computer(self.hostname)
        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch()
        ctx = self._browser.new_context(viewport={"width": self.viewport[0], "height": self.viewport[1]},
                                        device_scale_factor=1)
        self.page = ctx.new_page()
        self.page.goto(f"{SEED}/?computer={self.computer}")
        self.page.wait_for_timeout(1800)
        self.ui = Browser(self.page, episode="perception-eval")
        self.mind = agent.new_mind()
        self.runtime = harness.runtime_for(harness.Task("perc", lambda b, s: "", agent.SPEC, lambda: [scene_graph]))
        return self

    def __exit__(self, *exc: object) -> None:
        try:
            self._browser.close()
        finally:
            self._pw.stop()

    # -------------------------------------------------------------- driving

    def send(self, text: str, *, cap_cycles: int = 60) -> Turn:
        replies: list[str] = []
        cycles = [0]

        def on_cycle(m: object, t: object, i: object) -> None:
            cycles[0] += 1
            if cycles[0] > cap_cycles:
                raise RuntimeError("cycle cap")

        t0 = time.perf_counter()
        try:
            with self.tc.use(self.runtime):
                self.agent.respond(self.ui, self.mind, text, on_say=replies.append, on_cycle=on_cycle)
        except RuntimeError as exc:
            if "cycle cap" not in str(exc):
                raise
        held = self.mind._claims
        live = sum(1 for r in held.values() if not r.retracted)
        turn = Turn(text, replies, time.perf_counter() - t0, len(held), live, cycles[0])
        self.turns.append(turn)
        return turn

    # -------------------------------------------------------------- observing

    def observe(self) -> dict[str, object]:
        """Our own perception pass into our own store: ground truth about what is on screen."""
        import tensacode as tc

        from examples.browser_agents.mind import Fragment, SCREEN, integrate
        from tensacode.change import items_from_claims

        mine = tc.Store()
        with self.tc.use(self.runtime):
            frame = tc.parse(self.ui.observe(), Fragment, frame=len(self.turns))
        integrate(mine, frame)
        records = [r for r in mine.claims() if r.claim.scope == SCREEN]
        items = items_from_claims(records, lambda ref: mine.entities.get(ref))
        windows = sorted({i.window for i in items if i.window})
        labels = sorted({i.label for i in items if i.label})
        texts = [i.value for i in items if i.value]
        return {"items": items, "windows": windows, "labels": labels, "texts": texts}
