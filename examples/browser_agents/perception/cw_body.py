"""A body on the computerworld engine: the same agents, no browser and no simulator server.

    world = CwWorld(note_world(note, seed), seed)          # the harness owns this
    ui = CwBody(world.actor(), provider=CwProvider(), episode="desktop-1")
    screen = ui.observe(); ui.click(control); ui.fill(control, "ls ~/Desktop", submit=True)

``Browser``'s shape is kept (observe / click / fill / focused / field_text / screenshot /
caption / stats / last_scene), so minds written against Playwright run unchanged. What
differs underneath:

* **Acting** is ``env.step`` with ``pointer.v1`` / ``keyboard.v1`` envelopes, which the
  engine executes synchronously: after Enter, the command has already run. Nothing waits.
* **Perceiving** is ``env.scene()``, which is the engine's own state rather than a reading
  of a picture of it.
* **The terminal transcript is an efference copy.** This terminal prints output but does
  not echo the command that produced it, so the prompt lines an agent reads back are the
  body's record of what it typed, paired with the output that appeared after it. Those
  lines carry ``method="efference-copy"`` in their provenance. Everything else on screen
  comes from the engine.
"""

from __future__ import annotations

import io
import time
from dataclasses import dataclass, field

import tensorcode as tc

from ..browser import Click, Control, PressKey, Screen, Stats, TypeText, Scroll
from .protocol import PerceivedScene, Provider, Target, TextBlock, own


def replace_texts(scene: PerceivedScene, texts: tuple) -> PerceivedScene:
    """The same scene with a different set of text blocks (the raw one stays on ``last_scene``)."""
    import dataclasses

    return dataclasses.replace(scene, texts=tuple(sorted(texts, key=lambda t: (t.box[1], t.box[0]))))

TERMINAL = "Terminal"


@dataclass(frozen=True)
class Command:
    """One command the terminal ran, as the engine reports it."""

    command: str            # the line the engine printed, prompt and all
    stdout: str
    stderr: str = ""
    exit_code: int | None = None   # None when the engine did not say

    @property
    def ok(self) -> bool | None:
        return None if self.exit_code is None else self.exit_code == 0

    def lines(self) -> list[str]:
        return [line for line in (self.stdout + self.stderr).split("\n") if line.strip()]


@dataclass
class Surface:
    """One machine's screen and keyboard, as an actor sees them."""

    env: object
    machine: str
    user: str = "agent"
    width: int = 1280
    height: int = 800
    commands: list[tuple[str, int]] = field(default_factory=list)  # (command, lines on screen before it ran)

    # -- observation

    def scene(self) -> dict:
        return self.env.scene(self.width, self.height)

    def observation(self) -> dict:
        channels = self.env.observe().get("channels", {})
        semantic = channels.get("semantic.v1", {}).get(self.machine, {})
        return semantic if isinstance(semantic, dict) else {}

    def title(self) -> str:
        return self.observation().get("title", "")

    def elements(self) -> list[dict]:
        """Every semantic element, including the children of groups.

        The engine nests: a terminal entry is a group holding its command line, its output
        and its exit status. A reader that looked only at the top level saw a terminal with
        no output in it at all.
        """
        out: list[dict] = []
        frontier = list(self.observation().get("elements", ()))
        while frontier:
            element = frontier.pop(0)
            out.append(element)
            frontier[:0] = list(element.get("children", ()))
        return out

    def terminal_entries(self) -> list["Command"]:
        """What ran in this terminal, as the engine records it: one entry per command.

        Each carries the command line as the engine printed it, its output, and its **exit
        status** — which is the machine's own answer to "did that work?", in place of reading
        the screen for words like "not found".
        """
        by_id: dict[str, dict] = {}
        for element in self.elements():
            parts = str(element.get("id", "")).split(":")
            if len(parts) == 3 and parts[0] == "terminal-entry":
                by_id.setdefault(parts[1], {})[parts[2]] = element.get("text", "")
        out = []
        for index in sorted(by_id, key=lambda i: int(i) if i.isdigit() else 0):
            fields = by_id[index]
            status = fields.get("exit", "")
            digits = "".join(c for c in status if c.isdigit() or c == "-")
            out.append(Command(command=fields.get("command", ""),
                               stdout=fields.get("stdout", ""), stderr=fields.get("stderr", ""),
                               exit_code=int(digits) if digits else None))
        return out

    def terminal_output(self) -> str:
        return "".join(e.stdout + e.stderr for e in self.terminal_entries())

    def terminal_lines(self) -> list[str]:
        """The terminal's printed lines, unwrapped.

        The scene draws this text hard-wrapped to the window width, which splits words at
        the wrap. The engine's semantic channel carries the same text in its logical lines,
        so that is what a reader here gets — the engine's own state, not a reassembly of
        pictures of it. The wrapped visual lines stay available on the scene for anything
        working from pixels.
        """
        return [line for line in self.terminal_output().split("\n") if line.strip()]

    def has_terminal(self) -> bool:
        return any(e.get("id") == "terminal-input" for e in self.elements())

    def windows(self) -> list[str]:
        return [TERMINAL] if self.has_terminal() else []

    def picture(self):
        """The engine's own rasterization as an RGB array (for the visual pipeline)."""
        import numpy as np

        frame = self.env.render(self.width, self.height)
        buf = np.frombuffer(frame["rgba"], dtype=np.uint8).reshape(frame["height"], frame["width"], 4)
        return buf[:, :, :3].copy()

    def png(self, box: tuple[int, int, int, int] | None = None) -> bytes:
        from PIL import Image

        image = Image.fromarray(self.picture())
        if box:
            x, y, w, h = box
            image = image.crop((x, y, x + w, y + h))
        out = io.BytesIO()
        image.save(out, format="PNG")
        return out.getvalue()

    # -- action

    def act(self, family: str, op: str, payload: dict) -> tuple[bool, object]:
        result = self.env.step([{"family": family, "op": op, "machine": self.machine, "payload": payload}])
        outcome = result["outcomes"][0]
        return bool(outcome.get("success")), (outcome.get("value") if outcome.get("success") else outcome.get("error"))

    def submitted(self, command: str, lines_before: int) -> None:
        """Record a command this body ran, and how many lines had been printed before it did."""
        self.commands.append((command, lines_before))

    def transcript(self, prompt_y: int | None = None) -> list[TextBlock]:
        """Prompt and output lines for this session's commands, in screen order.

        All of it is the engine's own text now. The terminal records one entry per command —
        the line it printed, the output, the exit status — so there is nothing to reconstruct
        and no efference copy to keep: this body used to echo what it had typed, because the
        engine did not, and slice the printed lines by how many there had been before.

        Coordinates are a monotonic stand-in for screen order; the real wrapped positions are
        on the scene this was built from.
        """
        blocks: list[TextBlock] = []
        y = [0]

        def block(text: str, provenance) -> TextBlock:
            y[0] += 18
            return TextBlock(text=text, box=(0, y[0], 600, 16), section=TERMINAL, provenance=provenance)

        printed = own("computerworld", "semantic.v1", "terminal output, unwrapped")
        for entry in self.terminal_entries():
            if entry.command:
                blocks.append(block(entry.command, printed))
            for line in entry.lines():
                blocks.append(block(line, printed))
            if entry.exit_code:  # a failure is part of what the screen says; a zero is not news
                blocks.append(block(f"[exit {entry.exit_code}]", printed))
        blocks.append(block(f"{self.prompt()} {self.input_value()}".rstrip(),
                            own("computerworld", "scene+text", "the idle prompt the engine is drawing")))
        return blocks

    def prompt(self) -> str:
        """The prompt the engine is drawing, rather than one assembled here."""
        label = next((e.get("label", "") for e in self.elements() if e.get("id") == "terminal-input"), "")
        return label

    def input_value(self) -> str:
        return next((e.get("value", "") for e in self.elements() if e.get("id") == "terminal-input"), "")

    def terminal_window(self) -> str | None:
        """The id of the open terminal window, from the scene's own window ids."""
        from .computerworld import window_id

        identities = {window_id(node.get("interaction") or "")
                      for node in self.scene().get("nodes", ())
                      if (node.get("interaction") or "").endswith(":terminal-input")}
        identities.discard(None)
        return next(iter(identities)) if len(identities) == 1 else None

    def clear_terminal(self) -> bool:
        """Empty the terminal, which this shell has no ``clear`` command for.

        The window is closed and reopened, which genuinely empties it. The shell's working
        directory lives on the machine rather than the window, so it survives. The command
        the caller typed is kept in the transcript, because it did happen and the screen did
        clear.
        """
        wid = self.terminal_window()
        if wid is None:
            return False
        closed, _ = self.act("application.v1", "close", {"window": int(wid)})
        opened, _ = self.act("application.v1", "launch", {"kind": "terminal"})
        if not (closed and opened):
            return False
        self.commands.clear()
        return True


class Clock:
    """Stands in for a page's timers. The engine is synchronous: when a step returns, it is done.

    A mind written for a web UI waits for things to settle; here there is never anything in
    flight, so waiting does nothing and the next cycle simply observes again.
    """

    def __init__(self) -> None:
        self.waits = 0
        self.waited_ms = 0

    def wait_for_timeout(self, ms: int) -> None:
        self.waits += 1
        self.waited_ms += int(ms)


class CwExecutor:
    """UI actions as engine action envelopes."""

    def __init__(self, surface: Surface) -> None:
        self.surface = surface

    def execute(self, act: object, *, key: str | None) -> tc.Receipt:
        s = self.surface
        viewport = {"width": s.width, "height": s.height}
        if isinstance(act, Click):
            ok, detail = s.act("pointer.v1", "click", {**viewport, "x": act.x, "y": act.y})
        elif isinstance(act, TypeText):
            if act.replace:
                value = next((e.get("value", "") for e in s.elements() if e.get("id") == "terminal-input"), "")
                for _ in range(min(len(value), 512)):
                    s.act("keyboard.v1", "key", {"key": "Backspace"})
            if act.submit and act.text.strip() == "clear":  # a screen affordance, not a program in this shell
                ok = s.clear_terminal()
                detail = None if ok else "could not reopen the terminal"
                return tc.Receipt(act, "applied", idempotency_key=key) if ok else tc.Receipt(act, "failed", error=str(detail))
            ok, detail = s.act("keyboard.v1", "type", {"text": act.text}) if act.text else (True, None)
            if ok and act.submit:
                before = len(s.terminal_lines())
                ok, detail = s.act("keyboard.v1", "key", {"key": "Enter"})
                if ok:
                    s.submitted(act.text, before)
        elif isinstance(act, PressKey):
            ok, detail = s.act("keyboard.v1", "key", {"key": act.key})
        elif isinstance(act, Scroll):
            ok, detail = True, None  # this engine's launcher desktop does not scroll
        else:
            return tc.Receipt(act, "rejected", error="unsupported action")
        return tc.Receipt(act, "applied", idempotency_key=key) if ok else tc.Receipt(act, "failed", error=str(detail))


class CwBody:
    """What an agent program sees and does inside a computerworld machine."""

    def __init__(self, surface: Surface, provider: Provider, *, episode: str = "cw", on_step=None) -> None:
        self.surface, self.provider, self.episode = surface, provider, episode
        self.executor = CwExecutor(surface)
        self.on_step = on_step or (lambda: None)
        self.stats = Stats()
        self.last_scene: PerceivedScene | None = None  # what the provider saw, wrapped terminal text and all
        self.last_read: PerceivedScene | None = None  # what the mind was handed, with the transcript composed
        self.perception_ms: list[float] = []
        self._step = 0
        self.page = Clock()  # no browser: only the waiting a mind expects to be able to do

    # -- perception

    def target(self) -> Target:
        return Target(detail={"surface": self.surface}, grab=self.surface.picture)

    def observe(self, *, settle_ms: int = 0) -> Screen:
        t0 = time.perf_counter()
        scene = self.provider.perceive(self.target())
        self.last_scene = scene  # everything the provider saw, for fusion and comparisons
        terminal_sections = {element.section for element in scene.elements
                             if any(p.source == "computerworld" and p.locator.endswith(":terminal-input")
                                    for p in element.provenance)}
        if len(terminal_sections) == 1:
            section = next(iter(terminal_sections))
            scene = replace_texts(scene, tuple(t for t in scene.texts if t.section != section)
                                  + tuple(self.surface.transcript()))
        self.last_read = scene
        screen = scene.to_screen()
        dt = time.perf_counter() - t0
        self.perception_ms.append(dt * 1e3)
        self.stats.browser_s += dt
        self.stats.observations += 1
        return screen

    def grab(self):
        return self.surface.picture()

    # -- action

    def _do(self, act: object) -> tc.Receipt:
        self._step += 1
        t0 = time.perf_counter()
        receipt = tc.invoke(act, executor=self.executor, key=f"{self.episode}:{self._step}")
        self.stats.browser_s += time.perf_counter() - t0
        self.stats.actions += 1
        self.on_step()
        return receipt

    def click(self, control: Control, *, wait_ms: int = 0) -> tc.Receipt:
        point = control.point or (control.box[0] + control.box[2] // 2, control.box[1] + control.box[3] // 2)
        return self._do(Click(control.name or control.role, *point))

    def fill(self, control: Control, text: str, *, submit: bool = False) -> tc.Receipt:
        """Verified typing: click the field, confirm the keyboard reaches it, then type."""
        clicked = self.click(control)
        if clicked.status != "applied":
            return tc.Receipt(TypeText(control.name, text, submit), "rejected", error=f"could not click the field: {clicked.error}")
        if not self.focused(control, control.point):
            return tc.Receipt(TypeText(control.name, text, submit), "rejected",
                              error=f"keyboard focus not confirmed in {control.name or control.role!r}")
        return self._do(TypeText(control.name, text, submit))

    def focused(self, control: Control, point) -> bool:
        """Probe: type one character and see whether this field's own value changed."""
        before = self.field_text(control)
        self._do(TypeText(control.name, "x", False, replace=False))
        after = self.field_text(control)
        self._do(PressKey("Backspace"))
        return after is not None and after != before

    def field_text(self, control: Control) -> str | None:
        for element in self.surface.elements():
            if element.get("kind") == "input":
                return element.get("value", "")
        return None

    def scroll(self, dy: int) -> tc.Receipt:
        return self._do(Scroll(dy))

    def caption(self, text: str) -> None:
        pass

    def screenshot(self, box: tuple[int, int, int, int]) -> bytes:
        return self.surface.png(box)
