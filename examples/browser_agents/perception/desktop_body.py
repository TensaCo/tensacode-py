"""A body for a real X11 desktop: perceive with any provider, act with real input events.

    body = DesktopBody(window_title="TensaCode Scratch Window", provider=AtspiProvider())
    screen = body.observe()
    body.click(control); body.fill(control, "ledger-lab-7")

Perception comes from a provider (accessibility, vision, or both fused). Acting is
``xdotool``: pointer moves and clicks, and key events to the focused window. Screenshots
come from ImageMagick's ``import``.

**Scope.** This drives one window, named at construction. Every click is checked against
that window's current geometry, and every key event is sent only after input focus has been
put on that window *and verified* (X11 key events go to the focused window, whichever it
is). A click outside the window, or typing while focus is elsewhere, is refused with a
receipt instead of being sent. That is not a permission prompt, it is a
guard against this code touching windows that are not its own while running on somebody's
real desktop.
"""

from __future__ import annotations

import io
import subprocess
import time
from dataclasses import dataclass

import numpy as np

import tensorcode as tc

from ..browser import Click, PressKey, Screen, Stats, TypeText
from .protocol import PerceivedScene, Provider, Target


@dataclass
class X11Window:
    id: str
    title: str
    box: tuple[int, int, int, int]


def find_window(title: str) -> X11Window | None:
    out = subprocess.run(["xdotool", "search", "--name", title], capture_output=True, text=True).stdout.split()
    for wid in out:
        geometry = subprocess.run(["xdotool", "getwindowgeometry", "--shell", wid], capture_output=True, text=True).stdout
        values = dict(line.split("=", 1) for line in geometry.strip().splitlines() if "=" in line)
        name = subprocess.run(["xdotool", "getwindowname", wid], capture_output=True, text=True).stdout.strip()
        if not values or title.lower() not in name.lower():
            continue
        return X11Window(wid, name, (int(values["X"]), int(values["Y"]), int(values["WIDTH"]), int(values["HEIGHT"])))
    return None


class DesktopExecutor:
    """Real pointer and keyboard events, confined to one window."""

    def __init__(self, window_title: str) -> None:
        self.window_title = window_title

    def window(self) -> X11Window | None:
        return find_window(self.window_title)

    def focus_is_ours(self, window: X11Window, attempts: int = 3) -> bool:
        """Put input focus on our window and check that it is really there before any key event."""
        for _ in range(attempts):
            focused = subprocess.run(["xdotool", "getwindowfocus"], capture_output=True, text=True).stdout.strip()
            if focused == window.id:
                return True
            subprocess.run(["xdotool", "windowfocus", "--sync", window.id], check=False, capture_output=True)
            subprocess.run(["xdotool", "windowactivate", "--sync", window.id], check=False, capture_output=True)
            time.sleep(0.15)
        focused = subprocess.run(["xdotool", "getwindowfocus"], capture_output=True, text=True).stdout.strip()
        return focused == window.id

    def execute(self, act: object, *, key: str | None) -> tc.Receipt:
        window = self.window()
        if window is None:
            return tc.Receipt(act, "rejected", error=f"no window titled {self.window_title!r}")
        x, y, w, h = window.box
        if isinstance(act, Click):
            if not (x <= act.x <= x + w and y <= act.y <= y + h):
                return tc.Receipt(act, "rejected", error=f"({act.x},{act.y}) is outside {window.title!r} {window.box}: refusing to click another window")
            subprocess.run(["xdotool", "mousemove", str(act.x), str(act.y), "click", "1"], check=True)
        elif isinstance(act, TypeText):
            # GTK ignores XSendEvent, so real XTEST key events go to whatever holds input focus.
            # Focus must therefore be *verified* to be this window, or the keystrokes would land
            # in somebody else's window (a screenshot grab or a WM refusal can move focus).
            if not self.focus_is_ours(window):
                return tc.Receipt(act, "rejected", error=f"input focus is not on {window.title!r}: refusing to type into another window")
            if act.replace:
                subprocess.run(["xdotool", "key", "--clearmodifiers", "ctrl+a"], check=False)
            if act.text:
                subprocess.run(["xdotool", "type", "--clearmodifiers", "--delay", "12", act.text], check=True)
            if act.submit:
                subprocess.run(["xdotool", "key", "--clearmodifiers", "Return"], check=False)
        elif isinstance(act, PressKey):
            if not self.focus_is_ours(window):
                return tc.Receipt(act, "rejected", error=f"input focus is not on {window.title!r}: refusing to send keys to another window")
            subprocess.run(["xdotool", "key", "--clearmodifiers", act.key], check=False)
        else:
            return tc.Receipt(act, "rejected", error="unsupported action")
        return tc.Receipt(act, "applied", idempotency_key=key)


class DesktopBody:
    """What an agent program sees and does on a real desktop window."""

    def __init__(self, window_title: str, provider: Provider, *, episode: str = "desktop", whole_screen: bool = False) -> None:
        self.window_title, self.provider, self.episode = window_title, provider, episode
        self.executor = DesktopExecutor(window_title)
        self.whole_screen = whole_screen
        self.stats = Stats()
        self.last_scene: PerceivedScene | None = None
        self.perception_ms: list[float] = []
        self._step = 0
        self.page = None  # no browser here; kept so Browser-shaped code can check for it

    # -- perception

    def grab(self) -> np.ndarray:
        """A screenshot of this body's own window (or the whole screen when asked)."""
        from PIL import Image

        window = None if self.whole_screen else self.executor.window()
        args = ["import", "-window", window.id if window else "root", "-silent", "png:-"]
        shot = subprocess.run(args, capture_output=True, check=True).stdout
        return np.asarray(Image.open(io.BytesIO(shot)).convert("RGB"))

    def target(self) -> Target:
        """Vision sees only this window, so its coordinates are shifted into screen space, where
        accessibility already reports. Nothing outside the window is captured or read."""
        window = None if self.whole_screen else self.executor.window()
        origin = (window.box[0], window.box[1]) if window else (0, 0)
        return Target(grab=self.grab, window=None if self.whole_screen else self.window_title, image_origin=origin)

    def observe(self, *, settle_ms: int = 0) -> Screen:
        t0 = time.perf_counter()
        scene = self.provider.perceive(self.target())
        self.last_scene = scene
        dt = time.perf_counter() - t0
        self.perception_ms.append(dt * 1e3)
        self.stats.browser_s += dt
        self.stats.observations += 1
        return scene.to_screen()

    # -- acting

    def _do(self, act: object) -> tc.Receipt:
        self._step += 1
        receipt = tc.invoke(act, executor=self.executor, key=f"{self.episode}:{self._step}")
        self.stats.actions += 1
        return receipt

    def click(self, control) -> tc.Receipt:
        point = control.point or (control.box[0] + control.box[2] // 2, control.box[1] + control.box[3] // 2)
        return self._do(Click(control.name or control.role, *point))

    def focused(self, control, point) -> bool:
        """Probe: type a character and see whether this field's value changed."""
        before = self.field_text(control)
        self._do(TypeText(control.name, "x", False, replace=False))
        time.sleep(0.12)
        after = self.field_text(control)
        self._do(PressKey("BackSpace"))
        time.sleep(0.08)
        return after is not None and after != before

    def field_text(self, control) -> str | None:
        """Read the field's own value again from the provider (accessibility can, pixels cannot)."""
        scene = self.provider.perceive(self.target())
        for e in scene.elements:
            if e.role == control.role and e.name == control.name and abs(e.box[0] - control.box[0]) < 6 and abs(e.box[1] - control.box[1]) < 6:
                return e.value
        return None

    def fill(self, control, text: str, *, submit: bool = False) -> tc.Receipt:
        clicked = self.click(control)
        if clicked.status == "rejected":
            return tc.Receipt(TypeText(control.name, text, submit), "rejected", error=f"could not click the field: {clicked.error}")
        if not self.focused(control, control.point):
            return tc.Receipt(TypeText(control.name, text, submit), "rejected", error=f"keyboard focus not confirmed in {control.name or control.role!r}")
        return self._do(TypeText(control.name, text, submit))

    def caption(self, text: str) -> None:
        pass
