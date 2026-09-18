"""Run the desktop chore agent (tasks/desktop.py) seeing only pixels.

``PixelBrowser`` replaces DOM perception with ``perceive`` on a screenshot. Actions are
unchanged (mouse clicks at points, key presses). The agent program is unchanged; two
thin, explicit adapters sit between pixels and the task's vocabulary:

* efference copy: the body knows what it just typed, so an OCR'd prompt line that is a
  near-match (similarity >= 0.8) for a typed command is read as that command, and the
  prompt's cwd is normalized. Output text gets two OCR-confusion corrections only:
  curly/back quotes -> straight quotes, and "-/" at a word start -> "~/" (OCR reads "~"
  as "-"). Anything else OCR garbles in the note makes parsing fail, and the agent says so.
* vocabulary: the task looks for a textbox labeled "Shell input" and a dock button
  "Terminal". Vision names the prompt line a textbox with hint "command prompt" in the
  window titled "Terminal"; the adapter renames that one control. The dock icon's name
  comes from ``IconMemory`` (fit from labeled examples of the same icon theme).

The runnable pixel end-to-end lives in ``cw_e2e.py`` (on the computerworld engine); this
module keeps the adapters it shares.
"""

from __future__ import annotations

import dataclasses
import difflib
import io
import re
import time

from typing import Sequence

import numpy as np

from ..browser import Browser, Control, PressKey, Screen, Text, TypeText
from .icon_memory import IconMemory
from .perceive import perceive

SPACED_PROMPT = re.compile(r'^([\w.-]+@[\w.-]+?)\s?[:;.i]?\s?((?:~|["#]|[-_](?=[/$S5\u00a7\s]|$)|/)[\w./~ -]{0,120}?)[$S5\u00a7](?=\s|$)\s*(.*)$')  # "$" present, OCR put a space in the path
LOOSE_PROMPT = re.compile(r'^([\w.-]+@[\w.-]+?)\s?[:;.i]?\s?((?:~|["#]|[-_](?=[/$S5\u00a7\s]|$)|/)[\w./~-]*?)(?:[$S5\u00a7]?(?=\s|$))\s*(.*)$')  # ":" is sometimes read as "i" or dropped
QUOTES = str.maketrans({"‘": "'", "’": "'", "`": "'", "´": "'", "“": '"', "”": '"'})
HOME_PATH = re.compile(r"(?<![\w~.-])[-_\"#]/")  # OCR reads "~" as "-" or "\""; a path starting "-/" or "\"/" does not occur in shell text


def shell_reading(text: str) -> str:
    return HOME_PATH.sub("~/", text.translate(QUOTES))


def expect_typed(line: str, typed: list[str]) -> str:
    """Read a prompt line in light of what was typed (the body's own motor record)."""
    m = SPACED_PROMPT.match(line) or LOOSE_PROMPT.match(line)
    if not m:
        return line
    user, cwd, cmd = m[1], m[2].replace(" ", ""), m[3].strip()
    if cwd.startswith(("-", "_")):
        cwd = "~" + cwd[1:]
    best = max(typed, key=lambda t: difflib.SequenceMatcher(None, t, cmd).ratio(), default=None) if cmd else None
    if best is not None and difflib.SequenceMatcher(None, best, cmd).ratio() >= 0.8:
        cmd = best
    return f"{user}:{cwd}$ {cmd}".rstrip() if cmd else f"{user}:{cwd}$"


def reading_order(texts: tuple[Text, ...]) -> list[Text]:
    """Sort by line, then left to right; phrases on one line can differ by a pixel or two in top edge."""
    by_top = sorted(texts, key=lambda t: (t.section, t.box[1] + t.box[3] / 2))
    lines: list[list[Text]] = []
    for t in by_top:
        mid = t.box[1] + t.box[3] / 2
        last = lines[-1] if lines else None
        if last and last[0].section == t.section and abs(mid - (last[0].box[1] + last[0].box[3] / 2)) < 0.5 * max(1, min(t.box[3], last[0].box[3])):
            last.append(t)
        else:
            lines.append([t])
    return [t for line in lines for t in sorted(line, key=lambda t: t.box[0])]


DIGIT_RUN = re.compile(r"\d{4,}")


def uncertain_digit_runs(texts: Sequence[Text]) -> set[str]:
    """Digit strings that another reading on the same screen contradicts by a single character.

    Measured and off by default: on this task it refuses correct readings far too often,
    because unrelated identifiers legitimately differ by one digit (sequential seeds, ports,
    timestamps). Kept for screens where identifiers do repeat verbatim.

    A task-critical identifier (a project name, a file name) almost always appears in more
    than one place: the listing, the note, the command the agent typed. Two readings that
    differ in one digit cannot both be right, and neither is corroborated, so both are
    refused rather than guessed. A run read identically in two places is corroborated.
    """
    counts: dict[str, int] = {}
    for t in texts:
        for run in DIGIT_RUN.findall(t.text):
            counts[run] = counts.get(run, 0) + 1
    suspect = set()
    runs = list(counts)
    for i, a in enumerate(runs):
        for b in runs[i + 1:]:
            if len(a) == len(b) and sum(x != y for x, y in zip(a, b)) == 1:
                if counts[a] <= counts[b]:
                    suspect.add(a)
                if counts[b] <= counts[a]:
                    suspect.add(b)
    return suspect


def refuse_uncertain(texts: tuple[Text, ...]) -> tuple[Text, ...]:
    """Replace contradicted digit runs with "?" so a parser abstains instead of acting on a guess."""
    suspect = uncertain_digit_runs(texts)
    if not suspect:
        return texts
    out = []
    for t in texts:
        text = t.text
        for run in suspect:
            text = text.replace(run, "?" * len(run))
        out.append(dataclasses.replace(t, text=text) if text != t.text else t)
    return tuple(out)


def mask_low_confidence_digits(scene, threshold: float) -> tuple[Text, ...]:
    """Refuse digit-bearing words the recognizer is unsure about, rather than guessing them.

    A literal a task depends on (an identifier, a file name, an amount) is worth nothing if it
    is wrong, so a word containing digits that the recognizer scores below ``threshold`` is
    replaced by "?" per character. Prompt and command lines are exempt: the body already
    knows what it typed, so those are corroborated by efference copy rather than confidence. A parser then abstains and the agent escalates, instead of
    acting confidently on a misread. Measured on a terminal digit probe (960 words): at 0.95
    this catches 67% of misreads and refuses 6% of correct readings.
    """
    out = []
    for phrase in scene.phrases:
        if SPACED_PROMPT.match(phrase.text) or LOOSE_PROMPT.match(phrase.text):
            out.append(Text("text", phrase.text, next((t.section for t, _ in scene.texts if t.box == phrase.box), ""), phrase.box, 0))
            continue  # a prompt line is checked against what the body typed, not by confidence
        words = []
        for w in phrase.words:
            unsure = any(c.isdigit() for c in w.text) and w.conf < threshold
            words.append("?" * len(w.text) if unsure else w.text)
        text = " ".join(words)
        section = next((t.section for t, _ in scene.texts if t.box == phrase.box), "")
        out.append(Text("text", text, section, phrase.box, 0))
    return tuple(out)


def rejoin_wrapped(texts: tuple[Text, ...], typed: list[str]) -> tuple[Text, ...]:
    """A long typed command wraps onto following lines; read the lines together when that matches what was typed."""
    rows = reading_order(texts)
    out: list[Text] = []
    i = 0
    while i < len(rows):
        t = rows[i]
        m = SPACED_PROMPT.match(t.text) or LOOSE_PROMPT.match(t.text)
        if m and typed:
            cmd, j = m[3].strip(), i
            best = lambda c: max(difflib.SequenceMatcher(None, x, c).ratio() for x in typed) if c else 0.0  # noqa: E731
            same_row = lambda a, b: abs(a.box[1] - b.box[1]) < 0.6 * max(1, a.box[3]) and b.box[0] > a.box[0]  # noqa: E731
            while best(cmd) < 0.999 and j + 1 < len(rows) and rows[j + 1].section == t.section and not (SPACED_PROMPT.match(rows[j + 1].text) or LOOSE_PROMPT.match(rows[j + 1].text)) and (
                    same_row(rows[j], rows[j + 1]) or (cmd and 0 < rows[j + 1].box[1] - rows[j].box[1] < 2.2 * max(1, rows[j].box[3]))):
                longer = cmd + rows[j + 1].text.strip()
                spaced = cmd + " " + rows[j + 1].text.strip()
                cand = max((longer, spaced), key=best)
                if best(cand) <= best(cmd):
                    break
                cmd, j = cand, j + 1
            if j > i:
                merged = dataclasses.replace(t, text=f"{m[1]}:{m[2].replace(' ', '')}$ {cmd}", box=(t.box[0], t.box[1], max(r.box[0] + r.box[2] for r in rows[i:j + 1]) - t.box[0], rows[j].box[1] + rows[j].box[3] - t.box[1]))
                out.append(dataclasses.replace(merged, text=expect_typed(merged.text, typed)))
                i = j + 1
                continue
        out.append(dataclasses.replace(t, text=expect_typed(t.text, typed)))
        i += 1
    return tuple(out)


class PixelBrowser(Browser):
    def __init__(self, page: object, *, episode: str, ocr, detector, icon_memory: IconMemory | None, adapt=None, cross_check: bool = False, refuse_below: float = 0.0) -> None:
        super().__init__(page, episode=episode)
        self.ocr, self.detector, self.icon_memory, self.adapt = ocr, detector, icon_memory, adapt
        self.typed: list[str] = []
        self.focus_checks: list[bool] = []
        self.cross_check, self.refuse_below = cross_check, refuse_below
        self.perception_ms: list[float] = []

    def settled_frame(self, settle_ms: int = 1500, gap_ms: int = 120, changed: float = 0.0005) -> np.ndarray:
        """Wait for the screen to stop changing (two frames ``gap_ms`` apart nearly identical), like waiting out a spinner."""
        from PIL import Image

        grab = lambda: np.asarray(Image.open(io.BytesIO(self.page.screenshot(type="png"))).convert("RGB"))  # noqa: E731
        deadline = time.perf_counter() + settle_ms / 1000
        prev = grab()
        while True:
            self.page.wait_for_timeout(gap_ms)
            cur = grab()
            if (np.abs(cur.astype(np.int16) - prev).sum(-1) > 30).mean() <= changed or time.perf_counter() > deadline:
                return cur
            prev = cur

    def observe(self, *, settle_ms: int = 1500) -> Screen:
        t0 = time.perf_counter()
        rgb = self.settled_frame(settle_ms)
        scene = perceive(rgb, ocr=self.ocr, detector=self.detector, icon_namer=self.icon_memory)
        raw = mask_low_confidence_digits(scene, self.refuse_below) if self.refuse_below else tuple(t for t, _ in scene.texts)
        texts = tuple(dataclasses.replace(t, text=shell_reading(t.text)) for t in raw)
        read = rejoin_wrapped(texts, self.typed)
        screen = dataclasses.replace(scene.screen(), texts=refuse_uncertain(read) if self.cross_check else read)
        if self.adapt:
            screen = self.adapt(screen)
        dt = time.perf_counter() - t0
        self.perception_ms.append(dt * 1e3)
        self.stats.browser_s += dt
        self.stats.observations += 1
        return screen

    def fill(self, control, text: str, *, submit: bool = False):
        receipt = super().fill(control, text, submit=submit)
        if receipt.status != "rejected":
            self.typed.append(text)
        self.focus_checks.append(receipt.status != "rejected")
        return receipt

    def _grab(self) -> np.ndarray:
        from PIL import Image

        return np.asarray(Image.open(io.BytesIO(self.page.screenshot(type="png"))).convert("RGB")).astype(np.int16)

    def focused(self, control: Control, point: tuple[int, int]) -> bool:
        """Pixel body: a probe keystroke must visibly change the field, and nothing else.

        Type one character and see where the screen changed. A change inside the field's box
        means focus is there. The character is removed with Backspace either way, so a keystroke
        that landed in another field is undone. Changes elsewhere are ignored: other windows
        animate on their own (packet counters, clocks).
        """
        x, y, w, h = control.box
        pad = 6
        self.page.wait_for_timeout(60)
        before = self._grab()
        self._do(TypeText(control.name, "x", False, replace=False))
        changed = None
        for _ in range(6):
            self.page.wait_for_timeout(50)
            diff = np.abs(self._grab() - before).sum(-1) > 60
            if diff.any():
                changed = diff
                break
        inside = 0 if changed is None else int(changed[max(0, y - pad):y + h + pad, max(0, x - pad):x + w + pad].sum())
        self._do(PressKey("Backspace"))  # undo the probe wherever it landed (a no-op if nothing had focus)
        self.page.wait_for_timeout(40)
        return inside >= 15  # changes elsewhere are ignored: other windows animate (packet counters, clocks)


def terminal_vocabulary(screen: Screen) -> Screen:
    controls = []
    for c in screen.controls:
        if c.role == "textbox" and c.hint == "command prompt" and c.section == "Terminal":
            c = dataclasses.replace(c, name="Shell input")
        controls.append(c)
    return dataclasses.replace(screen, controls=tuple(controls))


def corroborated(spec):
    """The same mind, but text read from pixels is acted on only once read the same way in 2 frames."""
    from tensorcode.cognition import Corroboration

    return dataclasses.replace(spec, rules=[dataclasses.replace(r, established_only=True) for r in spec.rules],
                               corroboration=lambda: Corroboration(k=2, predicates=frozenset({"reads"})))


def classify_failure(row: dict) -> str | None:
    if row["correct"] == row["items"] and row["status"] == "done":
        return None
    reason = row["reason"] or ""
    if "note unreadable" in reason:
        return "note misread"
    if "keyboard focus" in reason or "could not click the field" in reason:
        return "keyboard focus not confirmed (escalated)"
    if row["status"] == "escalated" and any(i.startswith("waiting for `") for i in row["last_intentions"]):
        return "command completion not recognized" + (" (work was correct)" if row["correct"] == row["items"] else "")
    return f"other: {reason[:80]}"
