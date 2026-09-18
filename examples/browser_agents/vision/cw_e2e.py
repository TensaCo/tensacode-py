"""The desktop chore agent on the computerworld engine, seeing only the engine's pixels.

The same agent program as ``tasks/desktop.py`` runs on two bodies here, and the only thing
that differs is where perception comes from:

* ``CwBody`` + ``CwProvider`` — the engine's own scene. A reference, not a reading.
* ``CwPixelBody`` + ``CwPixelProvider`` — the engine's rendered frame through the visual
  pipeline (docTR text recognition, a UI-element detector, classical CV rules). Nothing the
  engine knows about its own state reaches perception.

Three adapters sit between pixels and this task's vocabulary, and each is a claim about
terminals rather than about this world:

* **Efference copy.** This terminal prints output but never echoes the command that
  produced it (neither does the engine's own scene), so the prompt lines the agent reads
  back are the body's record of what it typed, in front of the output that appeared while
  it ran. Those lines carry ``method="efference-copy"``; every printed line is OCR.
  Output is attributed to a command by how many lines were on screen when it was typed —
  and that count is itself a pixel reading, so a missed line shifts the attribution.
* **Wrapped lines.** A terminal wraps a long line at the right margin. A line whose right
  edge reaches the margin continues on the next one, and whether the join takes a space is
  read off the continuation's indent: a wrap inside a word resumes at the margin, a wrap at
  a space carries that space onto the next row. The reconstruction is scored against the
  engine's own logical lines (reported, never used).
* **Vocabulary.** The task looks for a textbox named ``Shell input`` and a dock button
  ``Terminal``. A console's pane is where typing goes, so the pane becomes that textbox; the
  live prompt inside it is found by its caret (a solid block, found by shape) rather than by
  reading a lone "$", which one frame's recognizer missed. The dock icon carries no text, so
  its name comes from ``IconMemory``, fit on tuning seeds from the structured provider's
  labels for the same icon art.

Run (venv needs computerworld, doctr, opencv):

    PYTHONPATH=src:. python -m examples.browser_agents.vision.cw_e2e --fit-icons icons.npz
    PYTHONPATH=src:. python -m examples.browser_agents.vision.cw_e2e --episodes 10 --icon-memory icons.npz
    PYTHONPATH=src:. python -m examples.browser_agents.vision.cw_e2e --episodes 10 --structured
"""

from __future__ import annotations

import argparse
import dataclasses
import difflib
import json
import re
import statistics
import time
from collections import Counter
from pathlib import Path

import numpy as np

import tensacode as tc

from ..browser import Control, PressKey, Screen, TypeText
from ..perception.computerworld import CwPixelProvider, CwProvider
from ..perception.cw_body import TERMINAL, CwBody, replace_texts
from ..perception.invariants import SameIdentifier, WellFormed, check, mask
from ..perception.protocol import TextBlock, center, inside, own
from .desktop_e2e import shell_reading
from .icon_memory import IconMemory

#: a bare "$" prompt, which is what this terminal draws; OCR reads the glyph as 5, S or § too
PROMPT_LINE = re.compile(r"^[$5S§](\s|$)")
#: the project identifier must be the same in the note and in the file name it was read from
PROJECT_ID = SameIdentifier("project id", r"[a-z]+-[a-z]+-(\d{4,})", r"task-(\d{4,})\.txt")
#: and a project name has a shape, which is what catches a misreading nothing contradicts
PROJECT_NAME = WellFormed("project name", r"(?:project called|project please:)\s+([^\s(]+)", r"[a-z]+-[a-z]+-\d{4,}")
INVARIANTS = (PROJECT_ID, PROJECT_NAME)
REFUSED = "<refused>"


# ------------------------------------------------------------------ reading a terminal from pixels


def terminal_box(scene) -> tuple[int, int, int, int] | None:
    windows = [r for r in scene.regions if r.kind == "window" and (r.label or "").lower().startswith("terminal")]
    return max(windows, key=lambda r: r.box[2] * r.box[3]).box if windows else None


def caret_in(rgb: np.ndarray, box: tuple[int, int, int, int], *, fill: float = 0.85) -> tuple[int, int, int, int] | None:
    """The text caret inside a window: a solid bright block on the character grid.

    A lone "$" prompt glyph is small, coloured and easy for a recognizer to miss on one
    frame; the caret next to it is a filled rectangle, so it is found by shape rather than
    by reading. Glyphs are rejected because they are not solid (a letter fills about half
    its box) and window chrome because of its size. The lowest match is the live prompt.
    """
    import cv2

    x, y, w, h = box
    crop = rgb[max(0, y):y + h, max(0, x):x + w]
    if crop.size == 0:
        return None
    lum = crop.astype(np.int16).sum(-1) / 3
    mask = (lum > float(np.median(lum)) + 60).astype(np.uint8)
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, 8)
    best = None
    for i in range(1, count):
        bx, by, bw, bh, area = stats[i]
        if not (4 <= bw <= 18 and 9 <= bh <= 30) or area < fill * bw * bh:
            continue
        if best is None or by > best[1]:
            best = (bx, by, bw, bh)
    return (x + best[0], y + best[1], best[2], best[3]) if best else None


def pane_lines(texts: tuple[TextBlock, ...], box: tuple[int, int, int, int], tol: int = 3) -> list[TextBlock]:
    """The text drawn in a terminal's output pane, separated from the window's own chrome.

    Output lines share a left margin (a character grid); a window's title and tab labels do
    not sit on it. So the margin is the leftmost text edge inside the window, and the pane
    starts at the first line standing on it. Everything above that — the title bar, a tab
    strip — is chrome. Indented output stays, because only the first aligned line is the cut.
    """
    inner = sorted((t for t in texts if inside(center(t.box), box)), key=lambda t: (t.box[1], t.box[0]))
    if not inner:
        return []
    margin = min(t.box[0] for t in inner)
    aligned = [t for t in inner if abs(t.box[0] - margin) <= tol]
    if not aligned:
        return []
    top = min(t.box[1] for t in aligned)
    return [t for t in inner if t.box[1] >= top - tol]


def pane_box(lines: list[TextBlock], window: tuple[int, int, int, int], caret: tuple[int, int, int, int] | None):
    """Where typing goes in a console: the output pane, not the one prompt line.

    Clicking anywhere in a terminal's pane puts the keyboard in the shell, and the prompt
    moves down the pane as output arrives, so the pane is the stable target. Its top is the
    first line of text (or the caret, on an empty terminal) and it runs to the window's
    inside edges.
    """
    tops = [t.box[1] for t in lines] + ([caret[1]] if caret else [])
    lefts = [t.box[0] for t in lines] + ([caret[0]] if caret else [])
    if not tops:
        return None
    top, left = min(tops) - 4, min(lefts) - 6
    return (left, top, window[0] + window[2] - 8 - left, window[1] + window[3] - 8 - top)


def wrapped_together(lines: list[TextBlock], right: int, slack: float = 1.2) -> list[str]:
    """Join lines the terminal wrapped: a line whose right edge reaches the margin continues below.

    ``slack`` is in characters, measured from the line's own glyph pitch, because the wrap
    lands within one character of the margin and OCR boxes are a pixel or two short.

    Whether the join takes a space is decided by where the continuation *starts*. A wrap
    inside a word continues at the left margin, so the pieces are glued ("notes/ w" + "ith").
    A wrap at a space carries that space onto the next row, which is visible as a
    one-character indent, so the pieces are joined with a space ("as 'initial" + "commit'.").
    Measured on the engine's own logical lines: guessing "no space" fuses words and produced
    a commit message of ``initialcommit`` that the agent then confirmed against its own
    misreading.
    """
    out: list[str] = []
    margin = min((line.box[0] for line in lines), default=0)
    open_line = False
    for line in lines:
        pitch = line.box[2] / max(1, len(line.text))
        full = line.box[0] + line.box[2] + slack * pitch >= right
        if open_line and out:
            out[-1] = out[-1] + (" " if line.box[0] - margin > 0.5 * pitch else "") + line.text
        else:
            out.append(line.text)
        open_line = full
    return out


class CwPixelBody(CwBody):
    """A computerworld body whose perception is the rendered frame and nothing else."""

    def __init__(self, surface, provider, *, episode: str = "cwpx", invariants: tuple = (), adapt=None) -> None:
        super().__init__(surface, provider, episode=episode)
        self.typed: list[str] = []  # commands this body submitted, in order
        self.before: list[int] = []  # printed lines read on screen when each was submitted
        self.printed: list[str] = []  # the last pixel reading of the terminal's printed lines
        self.prompt: str | None = None  # what is typed at the live prompt, when one is on screen
        self.focus_checks: list[bool] = []
        self.invariants, self.adapt = invariants, adapt
        self.refusals: list[str] = []
        self.line_scores: list[tuple[int, int]] = []  # (lines read exactly right, lines on screen), for reporting only
        self.window: tuple[int, int, int, int] | None = None  # the terminal window, remembered between frames
        self.pane: tuple[int, int, int, int] | None = None  # where typing goes, remembered the same way

    # -- perception

    def observe(self, *, settle_ms: int = 0) -> Screen:
        t0 = time.perf_counter()
        rgb = self.surface.picture()
        scene = self.provider.perceive(self.target())
        self.last_scene = scene
        texts = tuple(dataclasses.replace(t, text=shell_reading(t.text)) for t in scene.texts)
        box = self.terminal_window_box(scene, texts)
        if box is None:
            self.printed, self.prompt, self.pane = [], None, None
            self.last_read = scene = replace_texts(scene, texts)
        else:
            lines = pane_lines(texts, box)
            caret = caret_in(rgb, box)
            live = [t for t in lines if caret is not None and abs(center(t.box)[1] - center(caret)[1]) <= 0.6 * caret[3]]
            if caret is None and lines and PROMPT_LINE.match(lines[-1].text):
                live = lines[-1:]  # no caret found: fall back to reading the prompt glyph itself
            printed = [t for t in lines if t not in live and (caret is None or center(t.box)[1] < center(caret)[1])]
            self.prompt = PROMPT_LINE.sub("", " ".join(t.text for t in live)).strip() if (live or caret) else None
            self.printed = self.refuse(wrapped_together(printed, box[0] + box[2] - 8))
            self.pane = pane_box(printed or live, box, caret)
            self.line_scores.append(self.reading_error())
            read = {id(t) for t in (*printed, *live)}
            keep = tuple(t for t in texts if id(t) not in read)
            self.last_read = scene = replace_texts(scene, keep + tuple(self.transcript()))
        screen = self.vocabulary(scene.to_screen())
        dt = time.perf_counter() - t0
        self.perception_ms.append(dt * 1e3)
        self.stats.browser_s += dt
        self.stats.observations += 1
        return screen

    def terminal_window_box(self, scene, texts) -> tuple[int, int, int, int] | None:
        """The terminal window, and object permanence for it.

        Window detection works from frame edges and a title strip, and on a busy frame it
        can title a window after the first line of text inside it. A terminal does not move
        or rename itself between two frames, so the last box is kept as long as the screen
        still shows text inside it, and dropped when it does not.
        """
        box = terminal_box(scene)
        if box is not None:
            self.window = box
            return box
        if self.window is not None and any(inside(center(t.box), self.window) for t in texts):
            return self.window
        self.window = None
        return None

    def reading_error(self) -> tuple[int, int]:
        """How much of the terminal this reading got exactly right, against the engine's own lines.

        Measured, never used: the body compares its reconstruction (OCR plus the wrap rule)
        with ``terminal_lines()``, which is the engine's state. Lines are aligned before they
        are counted, so one line joined wrongly costs one line and does not throw off every
        line after it. Reported per episode, so the end-to-end numbers can be read next to
        how well the screen was actually read.
        """
        truth = [line.strip() for line in self.surface.terminal_lines()]
        read = [line.strip() for line in self.printed]
        matched = sum(block.size for block in difflib.SequenceMatcher(None, read, truth).get_matching_blocks())
        return (matched, len(truth))

    def refuse(self, printed: list[str]) -> list[str]:
        """Take out what the task's own invariants say cannot be a correct reading."""
        if not self.invariants:
            return printed
        verdict = check(printed, self.invariants)
        if not verdict.violations:
            return printed
        note = f"{verdict}"
        if note not in self.refusals:
            self.refusals.append(note)
        return mask(printed, self.invariants, verdict, REFUSED)

    def transcript(self) -> list[TextBlock]:
        """Prompt lines from this body's motor record, output lines from pixels, in screen order."""
        typed = own("computerworld-pixels", "efference-copy", "command typed by this body")
        printed = own("computerworld-pixels", "ocr", "terminal output read from the rendered frame")
        drawn = own("computerworld-pixels", "caret+ocr", "the live prompt, found by its caret")
        blocks: list[TextBlock] = []
        y = [0]

        def block(text: str, provenance) -> TextBlock:
            y[0] += 18
            blocks.append(TextBlock(text=text, box=(0, y[0], 600, 16), section=TERMINAL, provenance=provenance))

        head = self.before[0] if self.before else len(self.printed)
        for line in self.printed[:head]:
            block(line, printed)
        for index, command in enumerate(self.typed):
            upto = self.before[index + 1] if index + 1 < len(self.before) else len(self.printed)
            block(f"{self.surface.user}@{self.surface.machine}:$ {command}", typed)
            for line in self.printed[self.before[index]:upto]:
                block(line, printed)
        if self.prompt is not None:
            block(f"{self.surface.user}@{self.surface.machine}:$ {self.prompt}".rstrip(), drawn)
        return blocks

    def vocabulary(self, screen: Screen) -> Screen:
        """This task's names for what vision found: the console pane is the ``Shell input``."""
        controls = [c for c in screen.controls if not (c.role == "textbox" and c.hint == "command prompt")]
        if self.pane is not None:
            controls.insert(0, Control(role="textbox", name="Shell input", value=self.prompt or "", checked=None, disabled=False,
                                       hint="command prompt", group="", section=TERMINAL, input_type="",
                                       box=self.pane, point=center(self.pane), current=False, shown=""))
        screen = dataclasses.replace(screen, controls=tuple(controls))
        return self.adapt(screen) if self.adapt else screen

    # -- action

    def fill(self, control, text: str, *, submit: bool = False) -> tc.Receipt:
        """Click the field, confirm the keyboard reaches it in pixels, clear it, then type."""
        clicked = self.click(control)
        if clicked.status != "applied":
            return tc.Receipt(TypeText(control.name, text, submit), "rejected", error=f"could not click the field: {clicked.error}")
        if not self.focused(control, control.point):
            self.focus_checks.append(False)
            return tc.Receipt(TypeText(control.name, text, submit), "rejected",
                              error=f"keyboard focus not confirmed in {control.name or control.role!r}")
        for _ in range(min(len(control.value or ""), 256)):  # whatever is already typed, as read from pixels
            self._do(PressKey("Backspace"))
        before = len(self.printed)
        receipt = self._do(TypeText(control.name, text, submit, replace=False))
        self.focus_checks.append(receipt.status == "applied")
        if receipt.status == "applied" and submit:
            self.typed.append(text)
            self.before.append(before)
        return receipt

    def focused(self, control, point) -> bool:
        """Probe: type one character and see whether the field's own pixels changed."""
        before = self.surface.picture().astype(np.int16)
        self._do(TypeText(control.name, "x", False, replace=False))
        changed = np.abs(self.surface.picture().astype(np.int16) - before).sum(-1) > 60
        x, y, w, h = control.box
        pad = 10
        inside_box = int(changed[max(0, y - pad):y + h + pad, max(0, x - pad):x + w + pad].sum())
        self._do(PressKey("Backspace"))  # undo the probe wherever it landed
        return inside_box >= 15

    def field_text(self, control) -> str | None:
        return self.prompt


# ------------------------------------------------------------------ the dock's icon names


def fit_icons(seeds: list[int], memory: IconMemory, *, ocr=None, detector=None) -> dict:
    """Name the dock's icon art from the structured provider's labels for the same boxes.

    Free labels: the engine names its own dock buttons, so a tuning seed teaches the icon
    memory what a terminal glyph looks like. The evaluation seeds are different worlds and
    the memory is never refit during them.
    """
    from ..perception.visual import VisionProvider
    from ..perception.protocol import Target
    from ..worlds import note_world
    from ..worlds.runtime import CwWorld
    from ..tasks import desktop as task_module

    vision = VisionProvider(ocr=ocr, detector=detector)
    learned, seen = 0, 0
    for seed in seeds:
        world = CwWorld(note_world(task_module.chore(seed)["note"], seed), seed)
        surface = world.actor()
        rgb = surface.picture()
        named = CwProvider().perceive(Target(detail={"surface": surface})).elements
        pixels = vision.perceive(Target(grab=surface.picture))
        for element in pixels.elements:
            if element.name or element.role != "button":
                continue
            seen += 1
            match = next((e for e in named if e.name and inside(e.point, element.box) and e.box[2] < 120), None)
            if match is not None:
                learned += memory.remember(rgb, element.box, match.name)
    return {"tuning_seeds": seeds, "unnamed_icons_seen": seen, "examples_kept": learned, "names": sorted(set(memory.names))}


# ------------------------------------------------------------------ episodes


def classify_failure(row: dict) -> str | None:
    if row["status"] == "done" and row["correct"] == row["items"]:
        return None
    reason = row["reason"] or ""
    if "note unreadable" in reason:
        return "note misread (escalated)" if not row["refusals"] else "note refused by invariant (escalated)"
    if "no task note" in reason:
        return "no task note found (escalated)"
    if "keyboard focus" in reason or "could not click the field" in reason:
        return "keyboard focus not confirmed (escalated)"
    if row["status"] == "escalated" and any(i.startswith("waiting for `") for i in row["last_intentions"]):
        return "command completion not recognized" + (" (work was correct)" if row["correct"] == row["items"] else "")
    if row["status"] == "done" and row["correct"] != row["items"]:
        return "FALSE SUCCESS: claimed verified, files wrong"
    return f"other: {reason[:80]}"


def run(seed: int, *, structured: bool, ocr=None, detector=None, memory=None, invariants=(), spec=None, task_module=None) -> dict:
    from ..mind import run_mind
    from ..worlds import note_world
    from ..worlds.runtime import CwWorld
    from .. import harness

    task = harness.tasks(["desktop"])["desktop"]
    world = CwWorld(note_world(task_module.chore(seed)["note"], seed), seed, width=1280, height=800)
    if structured:
        ui = CwBody(world.actor(), CwProvider(), episode=f"cw-{seed}")
    else:
        from ..perception.visual import VisionProvider

        provider = CwPixelProvider(VisionProvider(ocr=ocr, detector=detector, icon_namer=memory))
        ui = CwPixelBody(world.actor(), provider, episode=f"cwpx-{seed}", invariants=invariants)
    runtime = harness.runtime_for(task)
    cycles: list[str] = []
    t0 = time.perf_counter()
    error, outcome = None, None
    with tc.use(runtime):
        try:
            outcome = run_mind(ui, spec or task.spec, on_cycle=lambda m, t, i: cycles.append(getattr(i, "why", None) or getattr(i, "reason", "")))
        except Exception as exc:  # noqa: BLE001 - an agent crash is a scored failure
            error = f"{type(exc).__name__}: {exc}"
    seconds = time.perf_counter() - t0
    score = task_module.score(world, seed)
    scores = getattr(ui, "line_scores", [])
    row = {
        "seed": seed, "correct": score["correct"], "items": score["items"],
        "status": getattr(outcome, "status", "error"), "reason": getattr(outcome, "reason", None) or error,
        "seconds": round(seconds, 2), "cycles": len(cycles), "actions": ui.stats.actions,
        "perception_ms_p50": round(statistics.median(ui.perception_ms), 1) if ui.perception_ms else None,
        "last_intentions": cycles[-4:], "checks": score.get("checks", {}),
        "typing_attempts": len(getattr(ui, "focus_checks", [])), "typing_rejected": getattr(ui, "focus_checks", []).count(False),
        "refusals": getattr(ui, "refusals", []),
        "lines_exact_vs_engine": list(scores[-1]) if scores else None,  # (lines read exactly right, lines on screen)
        "observes_with_a_misread_line": sum(1 for matched, total in scores if matched != total),
        "state_hash": score.get("state_hash"),
    }
    row["failure_class"] = classify_failure(row)
    return row


def main() -> None:
    from ..tasks import desktop as task_module

    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--first-seed", type=int, default=41001)
    ap.add_argument("--seeds", default=None, help="explicit comma-separated seeds, so every arm runs the same worlds")
    ap.add_argument("--structured", action="store_true", help="reference arm: the engine's own scene instead of its pixels")
    ap.add_argument("--reco-weights", default=None, help="fine-tuned text recognizer")
    ap.add_argument("--icon-memory", type=Path, default=None)
    ap.add_argument("--invariant", action="store_true", help="refuse a project id the note and the file name disagree about")
    ap.add_argument("--fit-icons", type=Path, default=None, help="fit an icon memory from tuning seeds and exit")
    ap.add_argument("--tuning-seeds", type=int, default=3)
    ap.add_argument("--label", default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--shots", type=Path, default=None)
    args = ap.parse_args()

    ocr = det = memory = None
    if not args.structured:
        from .models import DoctrOCR, IconDetector

        ocr, det = DoctrOCR(reco_weights=args.reco_weights), IconDetector()
        ocr.load()
        det.load()
    if args.fit_icons:
        memory = IconMemory()
        report = fit_icons([90001 + i for i in range(args.tuning_seeds)], memory, ocr=ocr, detector=det)
        memory.save(args.fit_icons)
        print(json.dumps(report))
        return
    if args.icon_memory:
        memory = IconMemory.load(args.icon_memory)

    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else [args.first_seed + k for k in range(args.episodes)]
    rows = []
    for seed in seeds:
        row = run(seed, structured=args.structured, ocr=ocr, detector=det, memory=memory,
                  invariants=INVARIANTS if args.invariant else (), task_module=task_module)
        rows.append(row)
        print(json.dumps(row), flush=True)
    mode = args.label or ("structured" if args.structured else "pixels" + (f"+{Path(args.reco_weights).stem}" if args.reco_weights else "") + ("+invariant" if args.invariant else ""))
    summary = {
        "mode": mode, "engine": "computerworld", "episodes": len(rows),
        "items_correct": sum(r["correct"] for r in rows), "items": sum(r["items"] for r in rows),
        "verified": sum(1 for r in rows if r["status"] == "done"),
        "verified_and_correct": sum(1 for r in rows if r["status"] == "done" and r["correct"] == r["items"]),
        "false_successes": [r["seed"] for r in rows if r["status"] == "done" and r["correct"] != r["items"]],
        "episodes_fully_correct": sum(1 for r in rows if r["correct"] == r["items"]),
        "refused": [r["seed"] for r in rows if r["refusals"]],
        "episodes_with_a_misread_line": sum(1 for r in rows if r["observes_with_a_misread_line"]),
        "lines_read_exactly": sum(r["lines_exact_vs_engine"][0] for r in rows if r["lines_exact_vs_engine"]),
        "lines_on_screen": sum(r["lines_exact_vs_engine"][1] for r in rows if r["lines_exact_vs_engine"]),
        "seconds_per_episode_p50": statistics.median(r["seconds"] for r in rows),
        "perception_ms_p50": statistics.median([r["perception_ms_p50"] for r in rows if r["perception_ms_p50"]] or [0]),
        "failure_classes": dict(Counter(r["failure_class"] for r in rows if r["failure_class"])),
        "rows": rows,
    }
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}))
    if args.out:
        args.out.write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
