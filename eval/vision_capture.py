"""Capture paired (screenshot, DOM scene) frames from Seed computers and the demo web apps.

    PYTHONPATH=src:. python eval/vision_capture.py --out DIR [--computers ubuntu-3,macos-2,windows-2]

The DOM scene (``Browser.observe``) is ground truth for the pixel perceiver. A frame is
kept only if two DOM reads taken just before and just after the screenshot agree, so
the pair describes the same instant. Each frame records the app just opened and a split:
``tune`` (Ubuntu Files/Terminal/Mail + web apps access/shop), ``test_app`` (other Ubuntu
apps + web apps recon/inbox/chart), ``test_os`` (macOS and Windows computers).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import random
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).parents[1] / "src"), str(Path(__file__).parents[1])]

from examples.browser_agents import harness  # noqa: E402
from examples.browser_agents.browser import Browser, Screen  # noqa: E402

SEED = "http://127.0.0.1:4391"

# Pixel ground truth the DOM scene lacks: words that are actually visible (hit-tested at
# their own location, so text under another window does not count) and window frames.
GROUND_TRUTH_JS = r"""
() => {
  const words = [];
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  const range = document.createRange();
  for (let n = walker.nextNode(); n; n = walker.nextNode()) {
    const text = n.nodeValue; if (!text || !text.trim()) continue;
    const el = n.parentElement; if (!el) continue;
    const cs = getComputedStyle(el); if (cs.visibility === 'hidden' || +cs.opacity === 0) continue;
    const re = /\S+/g; let m;
    while ((m = re.exec(text))) {
      range.setStart(n, m.index); range.setEnd(n, m.index + m[0].length);
      for (const r of range.getClientRects()) {
        if (r.width < 2 || r.height < 5 || r.right <= 0 || r.bottom <= 0 || r.left >= innerWidth || r.top >= innerHeight) continue;
        const top = document.elementFromPoint(Math.min(innerWidth - 1, Math.max(0, r.left + r.width / 2)), Math.min(innerHeight - 1, Math.max(0, r.top + r.height / 2)));
        if (top && (top === el || el.contains(top) || top.contains(el))) words.push({ text: m[0], box: [Math.round(r.x), Math.round(r.y), Math.round(r.width), Math.round(r.height)], size: parseFloat(cs.fontSize) });
      }
    }
  }
  const windows = [...document.querySelectorAll('article.app-window')].map((w) => {
    const r = w.getBoundingClientRect(); const head = w.querySelector('header');
    return { box: [Math.round(r.x), Math.round(r.y), Math.round(r.width), Math.round(r.height)], z: +getComputedStyle(w).zIndex || 0,
      focused: w.classList.contains('focused'), title: head ? head.innerText.replace(/\s+/g, ' ').trim().slice(0, 120) : '' };
  });
  return { words, windows };
}
"""
APPS = {
    "ubuntu": ["Files", "Terminal", "Mail", "Text Editor", "Slack", "Settings", "Firefox", "System Monitor", "Visual Studio Code", "App Center", "Chromium", "Rhythmbox", "Wireshark"],
    "macos": ["Finder", "Terminal", "Mail", "Safari", "Messages", "Slack", "Visual Studio Code", "App Store", "Settings", "Chromium", "ChatGPT"],
    "windows": ["File Explorer", "Terminal", "Outlook", "Microsoft Edge", "Microsoft Teams", "Slack", "Visual Studio Code", "Microsoft Store", "Settings", "Chromium"],
}
TUNE_APPS = {"Files", "Terminal", "Mail", "base"}
TUNE_WEB = {"access", "shop"}


def signature(s: Screen) -> tuple:
    return (tuple((c.role, c.name, c.box) for c in s.controls), tuple((t.text, t.box) for t in s.texts))


def capture(ui: Browser, out: Path, meta: dict, n: list[int]) -> bool:
    page = ui.page
    for _ in range(4):
        before = ui.observe()
        truth = page.evaluate(GROUND_TRUTH_JS)
        png = page.screenshot(type="png")
        after = ui.observe()
        if signature(before) == signature(after) and truth == page.evaluate(GROUND_TRUTH_JS):
            name = f"f{n[0]:04d}"
            (out / f"{name}.png").write_bytes(png)
            (out / f"{name}.json").write_text(json.dumps({"meta": meta, "screen": dataclasses.asdict(after), **truth}))
            n[0] += 1
            return True
        page.wait_for_timeout(350)
    return False


def split_for(os_name: str, app: str) -> str:
    if os_name == "web":
        return "tune" if app.split(":")[0] in TUNE_WEB else "test_app"
    if os_name != "ubuntu":
        return "test_os"
    return "tune" if app in TUNE_APPS else "test_app"


def drag_front_window(ui: Browser, title: str, rng: random.Random) -> bool:
    s = ui.observe()
    bars = [t for t in s.texts if t.text == title and t.section == title]
    if not bars:
        return False
    x, y, w, h = bars[0].box
    cx, cy = x + w // 2, y + h // 2
    page = ui.page
    page.mouse.move(cx, cy)
    page.mouse.down()
    page.mouse.move(cx + rng.randint(-260, 200), cy + rng.randint(-40, 180), steps=8)
    page.mouse.up()
    page.wait_for_timeout(400)
    return True


def run_computer(browser, computer: str, os_name: str, out: Path, n: list[int], rng: random.Random) -> None:
    context = browser.new_context(viewport={"width": 1280, "height": 800}, device_scale_factor=1)
    page = context.new_page()
    page.goto(f"{SEED}/?computer={computer}")
    page.wait_for_timeout(2500)
    ui = Browser(page, episode=f"capture-{computer}")
    capture(ui, out, {"computer": computer, "os": os_name, "app": "base", "split": split_for(os_name, "base")}, n)
    opened = 0
    for app in APPS[os_name]:
        s = ui.observe()
        launch = [c for c in s.controls if c.name == app and c.role == "button" and c.point]
        if not launch:
            continue
        ui.click(launch[-1])
        page.wait_for_timeout(1100)
        meta = {"computer": computer, "os": os_name, "app": app, "split": split_for(os_name, app)}
        capture(ui, out, meta, n)
        if app == "Terminal":
            box = [c for c in ui.observe().controls if c.role == "textbox" and c.point and ("Shell" in c.name or "terminal" in c.name.lower() or "command" in c.name.lower())]
            for cmd in ("ls -la", "echo hello world && date"):
                if box:
                    ui.fill(box[0], cmd, submit=True)
                    page.wait_for_timeout(700)
            capture(ui, out, {**meta, "app": app, "state": "commands"}, n)
        if app in ("Mail", "Outlook"):
            new = [c for c in ui.observe().controls if c.name in ("New Message", "New mail") and c.point]
            if new:
                ui.click(new[0])
                page.wait_for_timeout(700)
                capture(ui, out, {**meta, "state": "composer"}, n)
        opened += 1
        if opened % 3 == 0 and drag_front_window(ui, app, rng):
            capture(ui, out, {**meta, "state": "moved"}, n)
        if opened % 4 == 0:
            page.reload()
            page.wait_for_timeout(2500)
    context.close()


def run_web(browser, out: Path, n: list[int]) -> None:
    base, server = harness.serve()
    context = browser.new_context(viewport={"width": 1100, "height": 760}, device_scale_factor=1)
    harness.block_outside_network(context, base)
    page = context.new_page()
    ui = Browser(page, episode="capture-web")
    for name in ("access", "shop", "recon", "inbox", "chart"):
        for seed in (1, 2):
            page.goto(f"{base}/{name}.html?seed={seed}")
            page.wait_for_timeout(900)
            meta = {"computer": "web", "os": "web", "app": f"{name}:{seed}", "split": split_for("web", name)}
            capture(ui, out, meta, n)
            buttons = [c for c in ui.observe().controls if c.role in ("button", "tab") and c.point]
            if buttons:
                ui.click(buttons[min(1, len(buttons) - 1)])
                page.wait_for_timeout(500)
                capture(ui, out, {**meta, "state": "clicked"}, n)
    context.close()
    server.shutdown()


def main() -> None:
    from playwright.sync_api import sync_playwright

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--computers", default="ubuntu-3:ubuntu,macos-2:macos,windows-2:windows")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    n, rng = [0], random.Random(7)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        for spec in args.computers.split(","):
            computer, os_name = spec.split(":")
            run_computer(browser, computer, os_name, args.out, n, rng)
            print(computer, "frames so far", n[0], flush=True)
        run_web(browser, args.out, n)
        browser.close()
    print("frames", n[0])


if __name__ == "__main__":
    main()
