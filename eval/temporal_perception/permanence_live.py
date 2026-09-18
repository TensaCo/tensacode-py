"""Does an object survive leaving view — and does surviving re-introduce the stale-belief bug?

Snapshot scopes retract what is no longer perceived. That is right for beliefs about now and wrong
for objects, so ``permanence.py`` keeps object files. The danger is that permanence hands back the
bug retraction prevented: asserting an old reading as if it were current. This measures both sides
on the real desktop.

Two readers answer the same questions from the same object files:

    naive   the object is on file, so state its last-seen attribute in the present tense
    dated   ``assertable()`` first; if the object is not in view, give the value *with* its date

and three conditions, each with ground truth we establish ourselves:

    in_view          the thing is on screen                      (both should answer)
    absent_true      out of view, and still the case             (naive right by luck)
    absent_false     out of view, and no longer the case         (naive asserts a falsehood)

``absent_false`` is staged with a second actor: the simulator's own file API deletes a file while
its only on-screen trace has scrolled out of the terminal. Nothing on screen says so, which is
exactly the situation in which a present-tense answer from memory is a lie.

    python -m eval.temporal_perception.permanence_live
"""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

from tensacode.change import Watcher, WINDOW_CLOSED, attribute_windows
from tensacode.permanence import Objects

from .live_harness import SEED, Session

OUT = Path(__file__).resolve().parents[2] / "eval" / "results" / "permanence_live.json"


def api(method: str, path: str, body: dict | None = None) -> dict:
    req = urllib.request.Request(f"{SEED}{path}", method=method,
                                 data=json.dumps(body).encode() if body is not None else None,
                                 headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read() or b"{}")


def put_file(computer: str, path: str, content: str) -> None:
    api("PUT", f"/api/computers/{computer}/file", {"path": path, "content": content})


def remove_file(computer: str, path: str) -> None:
    api("DELETE", f"/api/computers/{computer}/file?path={urllib.parse.quote(path)}")


# --------------------------------------------------------------------- readers


def naive_answer(objects: Objects, needle: str) -> tuple[str, bool]:
    """Permanence without dates: if it is on file, say it is so. Returns (answer, asserted_now)."""
    for file in objects.known():
        if needle in (file.attributes.get("value") or "") or needle in file.label:
            return f"{needle} is on screen (in {file.window or 'the desktop'}).", True
    return f"I have no record of {needle}.", False


def dated_answer(objects: Objects, needle: str) -> tuple[str, bool]:
    """Permanence with dates: never a bare present-tense claim about something out of view."""
    for file in objects.known():
        if needle in (file.attributes.get("value") or "") or needle in file.label:
            if file.assertable("value"):
                return f"{needle} is on screen (in {file.window or 'the desktop'}).", True
            attribute = file.attribute("value")
            when = attribute.seen_at.isoformat(timespec="seconds") if attribute else "?"
            if not file.exists:
                return f"{needle} was in {file.window}, which closed at {when}; it is not there now.", False
            return f"I saw {needle} at {when}; I cannot see it now, so I cannot say it is still there.", False
    return f"I have no record of {needle}.", False


def fresh_answer(objects: Objects, needle: str, seconds: int = 60) -> tuple[str, bool]:
    """The compromise everyone reaches for: trust a reading that is only a minute old."""
    from datetime import timedelta
    for file in objects.known():
        if needle in (file.attributes.get("value") or "") or needle in file.label:
            if file.assertable("value", within=timedelta(seconds=seconds)):
                return f"{needle} is on screen (in {file.window or 'the desktop'}).", True
            return dated_answer(objects, needle)
    return f"I have no record of {needle}.", False


def main() -> None:
    watcher, objects = Watcher(), Objects()
    queries: list[dict] = []
    events: list[dict] = []

    with Session(hostname="permanence-eval") as s:
        computer = s.computer

        def look(tag: str) -> None:
            """Our own eye, our own registry: observe, diff, update object files."""
            items = attribute_windows(s.observe()["items"])  # chrome belongs to the window it sits in
            changes = watcher.see(items, tag=tag)
            report = objects.observe(watcher.latest())
            closed = 0
            for change in changes.of(WINDOW_CLOSED):  # only a close event makes an object cease to exist
                closed += objects.closed(change.window or change.label, at=watcher.latest().at)
            events.append({"tag": tag, "seen": report.seen, "new": report.new, "rematched": report.rematched,
                           "returned": report.returned, "absent": report.absent, "gone": len(objects.gone()),
                           "wrong_about_gone": report.wrong_about_gone, "marked_gone": closed})

        def ask(needle: str, condition: str, truth: bool, note: str) -> None:
            naive, naive_asserts = naive_answer(objects, needle)
            dated, dated_asserts = dated_answer(objects, needle)
            fresh, fresh_asserts = fresh_answer(objects, needle)
            queries.append({"needle": needle, "condition": condition, "true_now": truth, "note": note,
                            "naive": naive, "naive_asserted_present_tense": naive_asserts,
                            "dated": dated, "dated_asserted_present_tense": dated_asserts,
                            "fresh": fresh, "fresh_asserted_present_tense": fresh_asserts,
                            "fresh_wrong": fresh_asserts and not truth,
                            "naive_wrong": naive_asserts and not truth,
                            "dated_wrong": dated_asserts and not truth,
                            "dated_over_cautious": (not dated_asserts) and truth and condition != "in_view"})
            print(f"  [{condition}] {needle}: naive={'ASSERTS' if naive_asserts else 'holds back'} "
                  f"dated={'ASSERTS' if dated_asserts else 'holds back'} (true_now={truth})")

        print("0. baseline")
        look("baseline")

        print("1. six files we create ourselves, listed into the terminal so the screen carries them")
        kept = [f"perm-kept-{i}.txt" for i in (1, 2, 3)]
        doomed = [f"perm-doomed-{i}.txt" for i in (1, 2, 3)]
        for name in kept + doomed:
            put_file(computer, f"/home/agent/Desktop/{name}", "ground truth\n")
        time.sleep(0.8)
        s.send("list my desktop")
        look("listed")
        for name in kept + doomed:
            ask(name, "in_view", True, "just listed; the terminal shows it")

        print("2. push them out of view (more output); the files are untouched")
        for cmd in ("run `uname -a`", "run `df -h`", "run `ls -la /etc`", "run `ps aux`"):
            s.send(cmd)
        look("scrolled")
        visible = {i.value for i in watcher.latest().items.values() if i.value}
        still_shown = [n for n in kept + doomed if any(n in v for v in visible)]
        print(f"    still on screen after scrolling: {still_shown or 'none'}")

        print("3. a second actor deletes half of them while they are out of view")
        for name in doomed:
            remove_file(computer, f"/home/agent/Desktop/{name}")
        time.sleep(0.8)
        look("deleted-behind-its-back")
        for name in kept:
            ask(name, "in_view" if name in still_shown else "absent_true", True,
                "out of view, never touched: a present-tense answer happens to be true")
        for name in doomed:
            ask(name, "in_view" if name in still_shown else "absent_false", False,
                "deleted through the simulator's file API while out of view; nothing on screen says so")

        print("4. a window is closed: gone, not merely out of view")
        s.send("open the text editor")
        look("editor-open")
        editor_items = [i for i in watcher.latest().items.values() if i.window == "Text Editor"]
        closed_ok = False
        for control in s.ui.observe().controls:
            if (control.name or "").lower() == "close":
                s.ui.click(control)
                closed_ok = True
                break
        time.sleep(0.6)
        look("editor-closed")
        gone_files = objects.gone()
        print(f"    clicked Close: {closed_ok}; object files now gone: {len(gone_files)}")

        print("5. reopen: seeing it beats having written it off")
        s.send("open the text editor")
        look("editor-reopened")

    def rate(condition: str, field: str) -> dict:
        rows = [q for q in queries if q["condition"] == condition]
        n = sum(q[field] for q in rows)
        return {"n": len(rows), "count": n, "rate": round(n / len(rows), 3) if rows else None}

    absent = [q for q in queries if q["condition"].startswith("absent")]
    report = {
        "what": "does permanence survive absence without re-introducing stale beliefs",
        "provenance": {"environment": "Seed simulator (third-party)", "grader": "this script — it created and deleted the file itself and clicked Close itself",
                       "held_out": "n/a — ground truth is what the grader did, known before each question"},
        "queries": queries, "observations": events,
        "naive_stale_assertions": sum(q["naive_wrong"] for q in queries),
        "dated_stale_assertions": sum(q["dated_wrong"] for q in queries),
        "fresh_stale_assertions": sum(q["fresh_wrong"] for q in queries),
        "absent_queries": len(absent),
        "dated_over_caution": sum(q["dated_over_cautious"] for q in queries),
        "by_condition": {
            "in_view": {"naive_asserts": rate("in_view", "naive_asserted_present_tense"),
                        "dated_asserts": rate("in_view", "dated_asserted_present_tense")},
            "absent_true": {"naive_wrong": rate("absent_true", "naive_wrong"),
                            "dated_wrong": rate("absent_true", "dated_wrong"),
                            "dated_over_cautious": rate("absent_true", "dated_over_cautious"),
                            "fresh_answers": rate("absent_true", "fresh_asserted_present_tense")},
            "absent_false": {"naive_wrong": rate("absent_false", "naive_wrong"),
                             "dated_wrong": rate("absent_false", "dated_wrong"),
                             "fresh_wrong": rate("absent_false", "fresh_wrong")},
        },
        "identity": {"rematched_total": sum(e["rematched"] for e in events),
                     "returned_total": sum(e["returned"] for e in events),
                     "marked_gone_total": sum(e["marked_gone"] for e in events),
                     "wrong_about_gone_total": sum(e["wrong_about_gone"] for e in events)},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=1))
    print(f"\nnaive stale assertions {report['naive_stale_assertions']} · dated {report['dated_stale_assertions']} "
          f"· fresh-within-60s {report['fresh_stale_assertions']} "
          f"· absent queries {report['absent_queries']} · dated over-caution {report['dated_over_caution']}")
    print(f"identity: {report['identity']}")
    print(f"written to {OUT}")


if __name__ == "__main__":
    main()
