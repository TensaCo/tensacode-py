"""Does the live assistant know what changed since your last message?

Ground truth comes from us, not from the system: every trial either commands a change (open an
app) or makes one behind its back through the simulator's own API (write/delete a file the open
Files window is showing), and one trial changes nothing at all. That last one is the control that
measures noise — a live desktop has a clock in it, so "what changed" is only useful if it can
ignore what moves on its own.

    python -m eval.temporal_perception.change_live [--assistant http://127.0.0.1:8770]

Scoring, per trial: did the reply name the thing we changed (hit), and how many steady changes did
it report that we did not cause (false changes). Volatile changes are counted separately, because
reporting them would be the failure mode.
"""

from __future__ import annotations

import argparse
import json
import re
import socket
import threading
import time
import urllib.request
from pathlib import Path

SEED = "http://127.0.0.1:4391"
OUT = Path(__file__).resolve().parents[2] / "eval" / "results" / "change_live.json"


class Chat:
    """Drive the live assistant over its own HTTP interface and read its replies."""

    def __init__(self, base: str) -> None:
        self.base = base.rstrip("/")
        self.events: list[dict] = []
        threading.Thread(target=self._listen, daemon=True).start()
        time.sleep(1.5)

    def _listen(self) -> None:
        host, port = self.base.split("//")[1].split(":")
        sock = socket.create_connection((host, int(port)), timeout=3600)
        sock.sendall(b"GET /events HTTP/1.1\r\nHost: localhost\r\nAccept: text/event-stream\r\n\r\n")
        buf = b""
        sock.settimeout(3600)
        while True:
            try:
                chunk = sock.recv(65536)
            except Exception:
                return
            if not chunk:
                return
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                if line.startswith(b"data: "):
                    try:
                        self.events.append(json.loads(line[6:]))
                    except Exception:
                        pass

    def say(self, text: str, timeout: float = 180.0) -> list[str]:
        mark = len(self.events)
        urllib.request.urlopen(urllib.request.Request(f"{self.base}/say", data=json.dumps({"text": text}).encode(),
                                                      method="POST"), timeout=20).read()
        deadline = time.time() + timeout
        while time.time() < deadline:
            time.sleep(0.4)
            tail = self.events[mark:]
            if any(e.get("type") == "busy" and e.get("busy") is False for e in tail):
                break
        time.sleep(0.4)
        return [e["text"] for e in self.events[mark:] if e.get("type") == "chat" and e.get("from") == "agent"]


def api(method: str, path: str, body: dict | None = None) -> dict:
    req = urllib.request.Request(f"{SEED}{path}", method=method,
                                 data=json.dumps(body).encode() if body is not None else None,
                                 headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read() or b"{}")


def write_file(computer: str, path: str, content: str) -> None:
    api("PUT", f"/api/computers/{computer}/file", {"path": path, "content": content})


def delete_file(computer: str, path: str) -> None:
    try:
        api("DELETE", f"/api/computers/{computer}/file?path={path}")
    except Exception:  # noqa: BLE001 - nothing to clean
        pass


def count_reported(reply: str) -> int:
    """How many changes the reply actually listed."""
    if "Nothing changed" in reply or "nothing changed" in reply:
        return 0
    m = re.search(r"^(\d+) things? changed", reply)
    if m:
        return int(m[1])
    return len([ln for ln in reply.splitlines() if ln.strip().startswith(("- ", "• ")) or " appeared" in ln or " went away" in ln])


def ignored_volatile(reply: str) -> int:
    m = re.search(r"ignored (\d+) thing", reply)
    return int(m[1]) if m else 0


ASK = "what changed since my last message"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assistant", default="http://127.0.0.1:8770")
    ap.add_argument("--computer", default="ubuntu-2")
    args = ap.parse_args()
    chat = Chat(args.assistant)
    trials: list[dict] = []

    def trial(name: str, setup, expect: str | None, note: str) -> None:
        """One trial: a boundary message, our intervention, then the question."""
        chat.say("hello")  # boundary: the snapshot "since my last message" refers to
        caused = setup()
        replies = chat.say(ASK)
        reply = "\n".join(replies)
        hit = bool(expect) and expect.lower() in reply.lower()
        reported = count_reported(reply)
        false_changes = max(0, reported - (1 if hit else 0))
        trials.append({"trial": name, "intervention": note, "expected": expect, "hit": hit,
                       "reported": reported, "false_changes": false_changes,
                       "volatile_ignored": ignored_volatile(reply), "reply": reply[:400], "caused": caused})
        print(f"  {name}: hit={hit} reported={reported} false={false_changes} volatile_ignored={ignored_volatile(reply)}")
        print(f"    {reply[:200]}")

    print("1. control: nothing changed")
    trial("nothing_changed", lambda: "nothing", None, "no intervention at all")

    print("2. a file written behind its back, in a folder no open window is showing")
    path = "/home/agent/Desktop/chg-appeared.txt"

    def make_file() -> str:
        write_file(args.computer, path, "ground truth\n")
        time.sleep(1.5)
        return path
    # kept deliberately: the honest expectation is that the SCREEN does not change, because the
    # Files window is showing ~ and never re-reads the Desktop. A hit here would be a lie.
    trial("file_written_offscreen", make_file, None, "PUT a file on the Desktop via the Seed API (no window shows it)")
    delete_file(args.computer, path)

    print("3. text appears in the terminal (the assistant types it, so we know what to expect)")
    chat.say("run `echo warming-up`")  # open the terminal first, so the trial measures TEXT, not a window opening
    time.sleep(1.0)

    def run_cmd() -> str:
        chat.say("run `echo change-eval-marker`")
        time.sleep(1.0)
        return "change-eval-marker"
    trial("terminal_text", run_cmd, "change-eval-marker", "asked the assistant to echo a marker")

    print("4. an app window opens")

    def open_app() -> str:
        chat.say("open the text editor")
        time.sleep(1.0)
        return "Text Editor"
    trial("window_opened", open_app, "text editor", "asked the assistant to open the Text Editor")

    print("5. control again, after all that traffic")
    trial("nothing_changed_again", lambda: "nothing", None, "no intervention at all")

    hits = [t for t in trials if t["expected"]]
    controls = [t for t in trials if not t["expected"]]
    report = {
        "what": "can the live assistant say what changed since the last message, with ground truth we caused",
        "provenance": {"environment": "Seed simulator (third-party, not ours)", "grader": "this script, from interventions it performed itself",
                       "held_out": "n/a — every trial's truth is what we did, known before the question was asked"},
        "trials": trials,
        "detected": sum(t["hit"] for t in hits), "detectable": len(hits),
        "false_changes_total": sum(t["false_changes"] for t in trials),
        "controls_clean": sum(1 for t in controls if t["reported"] == 0), "controls": len(controls),
        "volatile_ignored_total": sum(t["volatile_ignored"] for t in trials),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=1))
    print(f"\ndetected {report['detected']}/{report['detectable']} · controls clean {report['controls_clean']}/{report['controls']} "
          f"· false changes {report['false_changes_total']} · volatile ignored {report['volatile_ignored_total']}")
    print(f"written to {OUT}")


if __name__ == "__main__":
    main()
