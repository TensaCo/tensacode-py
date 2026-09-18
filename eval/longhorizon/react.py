"""Calibration baseline: the same teacher model driving the same sandbox, with no tensorcode.

A plain ReAct loop: the model sees the task, its own notes, the last few steps and the tail of
the most recent command's output, and replies with one JSON action (run, write_file, note, done).
Same sandbox, same time and model-call budgets, same hidden checker. No perception through a
screen, no claims, no plan structure, no confirmations.
"""

from __future__ import annotations

import json
import re
import time
import urllib.request
from pathlib import Path

SYSTEM = """You are working in a Linux sandbox to finish a task. Reply with ONE JSON object and nothing else:
  {"thought": "<one sentence>", "do": "run", "command": "<one bash command>"}
  {"thought": "...", "do": "write_file", "path": "/app/...", "content": "<full file contents>"}
  {"thought": "...", "do": "note", "text": "<something to remember>"}
  {"thought": "...", "do": "done", "reply": "<what you achieved>"}
There is no network. Python with numpy, scipy, pandas, scikit-learn, mne, matplotlib, trimesh and
build123d is at /opt/tools/bin/python. Work inside /app. Check your work before saying done."""


def ask(url: str, system: str, user: str, max_new_tokens: int = 900) -> dict:
    body = json.dumps({"system": system, "user": user, "max_new_tokens": max_new_tokens}).encode()
    with urllib.request.urlopen(urllib.request.Request(url, data=body, method="POST"), timeout=900) as r:
        return json.loads(r.read())


def parse(text: str) -> dict | None:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    depth, start = 0, None
    for i, ch in enumerate(text):
        if ch == "{":
            depth += 1
            start = i if start is None else start
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    out = json.loads(text[start : i + 1])
                    return out if isinstance(out, dict) else None
                except json.JSONDecodeError:
                    start = None
    return None


def run_react(shell, instruction: str, *, teacher_url: str, minutes: float, max_calls: int, log: list[dict], tail_chars: int = 4000) -> dict:
    notes: list[str] = []
    history: list[str] = []
    deadline = time.time() + minutes * 60
    calls = 0
    last = ""
    while time.time() < deadline and calls < max_calls:
        prompt = (f"TASK:\n{instruction}\n\nNOTES:\n" + ("\n".join(f"- {n}" for n in notes) or "  (none)")
                  + "\n\nRECENT STEPS:\n" + ("\n".join(history[-6:]) or "  (none)")
                  + f"\n\nLAST OUTPUT:\n{last[-tail_chars:] or '(nothing yet)'}")
        try:
            answer = ask(teacher_url, SYSTEM, prompt)
        except Exception as exc:  # noqa: BLE001
            return {"status": "error", "reason": f"teacher unreachable: {exc}", "model_calls": calls}
        calls += 1
        step = parse(answer.get("text", ""))
        log.append({"t": time.time(), "prompt_tokens": answer.get("prompt_tokens"), "new_tokens": answer.get("new_tokens"), "seconds": answer.get("seconds"), "reply": answer.get("text"), "parsed": step})
        if step is None:
            history.append("  (unparseable reply; reminded to answer with one JSON object)")
            last = "Your last reply was not a single JSON object."
            continue
        do = step.get("do")
        if do == "done":
            return {"status": "done", "reason": step.get("reply", ""), "model_calls": calls}
        if do == "note":
            notes.append(str(step.get("text", ""))[:300])
            history.append(f"  note: {step.get('text', '')[:120]}")
            last = "noted"
            continue
        if do == "write_file":
            path, content = str(step.get("path", "")), str(step.get("content", ""))
            if not path.startswith("/app/"):
                last = "write_file only works under /app/"
                history.append(f"  write_file {path} -> refused (outside /app)")
                continue
            shell.run(f"mkdir -p $(dirname {path!r})")
            _wait(shell, deadline)
            payload = content.encode().hex()
            shell.run(f"python -c \"import sys,binascii,pathlib; pathlib.Path({path!r}).write_bytes(binascii.unhexlify('{payload}'))\"")
            out = _wait(shell, deadline)
            last = out or f"wrote {len(content)} bytes to {path}"
            history.append(f"  write_file {path} ({len(content)} bytes) -> exit {shell.blocks[-1]['exit']}")
            continue
        if do == "run":
            command = str(step.get("command", "")).strip()
            if not command:
                last = "run needs a command"
                continue
            shell.run(command)
            out = _wait(shell, deadline)
            code = shell.blocks[-1]["exit"]
            last = f"$ {command}\n(exit {code})\n{out}"
            history.append(f"  $ {command[:120]} -> exit {code}")
            continue
        last = f"unknown action {do!r}"
        history.append(f"  (unknown action {do!r})")
    return {"status": "out_of_budget", "reason": f"{calls} model calls, {minutes} min cap", "model_calls": calls}


def _wait(shell, deadline: float, step_timeout: float = 240.0) -> str:
    end = min(time.time() + step_timeout, deadline + 60)
    while shell.blocks and shell.blocks[-1]["exit"] is None and time.time() < end:
        time.sleep(0.2)
    if shell.blocks and shell.blocks[-1]["exit"] is None:
        shell.interrupt()
        time.sleep(1)
    return shell.blocks[-1]["out"] if shell.blocks else ""
