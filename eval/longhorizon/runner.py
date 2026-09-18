"""Run long-horizon tasks under three arms and score them with the hidden checkers.

    LH_SCRATCH=/path PYTHONPATH=src:. python -m eval.longhorizon.runner \
        --arm asis --label asis-1 [--tasks own/coding_ledger,tb2/fix-git] [--minutes 45]

Arms
    asis      the assistant exactly as it stands (grammar + skills + teacher-guided learning)
    improved  the same assistant with long-horizon settings switched on (see learning.LongHorizon)
    react     a plain ReAct loop on the same model and sandbox, with no tensorcode structure

The runner plays the user: it answers the assistant's questions from a fixed policy (approve
inside the sandbox, pick the first option when asked to choose) and may nudge it to keep going,
without ever revealing anything a checker looks at.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
import urllib.request
from pathlib import Path

from . import tasks as T
from .webterm import Shell, bwrap_command, serve

SP = T.SP
OUT = Path(__file__).parents[2] / "eval" / "results" / "longhorizon.json"
TEACHER = os.environ.get("TEACHER_URL", "http://127.0.0.1:8790/")
NUDGE = "Check your work against the task and fix anything that is wrong or missing. Say done when it is finished."


class Budget(RuntimeError):
    pass


def prepare(task: T.Task, label: str) -> Path:
    workdir = SP / "lh" / "runs" / label / task.id.replace("/", "__")
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True)
    task.setup(workdir)
    return workdir


def answer_for(question: str) -> str:
    q = question.lower()
    if "which one" in q or q.strip().endswith("?") and "\n1." in question:
        return "1"
    return "yes"


def run_assistant(task: T.Task, workdir: Path, *, port: int, improved: bool, minutes: float, nudges: int, record: dict) -> dict:
    import tensorcode as tc
    from playwright.sync_api import sync_playwright

    from examples.browser_agents import harness
    from examples.browser_agents.assistant import agent, learning
    from examples.browser_agents.browser import Browser
    from examples.browser_agents.mind import scene_graph

    learning.configure(long_horizon=improved)
    agent.BODY.teacher_url = TEACHER
    shell = Shell(bwrap_command(workdir, T.TOOLS))
    server = serve(shell, port)
    deadline = time.time() + minutes * 60
    calls0 = agent.BODY.model_calls
    transcript: list[tuple[str, str]] = []
    status, reason = "?", ""
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            ctx = browser.new_context(viewport={"width": 1280, "height": 800})
            page = ctx.new_page()
            page.goto(f"http://127.0.0.1:{port}/")
            page.wait_for_timeout(600)
            ui = Browser(page, episode=task.id)
            mind = agent.new_mind()
            agent.BODY.programs.clear()
            agent.BODY.jobs.clear()
            rt = harness.runtime_for(harness.Task("lh", lambda b, s: "", agent.SPEC, lambda: [scene_graph]))

            def on_cycle(m, t, i) -> None:
                if time.time() > deadline:
                    raise Budget("time cap reached")
                if agent.BODY.model_calls - calls0 > task.max_model_calls:
                    raise Budget("model-call cap reached")

            def on_say(text: str) -> None:
                transcript.append(("assistant", text))

            def send(text: str) -> None:
                transcript.append(("user", text))
                with tc.use(rt):
                    agent.respond(ui, mind, text, on_say=on_say, on_cycle=on_cycle)

            send(task.instruction)
            for _ in range(40):  # answer questions; nudge at most `nudges` times when it stops early
                if time.time() > deadline:
                    break
                awaiting = agent.one(mind, agent.ME, "awaiting") is not None
                if awaiting:
                    send(answer_for(transcript[-1][1] if transcript else ""))
                    continue
                if nudges > 0:
                    nudges -= 1
                    send(NUDGE)
                    continue
                break
            status, reason = "stopped", "assistant finished its requests"
            browser.close()
    except Budget as exc:
        status, reason = "out_of_budget", str(exc)
    except Exception as exc:  # noqa: BLE001
        status, reason = "error", f"{type(exc).__name__}: {exc}"
    finally:
        record["shell_log"] = [{k: v for k, v in b.items() if k != "out"} | {"out": b["out"][-3000:]} for b in shell.blocks]
        record["commands"] = len(shell.blocks)
        shell.close()
        server.shutdown()
    record["transcript"] = transcript
    return {"status": status, "reason": reason, "model_calls": agent.BODY.model_calls - calls0}


def run_react_arm(task: T.Task, workdir: Path, *, minutes: float, record: dict) -> dict:
    from .react import run_react

    shell = Shell(bwrap_command(workdir, T.TOOLS))
    calls: list[dict] = []
    try:
        out = run_react(shell, task.instruction, teacher_url=TEACHER, minutes=minutes, max_calls=task.max_model_calls, log=calls)
    finally:
        record["shell_log"] = [{k: v for k, v in b.items() if k != "out"} | {"out": b["out"][-3000:]} for b in shell.blocks]
        record["commands"] = len(shell.blocks)
        record["model_log"] = calls
        shell.close()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=("asis", "improved", "react"))
    ap.add_argument("--label", required=True)
    ap.add_argument("--tasks", default="")
    ap.add_argument("--minutes", type=float, default=None)
    ap.add_argument("--nudges", type=int, default=2)
    ap.add_argument("--max-calls", type=int, default=None, help="model-call budget per task (the primary budget; wall time is a safety cap)")
    ap.add_argument("--port", type=int, default=8830)
    args = ap.parse_args()
    chosen = T.by_id(args.tasks.split(",") if args.tasks else None)
    rows = []
    for n, task in enumerate(chosen):
        minutes = args.minutes if args.minutes is not None else task.minutes
        if args.max_calls is not None:
            task.max_model_calls = args.max_calls
        workdir = prepare(task, args.label)
        record: dict = {}
        t0 = time.time()
        print(f"\n=== {args.arm} {task.id} ({task.split}, {task.difficulty}) cap {minutes} min", flush=True)
        if args.arm == "react":
            outcome = run_react_arm(task, workdir, minutes=minutes, record=record)
        else:
            outcome = run_assistant(task, workdir, port=args.port + n, improved=args.arm == "improved", minutes=minutes, nudges=args.nudges, record=record)
        wall = time.time() - t0
        try:
            result = task.check(workdir)
        except Exception as exc:  # noqa: BLE001
            result = {"passed": False, "score": 0.0, "detail": f"checker error: {type(exc).__name__}: {exc}", "checks": {}}
        row = {"task": task.id, "split": task.split, "difficulty": task.difficulty, "arm": args.arm, "passed": bool(result["passed"]), "score": result["score"],
               "detail": result.get("detail", "")[:600], "checks": result.get("checks", {}), "status": outcome["status"], "reason": outcome["reason"][:300],
               "model_calls": outcome.get("model_calls", 0), "commands": record.get("commands", 0), "wall_seconds": round(wall, 1)}
        rows.append(row)
        logs = SP / "lh" / "logs" / args.label
        logs.mkdir(parents=True, exist_ok=True)
        (logs / f"{task.id.replace('/', '__')}.json").write_text(json.dumps({"row": row, **record, "verifier_log": result.get("log", "")}, indent=1, default=str))
        print(f"--- {task.id}: {'PASS' if row['passed'] else 'fail'} score={row['score']} status={row['status']} calls={row['model_calls']} cmds={row['commands']} {row['wall_seconds']}s\n    {row['detail'][:300]}", flush=True)
        results = json.loads(OUT.read_text()) if OUT.exists() else {}
        results.setdefault("runs", {})[args.label] = {"arm": args.arm, "date": time.strftime("%Y-%m-%d %H:%M"), "teacher": TEACHER, "tasks": rows}
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(results, indent=1))
    done = sum(r["passed"] for r in rows)
    print(f"\n{args.label} ({args.arm}): {done}/{len(rows)} passed; mean score {sum(r['score'] for r in rows) / max(len(rows), 1):.2f}")


if __name__ == "__main__":
    main()
