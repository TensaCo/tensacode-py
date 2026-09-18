"""Run the general agent over a held-out split and record what it did.

    python -m eval.general_agent.run_split --split dev [--limit N]

Each item gets a fresh computerworld desktop and a fresh agent, so one prompt cannot
leave state for the next. ``conversation_facts`` items replay their ``history`` turns
first. Images are not yet an input the agent takes (the vision path is not built), so
image items are run on their text alone and flagged.

What is measured now is what can be graded without trusting the agent's report:

* **false action** — the machine was changed (a write capability ran and its receipt
  was applied) for a prompt from a category that asks for no change to this machine
  (knowledge, arithmetic, open-ended, ambiguous, image and screen questions,
  conversation facts). Graded from the event log, not the reply;
* **outcome mix** — how many acts ended done / declined / not understood / unknown /
  answered / noted / mentioned;
* **parse coverage** — the share of word tokens the parse used, per sentence.

Task success on desktop and shell items needs graders written for computerworld; the
OSWorld and Mind2Web references describe other environments (eval/heldout/README.md).

Rules (eval/heldout/README.md): dev is for development; this script refuses ``test``
unless ``--i-am-running-the-test-split`` is given, and then prints aggregates only.
Results are appended to ``eval/results/general_agent_runs.jsonl``, never overwritten.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

sys.path[:0] = [str(Path(__file__).parents[2] / "src"), str(Path(__file__).parents[2])]

DATA = Path.home() / ".cache" / "tensorcode" / "heldout"
OUT = Path(__file__).parents[1] / "results" / "general_agent_runs.jsonl"
NO_CHANGE_ASKED = {"general_knowledge", "arithmetic", "open_ended", "ambiguous", "image_questions",
                   "screen_questions", "conversation_facts", "owner"}


def run_item(item: dict, reader: str = "grammar") -> dict:
    from examples.browser_agents.worlds import desktop_world
    from examples.browser_agents.worlds.runtime import CwWorld
    from examples.general_agent.desktop import DesktopPlugin
    from tensorcode.agent import Agent

    plugin = DesktopPlugin(CwWorld(desktop_world(), 0))
    agent = Agent([plugin], reader=_reader(reader))
    for turn in item.get("history") or []:
        text = turn.get("content") if isinstance(turn, dict) else str(turn)
        if text and (not isinstance(turn, dict) or turn.get("role", "user") == "user"):
            agent.turn(text[:4000])
    t0 = time.perf_counter()
    turn = agent.turn(item["text"])
    seconds = time.perf_counter() - t0
    writes = [e for e in turn.events if e["type"] == "receipt" and e["status"] == "applied"
              and any(c.name == e["capability"] and c.effect_kind != "read" for c in plugin.capabilities())]
    return {
        "id": item["id"], "category": item["category"],
        "statuses": [o.status for o in turn.outcomes],
        "coverage": [s.coverage for s in turn.sentences],
        "writes": [e["capability"] for e in writes],
        "false_action": bool(writes) and item["category"] in NO_CHANGE_ASKED,
        "had_image": bool(item.get("image")),
        "seconds": round(seconds, 3),
        "reply": turn.reply,
    }


_READER_CACHE: dict = {}


def _reader(kind: str):
    """``grammar``: the hand-written grammar. ``learned``: the treebank-trained parser."""
    if kind == "grammar":
        return None
    if kind not in _READER_CACHE:
        from tensorcode.agent.understand import LearnedReader

        _READER_CACHE[kind] = LearnedReader()
    return _READER_CACHE[kind]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=["dev", "calibration", "test"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--i-am-running-the-test-split", action="store_true")
    ap.add_argument("--show", action="store_true", help="print each dev item's reply (dev only)")
    ap.add_argument("--reader", default="grammar", choices=["grammar", "learned"])
    args = ap.parse_args()
    if args.split == "test" and not args.i_am_running_the_test_split:
        sys.exit("test is for headline numbers, run rarely: pass --i-am-running-the-test-split")
    path = DATA / f"{args.split}.jsonl"
    raw = path.read_bytes()
    manifest = json.loads((Path(__file__).parents[1] / "heldout" / "MANIFEST.json").read_text())
    want = manifest["splits"][args.split]["sha256"]
    got = hashlib.sha256(raw).hexdigest()
    if got != want:
        sys.exit(f"{args.split}.jsonl does not match the manifest ({got[:12]} != {want[:12]}): rebuild or --verify")
    items = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    if args.limit:
        items = items[: args.limit]
    rows = []
    for item in items:
        try:
            r = run_item(item, args.reader)
        except Exception as exc:  # noqa: BLE001 - a crash is a result, recorded as such
            r = {"id": item["id"], "category": item["category"], "crash": f"{type(exc).__name__}: {exc}",
                 "statuses": [], "coverage": [], "writes": [], "false_action": False}
        rows.append(r)
        if args.show and args.split == "dev":
            print(f"[{r['category']}] {item['text'][:110]!r}\n   -> {r.get('reply', r.get('crash'))[:220]}", flush=True)
    by_cat = collections.defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)
    summary = {}
    for cat, rs in sorted(by_cat.items()):
        statuses = collections.Counter(s for r in rs for s in r["statuses"])
        cov = [c for r in rs for c in r["coverage"]]
        summary[cat] = {"n": len(rs), "false_actions": sum(r["false_action"] for r in rs),
                        "crashes": sum(1 for r in rs if "crash" in r), "items_with_a_write": sum(1 for r in rs if r["writes"]),
                        "outcomes": dict(statuses), "median_coverage": round(statistics.median(cov), 3) if cov else None}
    total_fa = sum(v["false_actions"] for v in summary.values())
    no_change_items = sum(v["n"] for k, v in summary.items() if k in NO_CHANGE_ASKED)
    record = {"eval": "general_agent_split_run", "at": time.strftime("%Y-%m-%dT%H:%M:%S"), "split": args.split,
              "reader": args.reader,
              "split_sha256": got, "commit": _commit(), "n": len(rows),
              "false_action_rate": f"{total_fa}/{no_change_items}", "by_category": summary}
    if args.split == "dev":
        record["items"] = rows
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("a") as f:
        f.write(json.dumps(record) + "\n")
    print(json.dumps({k: v for k, v in record.items() if k != "items"}, indent=1))


def _commit() -> str:
    import subprocess

    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).parent, text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


if __name__ == "__main__":
    main()
