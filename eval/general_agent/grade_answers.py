"""Grade the agent's *answers* on the held-out prompts, against each source's own gold.

    python -m eval.general_agent.grade_answers [--split dev] [--reader learned|grammar]

Until now the dev runs measured safety (did it change anything it shouldn't?) and how
much it parsed. They never asked whether an answer was right. This does, using the
answers the datasets ship:

* ``general_knowledge`` — Natural Questions / WebQuestions accept a list of answers;
* ``arithmetic`` — GSM8K's final number;
* ``image_questions`` — VQA's majority answer over its ten annotators (the image is given
  to the agent, so the vision plugin is in play);
* ``screen_questions`` — ScreenQA's answers (likewise with the screenshot);
* ``conversation_facts`` — LongMemEval's answer, graded loosely: the gold's content words
  (names, numbers, rare words) must mostly appear. Reported separately because a loose
  grader can flatter;
* ``ambiguous`` — there is no single right answer: the wanted behaviour is to *ask*.
  Scored as "asked a clarifying question" vs "answered anyway";
* ``open_ended`` — no gold answer exists. Only "attempted vs declined" is reported.

Every item is scored three ways: **answered** (it committed to an answer), **correct**,
and **wrong** (answered and not correct). Abstaining is neither correct nor wrong, which
is the distinction the whole design rests on.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path[:0] = [str(Path(__file__).parents[2] / "src"), str(Path(__file__).parents[2])]

DATA = Path.home() / ".cache" / "tensorcode" / "heldout"
OUT = Path(__file__).parents[1] / "results" / "general_agent_answers.jsonl"
QA = ("general_knowledge", "arithmetic", "image_questions", "screen_questions", "conversation_facts",
      "ambiguous", "open_ended")
ABSTAIN = ("i don't know", "i do not know", "nothing i know", "couldn't recognise", "could not recognise",
           "didn't fully follow", "did not fully follow", "i can not", "i cannot", "there is nothing there",
           "it didn't ask me", "i noted")


def normalise(text: object) -> str:
    text = str(text)
    return " ".join("".join(c.lower() if c.isalnum() or c.isspace() else " " for c in text).split())


def golds(item: dict) -> list[str]:
    ref = item.get("reference") or {}
    if item["category"] == "arithmetic":
        return [str(ref.get("final", "")).replace(",", "")]
    if item["category"] == "image_questions":
        answers = ref.get("answers") or [ref.get("answer")]
        return [collections.Counter(a for a in answers if a).most_common(1)[0][0]] if answers else []
    if item["category"] in ("general_knowledge", "screen_questions"):
        return [a for a in (ref.get("answers") or []) if a]
    if item["category"] == "conversation_facts":
        return [str(ref.get("answer", ""))] if ref.get("answer") else []
    return []


def content_words(text: str) -> set[str]:
    """Words worth matching on: numbers, and words of four letters or more."""
    return {w for w in normalise(text).split() if w.isdigit() or len(w) >= 4}


def judge(item: dict, reply: str) -> dict:
    answered = not any(phrase in reply.lower() for phrase in ABSTAIN) and len(reply.strip()) > 0
    category = item["category"]
    if category == "ambiguous":
        return {"answered": answered, "asked_back": "?" in reply, "correct": None}
    if category == "open_ended":
        return {"answered": answered, "correct": None}
    gold = golds(item)
    if not gold:
        return {"answered": answered, "correct": None}
    said = normalise(reply)
    if category == "conversation_facts":  # loose: most of the gold's content words appear
        want = content_words(gold[0])
        overlap = len(want & content_words(reply)) / max(1, len(want))
        return {"answered": answered, "correct": bool(answered and overlap >= 0.5), "overlap": round(overlap, 2)}
    correct = any(normalise(g) and normalise(g) in said for g in gold)
    return {"answered": answered, "correct": bool(answered and correct)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=["dev", "calibration", "test"])
    ap.add_argument("--reader", default="learned", choices=["grammar", "learned"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--i-am-running-the-test-split", action="store_true")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()
    if args.split == "test" and not args.i_am_running_the_test_split:
        sys.exit("test is for headline numbers: pass --i-am-running-the-test-split")

    from examples.browser_agents.worlds import desktop_world
    from examples.browser_agents.worlds.runtime import CwWorld
    from examples.general_agent.desktop import DesktopPlugin
    from tensorcode.agent import Agent
    from tensorcode.agent.vision_plugin import VisionPlugin

    reader = None
    if args.reader == "learned":
        from tensorcode.agent.understand import LearnedReader

        reader = LearnedReader()
    raw = (DATA / f"{args.split}.jsonl").read_bytes()
    manifest = json.loads((Path(__file__).parents[1] / "heldout" / "MANIFEST.json").read_text())
    if hashlib.sha256(raw).hexdigest() != manifest["splits"][args.split]["sha256"]:
        sys.exit(f"{args.split}.jsonl does not match the manifest")
    items = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    items = [i for i in items if i["category"] in QA]
    if args.limit:
        items = items[: args.limit]
    rows = []
    for item in items:
        try:
            plugin = DesktopPlugin(CwWorld(desktop_world(), 0), learn=False)
            agent = Agent([plugin, VisionPlugin()], reader=reader)
            for turn in item.get("history") or []:
                for message in (turn.get("messages") or []) if isinstance(turn, dict) else []:
                    if message.get("role", "user") == "user" and message.get("content"):
                        agent.turn(str(message["content"])[:2000])
            images = []
            if item.get("image"):
                path = DATA / item["image"]
                if path.exists():
                    images = [path.read_bytes()]
            reply = agent.turn(item["text"], images=images).reply
        except Exception as exc:  # noqa: BLE001
            reply = f"__crash__ {type(exc).__name__}: {exc}"
        row = {"id": item["id"], "category": item["category"], "reply": reply, **judge(item, reply),
               "crash": reply.startswith("__crash__")}
        rows.append(row)
        if args.show:
            print(f"[{row['category']}] {item['text'][:80]!r}\n   -> {reply[:150]!r}  answered={row['answered']} correct={row['correct']}", flush=True)
    summary = {}
    for cat in QA:
        rs = [r for r in rows if r["category"] == cat]
        if not rs:
            continue
        gradable = [r for r in rs if r["correct"] is not None]
        answered = [r for r in rs if r["answered"]]
        correct = [r for r in gradable if r["correct"]]
        wrong = [r for r in gradable if r["answered"] and not r["correct"]]
        summary[cat] = {"n": len(rs), "answered": len(answered), "correct": len(correct), "wrong": len(wrong),
                        "crashes": sum(1 for r in rs if r["crash"]),
                        "accuracy": round(len(correct) / len(gradable), 3) if gradable else None,
                        "precision_when_answering": round(len(correct) / len(answered), 3) if answered and gradable else None}
        if cat == "ambiguous":
            summary[cat]["asked_back"] = sum(1 for r in rs if r.get("asked_back"))
    record = {"eval": "general_agent_answers", "at": time.strftime("%Y-%m-%dT%H:%M:%S"), "split": args.split,
              "reader": args.reader, "n": len(rows), "by_category": summary}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("a") as f:
        f.write(json.dumps({**record, "items": rows if args.split == "dev" else []}) + "\n")
    print(json.dumps(record, indent=1))


if __name__ == "__main__":
    main()
