"""Measure the four social faculties, before and after, on both halves.

    python -m eval.social.measure [--out eval/results/social_measure.json]

Runs offline against a fake machine (the harness pattern from tests/test_assistant_procedures),
so it is deterministic and needs no server. Each faculty reports the rate its prediction named,
and where a prediction had two rates (a gain bought with a loss) both are here.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import tensacode as tc  # noqa: E402
from examples.browser_agents.assistant import interpreter as I  # noqa: E402
from examples.browser_agents.assistant import procedures as PR  # noqa: E402
from examples.browser_agents.assistant import programs as P  # noqa: E402
from examples.browser_agents.assistant.agent import _read_with_grammar, hear  # noqa: E402
from examples.browser_agents.assistant.language import Frame, parse_message  # noqa: E402
from examples.browser_agents.mind import one  # noqa: E402
from eval.social.cases import CLEAR, INDIRECT, PRESUPPOSED, UNDERDETERMINED  # noqa: E402

HOME = "/home/agent"
TREE = {
    f"{HOME}/Desktop": None,
    f"{HOME}/Desktop/notes.txt": "hello\nworld\n",
    f"{HOME}/Desktop/reports.txt": "q3\n",
    f"{HOME}/Desktop/shopping.txt": "eggs\n",
    f"{HOME}/Desktop/photo.png": "",
    f"{HOME}/Desktop/recipes": None,
    f"{HOME}/Desktop/recipes/pasta.txt": "boil water\n",
    f"{HOME}/Documents": None,
    f"{HOME}/Documents/budget.xlsx": "",
    f"{HOME}/Downloads": None,
    f"{HOME}/Downloads/invoice.pdf": "",
    f"{HOME}/Downloads/setup.sh": "",
    f"{HOME}/Downloads/screenshot.png": "",
    f"{HOME}/Pictures": None,
    f"{HOME}/Pictures/trip.jpg": "",
    f"{HOME}/Pictures/holiday.jpg": "",
    f"{HOME}/Projects": None,
    f"{HOME}/Projects/alpha": None,
    f"{HOME}/Projects/alpha/main.py": "",
    f"{HOME}/Projects/notes.md": "",
}
# ~/Videos is deliberately absent: one case presupposes it.


class FakeShell:
    """Enough file system to answer what the assistant actually types."""

    def __init__(self, tree: dict) -> None:
        self.tree, self.commands = dict(tree), []

    def kind(self, path: str) -> str | None:
        if path in self.tree:
            return "directory" if self.tree[path] is None else "regular file"
        if any(p.startswith(path.rstrip("/") + "/") for p in self.tree):
            return "directory"
        return None

    def run(self, command: str) -> P.Output:
        self.commands.append(command)
        if command.startswith("stat -c"):
            out, err = [], []
            for raw in re.findall(r"(?:'([^']*)'|(\S+))", command[len("stat -c '%F|%s|%n' "):]):
                path = raw[0] or raw[1]
                if not path or path.startswith("-") or "%" in path:
                    continue
                k = self.kind(path)
                out.append(f"{k}|{len(self.tree.get(path) or '')}|{path}") if k else err.append(
                    f"stat: cannot statx '{path}': No such file or directory")
            return P.Output("\n".join(out + err))
        if m := re.match(r"^ls -1pA '?([^']+)'?$", command):
            root = m[1].rstrip("/")
            if self.kind(root) is None:
                return P.Output(f"ls: cannot access '{root}': No such file or directory")
            names = set()
            for p in self.tree:
                if p.startswith(root + "/"):
                    rest = p[len(root) + 1:]
                    names.add(rest.split("/")[0] + ("/" if "/" in rest or self.tree[p] is None else ""))
            return P.Output("\n".join(sorted(names)))
        if m := re.match(r"^(?:head|tail) -n \d+ '?([^']+)'?$", command):
            return P.Output(self.tree.get(m[1]) or "")
        if m := re.match(r"^find (\S+)(?: -maxdepth (\d+))?(?: -i?name '?([^'\s]+)'?)?(?: -type ([fd]))?", command):
            root, depth, pat, typ = m[1].rstrip("/"), m[2], m[3] or "*", m[4]
            ci = " -iname " in command
            out = []
            for path in sorted(self.tree):
                if not path.startswith(root + "/"):
                    continue
                rest = path[len(root) + 1:]
                if depth and rest.count("/") + 1 > int(depth):
                    continue
                is_dir = self.tree[path] is None
                if typ == "f" and is_dir or typ == "d" and not is_dir:
                    continue
                name = rest.rsplit("/", 1)[-1]
                if fnmatch.fnmatch(name.lower() if ci else name, pat.lower() if ci else pat):
                    out.append(path)
            return P.Output("\n".join(out))
        if command.startswith(("mkdir", "mv", "rm", "cp", "printf", "touch")):
            return P.Output("")
        return P.Output("")


def run_request(frame: Frame, shell: FakeShell, answers: list[str], *, mind: tc.Store | None = None,
                turn: int = 1) -> tuple[list[str], list[str], tc.Store]:
    """Walk one request through the procedure engine; return what it said and what it typed."""
    mind = mind if mind is not None else tc.Store()
    req = tc.Ref(f"request:{turn}.0")
    said, answers = [], list(answers)

    def say(mind_, req_, text, cycle):
        said.append(text)
        return I.Thought()

    host = I.Host(say=say, find=lambda pid: PR.BY_ID.get(pid))
    proc = PR.BY_ACT.get(frame.act) or PR.BY_ID["unknown"]
    env = {"slot": dict(frame.slots), "words": frame.words, "act": frame.act, "turn": turn,
           "cwd": HOME, "focus": None, "focus_kind": None, "known": {}}
    before = len(shell.commands)
    from examples.browser_agents.assistant.memory import note as _note

    _note(mind, [tc.Claim(req, "order", (turn, 0))], f"utterance:{turn}")  # as hear() would
    I.begin(mind, req, proc, env, 0)
    value = None
    for cycle in range(300):
        I.advance(mind, req, value, cycle, host)
        if I.one(mind, req, "status") in ("done", "failed"):
            break
        doing = I.one(mind, req, "doing")
        if doing is None:
            break
        kind = I.one(mind, doing, "kind")
        if kind == "run":
            value = shell.run(I.one(mind, doing, "command"))
        elif kind == "ask":
            value = Frame("choose", answers.pop(0)) if answers else Frame("cancel", "no")
        elif kind == "look":
            value = P.Seen((), ())
        else:
            value = True
    return said, shell.commands[before:], mind



def read_frame(text: str, turn: int = 1) -> Frame:
    """Read a message the way the live assistant reads it: every tier, in order, with the gate.

    The first version of this harness asked ``parse_message`` alone, and so missed that the
    symbolic grammar tier already turned "I can't find my keys" into a search before the
    affordance gate was ever consulted. Measuring anything but ``hear`` measures a path no
    user takes.
    """
    mind = tc.Store()
    hear(mind, text, turn)
    req = tc.Ref(f"request:{turn}.0")
    act = one(mind, req, "act") or "unknown"
    slots = {r.claim.predicate[len("slot:"):]: r.claim.object
             for r in mind.claims(req) if r.claim.predicate.startswith("slot:")}
    return Frame(str(act), text, {k: v for k, v in slots.items() if v is not None})


def read_frame_before(text: str) -> Frame:
    """The same chain as it stood before this work: rules, then the grammar tier, ungated."""
    frames = parse_message(text)
    if frames and frames[0].act != "unknown":
        return frames[0]
    return _read_with_grammar(text) or Frame("unknown", text)


# --------------------------------------------------------------- faculties


def measure_indirect() -> dict:
    """Requests in another form should become acts; near misses must not. Both rates, both halves."""
    rows = []
    for case in INDIRECT:
        frame, was = read_frame(case.text), read_frame_before(case.text)
        read, before = frame.act, was.act
        if case.want == "no_action":
            ok = read == "unknown"
        else:
            ok = read == case.want and all(
                str(frame.slots.get(k, "")).lower() == str(v).lower() for k, v in case.slots.items())
        was = before == "unknown" if case.want == "no_action" else before == case.want
        rows.append({"text": case.text, "want": case.want, "read": read, "ok": ok, "ok_before": was,
                     "provenance": case.provenance, "half": case.half, "note": case.note})
    return _split_rates(rows, positive=lambda r: r["want"] != "no_action")


def measure_clarification() -> dict:
    """Does an underdetermined goal become answerable after one question, without pestering clear ones?"""
    answerable, rows = [], []
    for case in UNDERDETERMINED:
        frame = read_frame(case.text)
        act = frame.act
        shell = FakeShell(TREE)
        said, commands, mind = ([], [], tc.Store())
        asked = acted = False
        if act == "clarify_goal":
            said, commands, mind = run_request(frame, shell, answers=["1"])  # "1" = the first option offered
            asked = any("?" in s for s in said[:1]) or any("organized" in s.lower() for s in said)
            acted = any(c.startswith("mkdir") or " && mv " in c for c in commands)
        rows.append({"text": case.text, "act": act, "asked": asked, "acted_after_answer": acted,
                     "provenance": case.provenance, "half": case.half, "said": said[:2], "commands": commands[:3]})
        answerable.append(bool(asked and acted))
    false_clar = []
    for case in CLEAR:
        act = read_frame(case.text).act
        false_clar.append({"text": case.text, "act": act, "want": case.want,
                           "asked": act == "clarify_goal", "provenance": case.provenance, "half": case.half})
    def rate(items, key):
        return round(sum(bool(i[key]) for i in items) / len(items), 4) if items else None
    halves = {}
    for half in ("design", "heldout"):
        u = [r for r in rows if r["half"] == half]
        c = [r for r in false_clar if r["half"] == half]
        halves[half] = {
            "n_underdetermined": len(u), "answerable_after_one_question": rate(u, "acted_after_answer"),
            "n_clear": len(c), "false_clarification_rate": rate(c, "asked"),
        }
    return {"halves": halves, "underdetermined": rows, "clear": false_clar}


def measure_presupposition(*, ablate: bool = False) -> dict:
    """A false presupposition should be corrected with the near miss named, not reported as absence.

    ``ablate`` blinds the near-miss search, which reproduces the older mind exactly: it still
    looks, still fails, and still reports a bare absence. That is the "before" column.
    """
    import tensacode.social as S

    real, rows = S.near_names, []
    if ablate:
        S.near_names = lambda *a, **k: []
    for case in PRESUPPOSED:
        frame = read_frame(case.text)
        if frame.act == "unknown":
            rows.append({"text": case.text, "ok": False, "reply": "(not understood)", "half": case.half,
                         "provenance": case.provenance, "want": case.want})
            continue
        said, _cmds, _mind = run_request(frame, FakeShell(TREE), answers=[])
        reply = said[-1] if said else ""
        low = reply.lower()
        if case.want == "correct":  # wrong name: name the near miss
            near = str(case.slots.get("near", ""))
            ok = "did you mean" in low and (near.lower() in low if near else True)
        elif case.want == "located":  # right name, wrong place: say where it actually is
            ok = str(case.slots.get("at", "")).lower() in low
        elif case.want == "no_folder":  # a place that does not exist is not an empty place
            ok = "there's no" in low and "empty" not in low
        else:  # nothing near it: a bare absence is the honest reply
            ok = "couldn't find" in low and "did you mean" not in low
        rows.append({"text": case.text, "ok": ok, "reply": reply, "want": case.want,
                     "provenance": case.provenance, "half": case.half, "note": case.note})
    S.near_names = real
    return _split_rates(rows, positive=lambda r: True)


def measure_common_ground() -> dict:
    """The same fact twice should be marked, and "what did I tell you" answered in order."""
    from tensacode.social import CommonGround

    mind = tc.Store()
    shell = FakeShell(TREE)
    said = []
    for turn, text in enumerate(["my name is Jacob", "what is my name", "what is my name",
                                 "my favourite colour is green", "what did I tell you"], start=1):
        out, _c, _m = run_request(read_frame(text, turn), shell, answers=[], mind=mind, turn=turn)
        said.append({"turn": turn, "you": text, "reply": out[-1] if out else ""})
    ground = CommonGround(mind)
    told = ground.told_me()
    first_answer, second_answer = said[1]["reply"], said[2]["reply"]
    listing = said[4]["reply"]
    return {
        "transcript": said,
        "second_mention_marked": "as i mentioned" in second_answer.lower(),
        "first_mention_unmarked": "as i mentioned" not in first_answer.lower(),
        "what_did_i_tell_you_complete": all(w in listing for w in ("name", "colour")) or all(
            w in listing for w in ("Jacob", "green")),
        "what_did_i_tell_you_in_order": listing.find("Jacob") < listing.find("green") if "green" in listing else False,
        "grounded_from_you": len(told),
        "grounding_claims": len(mind.claims(predicate="shared_via")),
    }


def _split_rates(rows: list[dict], positive) -> dict:
    out = {"rows": rows, "halves": {}}
    for half in ("design", "heldout"):
        here = [r for r in rows if r["half"] == half]
        pos = [r for r in here if positive(r) and r.get("want") != "no_action"]
        neg = [r for r in here if r.get("want") == "no_action"]
        out["halves"][half] = {
            "n": len(here),
            "requests_read_correctly": round(sum(r["ok"] for r in pos) / len(pos), 4) if pos else None,
            "n_requests": len(pos),
            "non_requests_left_alone": round(sum(r["ok"] for r in neg) / len(neg), 4) if neg else None,
            "n_non_requests": len(neg),
        }
        if any("ok_before" in r for r in here):
            out["halves"][half]["requests_before"] = (
                round(sum(r.get("ok_before", False) for r in pos) / len(pos), 4) if pos else None)
            out["halves"][half]["non_requests_before"] = (
                round(sum(r.get("ok_before", False) for r in neg) / len(neg), 4) if neg else None)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "eval/results/social_measure.json")
    args = ap.parse_args()
    report = {
        "provenance_note": "USER rows are the owner's own words; FLOOR rows were authored here and split "
                           "design/heldout. A floor measures obvious breakage, not coverage.",
        "indirect": measure_indirect(),
        "clarification": measure_clarification(),
        "presupposition": measure_presupposition(),
        "presupposition_ablated": measure_presupposition(ablate=True),
        "common_ground": measure_common_ground(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, default=str))
    for name in ("indirect", "presupposition", "presupposition_ablated"):
        print(f"== {name}")
        for half, r in report[name]["halves"].items():
            was = f"  (before {r['requests_before']})" if r.get("requests_before") is not None else ""
            print(f"   {half:<8} requests {r['requests_read_correctly']} (n={r['n_requests']}){was}  "
                  f"non-requests left alone {r['non_requests_left_alone']} (n={r['n_non_requests']})")
    print("== clarification")
    for half, r in report["clarification"]["halves"].items():
        print(f"   {half:<8} answerable after one question {r['answerable_after_one_question']} (n={r['n_underdetermined']})  "
              f"false clarification {r['false_clarification_rate']} (n={r['n_clear']})")
    cg = report["common_ground"]
    print("== common ground")
    print(f"   second mention marked: {cg['second_mention_marked']}  first unmarked: {cg['first_mention_unmarked']}  "
          f"told-you listing complete: {cg['what_did_i_tell_you_complete']}  "
          f"in order: {cg['what_did_i_tell_you_in_order']}  grounded from you: {cg['grounded_from_you']}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
