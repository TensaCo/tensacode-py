"""Live probes for the axes no existing result file covers.

Each probe drives the running assistant through its HTTP interface and grades from
something the agent does not control: the simulator's shell, or our own DOM read of the
desktop page. Both the environment and the grader are still ours, but the grader is a
separate process on a separate path, and every row says so.
"""

from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass, field

from .client import Chat, listing, read_file, shell

HOME, DESK = "/home/agent", "/home/agent/Desktop"


@dataclass
class Case:
    name: str
    said: list[str]
    checks: dict[str, bool] = field(default_factory=dict)
    replies: list[str] = field(default_factory=list)
    acts: list[str] = field(default_factory=list)
    seconds: float = 0.0
    model_calls: int = 0

    #: checks that are alternative readings of the same question: any one passing is enough
    either_or: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        if not self.checks:
            return False
        hard = [v for k, v in self.checks.items() if k not in self.either_or]
        soft = [v for k, v in self.checks.items() if k in self.either_or]
        return all(hard) and (any(soft) if soft else True)


# ------------------------------------------------------------ compositionality


#: Five clauses that each leave a separately checkable mark on the machine. Depth k is
#: the first k clauses in ONE message, so depth is composition inside a single request.
def _ladder(tag: str) -> list[tuple[str, callable]]:
    a, b = f"{tag}a", f"{tag}b"
    return [
        (f"make a folder called {a} on the desktop",
         lambda: f"{a}/" in listing(DESK)),
        (f"create note.txt in it saying 'alpha'",
         lambda: "alpha" in (read_file(f"{DESK}/{a}/note.txt") or "")),
        (f"add 'beta' to note.txt",
         lambda: "beta" in (read_file(f"{DESK}/{a}/note.txt") or "")),
        (f"make a folder called {b} on the desktop",
         lambda: f"{b}/" in listing(DESK)),
        (f"copy note.txt to {b}",
         lambda: (read_file(f"{DESK}/{b}/note.txt") or "").strip().startswith("alpha")),
    ]


def compositionality(chat: Chat, depths: range = range(1, 6)) -> list[Case]:
    """Accuracy against the number of chained clauses in one message."""
    out = []
    for depth in depths:
        tag = f"cd{depth}x{int(time.time()) % 10000}"
        steps = _ladder(tag)[:depth]
        message = " then ".join(s for s, _ in steps)
        turn = chat.say(message)
        time.sleep(0.6)  # the machine settles before we read it
        case = Case(name=f"depth-{depth}", said=[message], replies=turn.replies, acts=turn.acts,
                    seconds=turn.seconds, model_calls=turn.model_calls)
        for i, (said, check) in enumerate(steps, start=1):
            try:
                case.checks[f"clause-{i}"] = bool(check())
            except Exception as exc:  # noqa: BLE001 - a failed check is a failed clause
                case.checks[f"clause-{i}"] = False
                case.replies.append(f"[check error: {exc}]")
        shell(f"rm -r {DESK}/{tag}a {DESK}/{tag}b")
        out.append(case)
    return out


# --------------------------------------------------------------- belief revision


def belief_revision(chat: Chat) -> list[Case]:
    """Told facts, replacement, forgetting, and whether a world change is re-perceived."""
    out: list[Case] = []

    t1 = chat.say("my name is Jacob")
    t2 = chat.say("what is my name")
    out.append(Case("told-fact-recalled", ["my name is Jacob", "what is my name"],
                    {"answers Jacob": "jacob" in t2.reply.lower()},
                    t1.replies + t2.replies, t1.acts + t2.acts, t1.seconds + t2.seconds, t1.model_calls + t2.model_calls))

    t3 = chat.say("my name is Sam")
    t4 = chat.say("what is my name")
    low = t4.reply.lower()
    out.append(Case("replacement-supersedes", ["my name is Sam", "what is my name"],
                    {"answers Sam": "sam" in low, "no longer says Jacob": "jacob" not in low},
                    t3.replies + t4.replies, t3.acts + t4.acts, t3.seconds + t4.seconds, t3.model_calls + t4.model_calls))

    t5 = chat.say("forget my name")
    t6 = chat.say("what is my name")
    low6 = t6.reply.lower()
    forgot = ("sam" not in low6 and "jacob" not in low6)
    out.append(Case("forgetting-takes-effect", ["forget my name", "what is my name"],
                    {"no longer answers a name": forgot},
                    t5.replies + t6.replies, t5.acts + t6.acts, t5.seconds + t6.seconds, t5.model_calls + t6.model_calls))

    # a change made behind its back: the world moves, the belief must not persist
    tag = f"stale{int(time.time()) % 10000}"
    t7 = chat.say(f"make a folder called {tag} on the desktop")
    made = f"{tag}/" in listing(DESK)
    t8 = chat.say("whats on my desktop")
    saw_it = tag in t8.reply
    shell(f"rm -r {DESK}/{tag}")
    gone = f"{tag}/" not in listing(DESK)
    t9 = chat.say("whats on my desktop")
    out.append(Case("stale-belief-after-world-change",
                    [f"make a folder called {tag} on the desktop", "whats on my desktop", "(deleted externally)", "whats on my desktop"],
                    {"created it": made, "listed it while present": saw_it, "deletion took effect": gone,
                     "does not list it after deletion": tag not in t9.reply},
                    t7.replies + t8.replies + t9.replies, t7.acts + t8.acts + t9.acts,
                    t7.seconds + t8.seconds + t9.seconds, t7.model_calls + t8.model_calls + t9.model_calls))
    return out


# ------------------------------------------------------------- grounding quality


def dock_truth() -> list[str] | None:
    """Our own DOM read of the desktop page: independent of the agent's perceiver."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return None
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_context(viewport={"width": 1280, "height": 800}).new_page()
            page.goto("http://127.0.0.1:4391/?computer=ubuntu-2")
            page.wait_for_timeout(2500)
            names = page.evaluate(
                """() => {
                    const all = [...document.querySelectorAll('button[aria-label]')].map(b => ({
                        name: b.getAttribute('aria-label'),
                        r: b.getBoundingClientRect(),
                    })).filter(b => b.r.width > 8 && b.r.height > 8);
                    const byX = {};
                    for (const b of all) {
                        const k = Math.round(b.r.x / 6) * 6;
                        (byX[k] = byX[k] || []).push(b);
                    }
                    let best = [];
                    for (const k of Object.keys(byX)) if (byX[k].length > best.length) best = byX[k];
                    return best.sort((a, b) => a.r.y - b.r.y).map(b => b.name);
                }"""
            )
            browser.close()
            return names
    except Exception:  # noqa: BLE001 - no truth rather than a guessed truth
        return None


def grounding(chat: Chat) -> list[Case]:
    """Does the citation an answer offers actually support it, where we can check?"""
    out: list[Case] = []

    chat.say("my name is Priya")
    t = chat.say("what is my name")
    low = t.reply.lower()
    out.append(Case("told-fact-cites-being-told", ["my name is Priya", "what is my name"],
                    {"answer correct": "priya" in low, "cites that I told it": "told" in low},
                    t.replies, t.acts, t.seconds, t.model_calls))

    truth = dock_truth()
    t2 = chat.say("how many icons are in the sidebar")
    said = t2.reply
    number = re.search(r"\b(\d+)\b", said)
    checks = {"cites the screen": bool(re.search(r"seen on screen|on screen", said, re.I))}
    if truth is None:
        checks["count matches an independent DOM read"] = False
        note = "no independent truth available (playwright missing or page unreachable)"
    else:
        # the strip holds app launchers plus an app-grid button ("Show Applications"),
        # so "how many icons" has two defensible answers; score both and say so.
        apps_only = [n for n in truth if n.lower() != "show applications"]
        note = (f"independent DOM read: {len(truth)} buttons in the strip "
                f"({len(apps_only)} app launchers + {len(truth) - len(apps_only)} app-grid button): "
                f"{', '.join(truth)}")
        given = int(number[1]) if number else None
        checks["count matches the strip (app launchers + grid button)"] = given == len(truth)
        checks["count matches the app launchers alone"] = given == len(apps_only)
        named = [n for n in apps_only if n.lower() in said.lower()]
        checks["names at least 80% of the app launchers"] = len(named) >= 0.8 * len(apps_only)
    case = Case("screen-answer-matches-independent-read", ["how many icons are in the sidebar"], checks,
                t2.replies + [f"[{note}]"], t2.acts, t2.seconds, t2.model_calls,
                either_or=("count matches the strip (app launchers + grid button)",
                           "count matches the app launchers alone"))
    out.append(case)

    t3 = chat.say("what color is the display")
    said3 = t3.reply
    out.append(Case("pixel-answer-cites-pixels", ["what color is the display"],
                    {"cites looking at pixels": bool(re.search(r"pixel", said3, re.I)),
                     "gives a colour with proportions": bool(re.search(r"\d+%", said3))},
                    t3.replies + ["[truth of the colour itself not independently checked here]"], t3.acts, t3.seconds, t3.model_calls))
    return out


def run_all(chat: Chat) -> dict:
    suites = {
        "compositionality": compositionality(chat),
        "belief_revision": belief_revision(chat),
        "grounding": grounding(chat),
    }
    return {name: [asdict(c) | {"passed": c.passed} for c in cases] for name, cases in suites.items()}
