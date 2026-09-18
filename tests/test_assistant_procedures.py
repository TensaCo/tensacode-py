"""Procedures (data) must say exactly what the generators said, on the same fake machine.

The old motor programs are still in ``programs.py``; each case here runs both engines against
one fake shell and requires the same replies, the same commands, in the same order.
"""

from __future__ import annotations

import re

import pytest
import tensacode as tc

from examples.browser_agents.assistant import interpreter as I
from examples.browser_agents.assistant import procedure as L
from examples.browser_agents.assistant import procedures as PR
from examples.browser_agents.assistant import programs as P
from examples.browser_agents.assistant.language import Frame

HOME = "/home/agent"


class FakeShell:
    """Just enough of a file system to answer the commands the assistant actually types."""

    def __init__(self, tree: dict[str, str | None]):
        self.tree = dict(tree)  # path -> contents (str) or None for a directory
        self.commands: list[str] = []

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
                if k:
                    out.append(f"{k}|{len(self.tree.get(path) or '')}|{path}")
                else:
                    err.append(f"stat: cannot statx '{path}': No such file or directory")
            return P.Output("\n".join(out + err))
        if m := re.match(r"^ls -1pA '?([^']+)'?$", command):
            root = m[1].rstrip("/")
            names = set()
            for p in self.tree:
                if p.startswith(root + "/"):
                    rest = p[len(root) + 1:]
                    names.add(rest.split("/")[0] + ("/" if "/" in rest or self.tree[p] is None else ""))
            return P.Output("\n".join(sorted(names)))
        if m := re.match(r"^head -n \d+ '?([^']+)'?$", command) or re.match(r"^tail -n \d+ '?([^']+)'?$", command):
            return P.Output(self.tree.get(m[1]) or "")
        if m := re.match(r"^mkdir -p '?([^']+)'?$", command):
            self.tree[m[1]] = None
            return P.Output("")
        if m := re.match(r"^printf '%s\\n' '(.*)' (>>?) '?([^']+)'?$", command):
            text, mode, path = m[1], m[2], m[3]
            old = self.tree.get(path) or ""
            self.tree[path] = (old + text + "\n") if mode == ">>" else text + "\n"
            return P.Output("")
        if m := re.match(r"^touch '?([^']+)'?$", command):
            self.tree.setdefault(m[1], "")
            return P.Output("")
        if m := re.match(r"^rm (-r )?'?([^']+)'?$", command):
            path = m[2]
            gone = [p for p in self.tree if p == path or p.startswith(path.rstrip("/") + "/")]
            for p in gone:
                del self.tree[p]
            return P.Output("" if gone else f"rm: {path}: No such file or directory")
        if m := re.match(r"^mv '?([^']+)'? '?([^']+)'?$", command):
            src, dest = m[1], m[2]
            for p in [p for p in self.tree if p == src or p.startswith(src.rstrip("/") + "/")]:
                self.tree[dest + p[len(src):]] = self.tree.pop(p)
            return P.Output("")
        if m := re.match(r"^cp (-r )?'?([^']+)'? '?([^']+)'?$", command):
            src, dest = m[2], m[3]
            for p in [p for p in self.tree if p == src or p.startswith(src.rstrip("/") + "/")]:
                self.tree[dest + p[len(src):]] = self.tree[p]
            return P.Output("")
        if m := re.match(r"^wc -(\w) < '?([^']+)'?$", command):
            body = self.tree.get(m[2]) or ""
            return P.Output(str(len(body.split()) if m[1] == "w" else len(body.splitlines())))
        if m := re.match(r"^du -sh '?([^']+)'?$", command):
            return P.Output(f"4.0K\t{m[1]}")
        if command == "date":
            return P.Output("Thu Jan  1 10:00:00 UTC 2026")
        if command == "whoami":
            return P.Output("agent")
        if command == "nproc":
            return P.Output("8")
        if m := re.match(r"^which '?([^']+)'?$", command):
            return P.Output(f"/usr/bin/{m[1]}" if m[1] == "git" else f"which: {m[1]}: not found")
        if m := re.match(r"^find '?([^']+)'? -name '?([^']+)'?.*", command):
            root, pattern = m[1], m[2].replace("*", ".*")
            return P.Output("\n".join(p for p in sorted(self.tree) if p.startswith(root) and re.fullmatch(pattern, p.rsplit("/", 1)[-1])))
        if command.startswith("grep -ril"):
            needle = re.search(r"grep -ril (?:'([^']*)'|(\S+))", command)
            needle = needle[1] if needle[1] is not None else needle[2]
            return P.Output("\n".join(p for p, body in sorted(self.tree.items()) if body and needle.lower() in body.lower()))
        if command.startswith("cd "):
            return P.Output("")
        if command.startswith("echo "):
            return P.Output(command[5:].strip("'"))
        return P.Output("")


def run_generator(act: str, slots: dict, shell: FakeShell, ctx: P.Context, answers: list[str]) -> list[str]:
    program = P.PROGRAMS[act](Frame(act, slots.get("words", act), slots), ctx)
    said, value, answers = [], None, list(answers)
    while True:
        try:
            action = program.send(value)
        except StopIteration:
            return said
        value = None
        if isinstance(action, P.Say):
            said.append(action.text)
        elif isinstance(action, P.Ask):
            said.append(action.question)
            value = Frame("choose", answers.pop(0)) if answers else Frame("cancel", "no")
        elif isinstance(action, P.Run):
            value = shell.run(action.command)
        elif isinstance(action, P.Focus):
            ctx.focus, ctx.focus_kind = action.path, action.kind
            ctx.known[action.path] = action.kind
        elif isinstance(action, P.OpenApp):
            value = True
        elif isinstance(action, P.Look):
            value = P.Seen((), ())


def run_procedure(act: str, slots: dict, shell: FakeShell, ctx: P.Context, answers: list[str]) -> tuple[list[str], tc.Store]:
    mind = tc.Store()
    req = tc.Ref("request:1.0")
    said, answers = [], list(answers)

    def say(mind_, req_, text, cycle):
        said.append(text)
        return I.Thought()

    host = I.Host(say=say, find=lambda pid: PR.BY_ID.get(pid))
    env = {"slot": dict(slots), "words": slots.get("words", act), "act": act, "turn": 1,
           "cwd": ctx.cwd, "focus": ctx.focus, "focus_kind": ctx.focus_kind, "known": dict(ctx.known)}
    I.begin(mind, req, PR.BY_ACT[act], env, 0)
    value = None
    for cycle in range(200):
        I.advance(mind, req, value, cycle, host)
        status = I.one(mind, req, "status")
        if status in ("done", "failed"):
            return said, mind
        doing = I.one(mind, req, "doing")
        if doing is None:
            return said, mind
        kind = I.one(mind, doing, "kind")
        if kind == "run":
            value = shell.run(I.one(mind, doing, "command"))
        elif kind == "ask":
            value = Frame("choose", answers.pop(0)) if answers else Frame("cancel", "no")
        elif kind == "look":
            value = P.Seen((), ())
        elif kind == "open_app":
            value = True
        else:
            value = True
    raise AssertionError("procedure did not finish")


TREE = {
    f"{HOME}/Desktop": None,
    f"{HOME}/Desktop/notes.txt": "hello\nworld\n",
    f"{HOME}/Desktop/recipes": None,
    f"{HOME}/Desktop/recipes/pasta.txt": "boil water\n",
    f"{HOME}/Documents": None,
    f"{HOME}/Documents/report.pdf": "dns and dashboards\n",
}

CASES = [
    ("list", {"place": "~/Desktop"}, [], None),
    ("list", {"place": "~/Desktop", "count": True}, [], None),
    ("list", {"place": "~/Desktop/nope"}, [], None),
    ("read", {"target": "~/Desktop/notes.txt"}, [], None),
    ("read", {"target": "~/Desktop/recipes"}, [], None),       # a folder: falls through to a listing
    ("read", {"target": "~/Desktop/missing.txt"}, [], None),
    ("create_folder", {"name": "plans", "place": "~/Desktop"}, [], None),
    ("create_folder", {"name": "recipes", "place": "~/Desktop"}, [], None),   # already there
    ("create_folder", {}, [], None),
    ("create_file", {"name": "todo.md", "place": "~/Desktop", "text": "first"}, [], None),
    ("create_file", {"name": "notes.txt", "place": "~/Desktop"}, [], None),   # already there
    ("create_file", {"name": "x.txt", "place": "~/Desktop/nowhere"}, [], None),
    ("write", {"target": "~/Desktop/notes.txt", "text": "third", "append": True}, [], None),
    ("write", {"target": "~/Desktop/notes.txt", "text": "only", "append": False}, [], None),
    ("write", {"target": "~/Desktop/recipes", "text": "x", "append": False}, [], None),
    ("delete", {"target": "~/Desktop/notes.txt"}, [], None),
    ("delete", {"target": "~/Desktop/recipes"}, [], None),
    ("delete", {"target": "~/Desktop/gone.txt"}, [], None),
    ("delete", {"target": "~/Desktop"}, [], "delete everything on my desktop"),
    ("rename", {"target": "~/Desktop/notes.txt", "new_name": "ideas.txt"}, [], None),
    ("rename", {"target": "~/Desktop/notes.txt", "new_name": "pasta.txt"}, [], None),
    ("rename", {"target": "~/Desktop/notes.txt"}, [], None),
    ("move", {"target": "~/Desktop/notes.txt", "dest": "~/Documents"}, [], None),
    ("copy", {"target": "~/Desktop/notes.txt", "dest": "~/Documents"}, [], None),
    ("find", {"pattern": "*.txt", "place": "~"}, [], None),
    ("find", {"pattern": "*.zzz", "place": "~"}, [], None),
    ("grep", {"needle": "dns", "place": "~"}, [], None),
    ("grep", {"needle": "absent", "place": "~"}, [], None),
    ("count", {"target": "~/Desktop/notes.txt", "unit": "lines"}, [], None),
    ("count", {"target": "~/Desktop/notes.txt", "unit": "words"}, [], None),
    ("size", {"target": "~/Documents"}, [], None),
    ("info", {"topic": "date"}, [], None),
    ("info", {"topic": "user"}, [], None),
    ("info", {"topic": "cpus"}, [], None),
    ("which", {"program": "git"}, [], None),
    ("which", {"program": "cowsay"}, [], None),
    ("cd", {"target": "~/Documents"}, [], None),
    ("open_app", {"app": "Firefox"}, [], None),
    ("run", {"command": "echo 'hi there'"}, [], None),
    ("greet", {}, [], None),
    ("thanks", {}, [], None),
    ("help", {}, [], None),
    ("unknown", {}, [], "make me a sandwich"),
    # a bare name that exists in two places: the same question, and the same answer taken
    ("read", {"target": "pasta.txt"}, ["1"], None),
    ("list", {"place": "recipes"}, [], None),
]


@pytest.mark.parametrize("act,slots,answers,words", CASES, ids=[f"{c[0]}:{sorted(c[1])}" for c in CASES])
def test_procedure_says_what_the_generator_said(act, slots, answers, words):
    slots = dict(slots)
    if words:
        slots["words"] = words
    old_shell, new_shell = FakeShell(TREE), FakeShell(TREE)
    old_said = run_generator(act, slots, old_shell, P.Context(cwd=HOME), answers)
    new_said, mind = run_procedure(act, slots, new_shell, P.Context(cwd=HOME), answers)
    assert new_said == old_said, f"replies differ\nold: {old_said}\nnew: {new_said}"
    # the procedures may do LESS work (the kind check happens once, not once per delegation), never more
    assert len(new_shell.commands) <= len(old_shell.commands), f"more commands than before\nold: {old_shell.commands}\nnew: {new_shell.commands}"
    assert [c for c in new_shell.commands if not c.startswith("stat -c")] == [c for c in old_shell.commands if not c.startswith("stat -c")], \
        f"acting commands differ\nold: {old_shell.commands}\nnew: {new_shell.commands}"
    assert new_shell.tree == old_shell.tree  # and the machine ends up in the same state


def test_the_graph_shows_the_steps_a_request_took():
    shell = FakeShell(TREE)
    said, mind = run_procedure("create_folder", {"name": "plans", "place": "~/Desktop"}, shell, P.Context(cwd=HOME), [])
    trace = I.trace_of(mind, tc.Ref("request:1.0"))
    assert said == ["Created the folder ~/Desktop/plans."]
    assert {t.split("[")[0] for t in trace} == {"create_folder", "folder_for_new", "resolve"}, trace
    assert sum(" run " in f" {t} " for t in trace) == 3  # look before acting, act, look again — all three in the graph
    assert any("call" in t for t in trace)  # including the sub-procedure it called to settle the folder
    assert int(trace[0].split("[")[1].split("]")[0]) > 0  # step 0 was skipped: its guard did not hold
    # the reply rests on the command's output, which explain() can walk back to
    frames = [r.claim.subject for r in mind.claims(predicate="bound")]
    assert frames, "bindings should record what they were derived from"
    from tensacode.cognition import explain

    lines = "\n".join(explain(mind, mind.claims(predicate="bound")[0].id))
    assert "from:" in lines


def test_every_procedure_is_checkable_and_round_trips():
    for proc in PR.PROCEDURES:
        proc.check()
        assert L.Procedure.from_json(proc.to_json()) == proc


def test_a_broken_procedure_is_refused_not_run():
    with pytest.raises(L.ProcedureError):
        L.Procedure(id="bad", steps=[{"do": "dance"}]).check()
    with pytest.raises(L.ProcedureError):
        L.Procedure(id="bad2", steps=[{"do": "compute", "prim": "no_such_primitive"}]).check()
    with pytest.raises(L.ProcedureError):
        L.Procedure(id="bad3", steps=[{"when": {"nonsense": 1}, "do": "stop"}]).check()


def test_the_learner_is_a_procedure_too():
    """The teacher-guided loop runs on the same interpreter: its steps and state are claims."""
    import json

    from examples.browser_agents.assistant import learning as LN

    replies = [
        json.dumps({"thought": "open the editor", "do": "open_app", "app": "Text Editor"}),
        json.dumps({"thought": "type it", "do": "run", "command": "echo hello > /tmp/learned.txt", "effect": True}),
        json.dumps({"thought": "finished", "do": "done", "reply": "wrote the file"}),
        json.dumps({"pattern": "jot {text}", "slots": {"text": "hello"}}),
    ]
    shell = FakeShell(TREE)
    mind, req, said = tc.Store(), tc.Ref("request:9.0"), []
    host = I.Host(say=lambda m, r, t, c: (said.append(t), I.Thought())[1],
                  find=lambda pid: PR.BY_ID.get(pid) or LN.LEARNER_PROCS.get(pid))
    I.begin(mind, req, LN.LEARNER_PROCS["learn"], {"slot": {}, "words": "jot hello", "act": "unknown",
                                                   "request": "jot hello", "note": "", "cwd": HOME,
                                                   "focus": None, "focus_kind": None, "known": {}}, 0)
    value = None
    for cycle in range(120):
        I.advance(mind, req, value, cycle, host)
        if I.one(mind, req, "status") in ("done", "failed"):
            break
        doing = I.one(mind, req, "doing")
        kind = I.one(mind, doing, "kind") if doing is not None else None
        if kind == "run":
            value = shell.run(I.one(mind, doing, "command"))
        elif kind == "look":
            value = P.Seen((("button", "Text Editor"),), (("Text Editor", "untitled"),))
        elif kind == "teach":
            value = replies.pop(0) if replies else None
        elif kind == "open_app":
            value = True
        else:
            break
    assert any("work it out once with my local teacher model" in t for t in said), said
    assert any("echo hello" in t for t in said), said  # it reports what it observed itself doing
    trace = I.trace_of(mind, req)
    assert {t.split("[")[0] for t in trace} >= {"learn", "learn_turn"}, trace
    assert any("teach" in t for t in trace)  # the model calls are steps in the graph like any other


def test_a_learned_skill_is_the_same_kind_of_procedure(tmp_path, monkeypatch):
    """Learned and hand-written procedures run on one interpreter; replay adopts a skill that works on a new case."""
    from examples.browser_agents.assistant import learning as LN

    monkeypatch.setattr(LN.LIBRARY, "path", tmp_path / "skills.json")
    monkeypatch.setattr(LN.LIBRARY, "skills", [])
    trace = [
        {"do": "open_app", "app": "Slack", "effect": False, "ok": True, "screen": []},
        {"do": "fill", "label": "Message agent-runs on slack", "text": "@Ada Kernel I'm happy", "effect": False, "ok": True,
         "screen": ["Ada Kernel 7:45 AM"]},
        {"do": "click", "label": "Send message", "effect": True, "ok": True, "screen": ["Ada Kernel 7:45 AM"]},
    ]
    skill, why = LN.compile_skill("tell ada im happy", trace, "Posted “@{person_name} {message|sentence}”.",
                                  "tell {person} {message}", {"person": "ada", "message": "im happy"})
    assert skill is not None, why
    LN.LIBRARY.add(skill)
    proc = LN.procedure_for_skill(skill.id)
    proc.check()
    assert proc.author == "learned-from-trace"

    mind, req, said, typed = tc.Store(), tc.Ref("request:5.0"), [], []
    host = I.Host(say=lambda m, r, t, c: (said.append(t), I.Thought())[1],
                  find=lambda pid: PR.BY_ID.get(pid) or (proc if pid == proc.id else None),
                  on_finish=LN.record_outcome)
    I.begin(mind, req, proc, {"slot": {"person": "maya", "message": "the build is green"}, "words": "tell maya the build is green",
                              "act": "learned", "cwd": HOME, "focus": None, "focus_kind": None, "known": {}}, 0)
    value = None
    for cycle in range(60):
        I.advance(mind, req, value, cycle, host)
        if I.one(mind, req, "status") in ("done", "failed"):
            break
        doing = I.one(mind, req, "doing")
        kind = I.one(mind, doing, "kind") if doing is not None else None
        if kind == "look":
            value = P.Seen((("button", "Send message"),), (("Slack", "Maya Chen 8:01 AM"), ("Slack", "@Maya Chen The build is green")))
        elif kind == "fill":
            typed.append(I.one(mind, doing, "text"))
            value = True
        else:
            value = True
    assert typed == ["@Maya Chen The build is green"]  # the name came off the screen, the message from the request
    assert "no model used" in said[0]
    assert LN.LIBRARY.skills[0].status == "adopted"  # it worked on values it was not learned from
    assert [h[1] for h in LN.LIBRARY.skills[0].history] == ["adopted"]
