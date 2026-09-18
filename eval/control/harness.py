"""Drive the assistant's control layer with a stubbed body.

What is real here: ``hear``, ``intentions`` (which request to pursue, suspend, resume or
arbitrate between), the decoders, the procedure interpreter, the procedures themselves, and the
claims all of it writes. What is stubbed: the screen and the terminal — a command is answered by
a fake file system rather than by typing into Seed. So these measurements are about *control
decisions*, and they say nothing about perception or typing.

Provenance: environment ours (fake shell, adapted from tests/test_assistant_procedures.py),
grader ours (the assertions and counts below), held out where a case says so.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

import tensacode as tc

from examples.browser_agents.assistant import agent
from examples.browser_agents.assistant import interpreter as I
from examples.browser_agents.assistant import programs as P

HOME = "/home/agent"


class FakeShell:
    """Just enough of a file system to answer the commands the assistant actually types."""

    def __init__(self, tree: dict[str, str | None], *, fail: dict[str, int] | None = None):
        self.tree = dict(tree)
        self.commands: list[str] = []
        self.fail = dict(fail or {})  # command prefix -> how many times to fail before working

    def kind(self, path: str) -> str | None:
        if path in self.tree:
            return "directory" if self.tree[path] is None else "regular file"
        if any(p.startswith(path.rstrip("/") + "/") for p in self.tree):
            return "directory"
        return None

    def run(self, command: str) -> P.Output:
        self.commands.append(command)
        for prefix, left in list(self.fail.items()):
            if command.startswith(prefix) and left > 0:
                self.fail[prefix] = left - 1
                # phrased the way a real shell phrases it, because the assistant's own
                # output_facts decides what an error is by matching that phrasing
                return P.Output(f"{command.split()[0]}: cannot read input: Input/output error")
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
            gone = [p for p in self.tree if p == m[2] or p.startswith(m[2].rstrip("/") + "/")]
            for p in gone:
                del self.tree[p]
            return P.Output("" if gone else f"rm: {m[2]}: No such file or directory")
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
        if m := re.match(r"^find '?([^']+)'? -name '?([^']+)'?.*", command):
            root, pattern = m[1], m[2].replace("*", ".*")
            return P.Output("\n".join(p for p in sorted(self.tree) if p.startswith(root) and re.fullmatch(pattern, p.rsplit("/", 1)[-1])))
        if command.startswith("cd ") or command.startswith("clear"):
            return P.Output("")
        if command.startswith("echo "):
            return P.Output(command[5:].strip("'"))
        return P.Output("")


@dataclass
class Turn:
    text: str
    replies: list[str] = field(default_factory=list)
    commands: list[str] = field(default_factory=list)
    intentions: list[str] = field(default_factory=list)
    cycles: int = 0
    seconds: float = 0.0


class Conversation:
    """A conversation whose control decisions are real and whose hands are fake."""

    def __init__(self, tree: dict[str, str | None], *, chunks=None, fail: dict[str, int] | None = None,
                 answers: list[str] | None = None, budget: int = 400):
        self.mind = agent.new_mind()
        self.shell = FakeShell(tree, fail=fail)
        self.answers = list(answers or [])
        self.budget = budget
        self.turns: list[Turn] = []
        # the assistant's own host, because that is the one its decoders use: passing a second
        # host here would have half a turn deliberating and half of it automatized
        self.host = agent.HOST
        self.host.chunks = chunks
        self._replies: list[str] = []
        agent.BODY.turn = 0

    def _pending(self) -> tuple[tc.Ref, tc.Ref, str] | None:
        for req in agent.open_requests(self.mind):
            act = agent.one(self.mind, req, "doing")
            if act is not None and agent.one(self.mind, req, "status") == "running":
                return req, act, str(agent.one(self.mind, act, "kind"))
        return None

    def _serve_body(self, req: tc.Ref, act: tc.Ref, kind: str, cycle: int) -> None:
        """Satisfy whatever the procedure asked the body for."""
        if kind == "run":
            value = self.shell.run(str(agent.one(self.mind, act, "command")))
        elif kind == "look":
            value = P.Seen((), ())
        elif kind == "open_app":
            value = True
        elif kind == "sample":
            value = {"colors": [[255, 255, 255, 1.0]], "mean": [255, 255, 255], "box": [0, 0, 1, 1], "unavailable": None}
        else:
            value = True
        I.advance(self.mind, req, value, cycle, self.host)

    def say(self, text: str) -> Turn:
        agent.BODY.turn += 1
        self._replies = []
        agent.BODY.say = self._replies.append  # everything the assistant says goes through BODY.say
        turn = Turn(text)
        before = len(self.shell.commands)
        started = time.perf_counter()
        agent.hear(self.mind, text, agent.BODY.turn)
        for cycle in range(self.budget):
            pending = self._pending()
            if pending is not None:  # the body owes the procedure an answer: give it one
                self._serve_body(*pending, cycle)
                turn.cycles += 1
                continue
            options = agent.intentions(self.mind)
            best = max(options, key=lambda i: getattr(i, "priority", 0.0))
            turn.intentions.append(type(best).__name__)
            turn.cycles += 1
            if type(best).__name__ == "Finish":
                break
            if isinstance(best, agent.Advance) and getattr(best, "value", None) is None and self._awaiting_answer(best):
                pass
            decoder = agent.SPEC.decoders.get(type(best))
            if decoder is None:  # a body-level intention with no pending act: nothing to serve, stop
                break
            decoder(best, self.mind, None, cycle)
            if isinstance(best, (agent.Suspend, agent.Resume)):
                continue
            # a procedure that asked a question stops the turn unless an answer is scripted
            asked = self._question_open()
            if asked is not None and self.answers:
                reply = self.answers.pop(0)
                agent.hear(self.mind, reply, agent.BODY.turn)  # the answer arrives in the same turn
        turn.replies = list(self._replies)
        turn.commands = self.shell.commands[before:]
        turn.seconds = time.perf_counter() - started
        self.turns.append(turn)
        return turn

    def _awaiting_answer(self, intention) -> bool:
        return agent.one(self.mind, intention.request, "status") == "awaiting"

    def _question_open(self) -> tc.Ref | None:
        return agent.one(self.mind, agent.ME, "awaiting")

    # ----------------------------------------------------------------- reading
    def status(self, req: str) -> str:
        return str(agent.one(self.mind, tc.Ref(req), "status"))

    def claims(self) -> int:
        return len(self.mind._claims)

    def requests(self) -> list[str]:
        return [r.claim.subject.id for r in self.mind.claims(predicate="status")]


TREE = {
    f"{HOME}/Desktop": None,
    f"{HOME}/Desktop/notes.txt": "hello\nworld\n",
    f"{HOME}/Desktop/keep": None,
    f"{HOME}/Desktop/keep/a.txt": "a\n",
    f"{HOME}/Desktop/keep/b.txt": "b\n",
    f"{HOME}/Documents": None,
    f"{HOME}/Documents/report.pdf": "dns and dashboards\n",
}


def step_runs(conv: "Conversation") -> dict[tuple[str, str, int, int], int]:
    """How many times each step of each goal executed.

    A step's identity is (what the user asked, procedure, frame depth, program counter), read
    back off the store. Goals are keyed by the *words*, so a request restated after being
    dropped counts as the same goal — which is the point: re-running its steps is redone work,
    however new the request object is. Frame depth is part of the key so that a procedure which
    recurses (project_step) is not mistaken for one that repeated itself.
    """
    counts: dict[tuple[str, str, int, int], int] = {}
    for record in conv.mind.claims(predicate="index"):
        step = record.claim.subject  # step:frame:request:N#d@pc
        frame = agent.one(conv.mind, step, "of")
        proc = agent.one(conv.mind, step, "procedure")
        if frame is None or proc is None:
            continue
        req_id, _, depth = frame.id[len("frame:"):].partition("#")
        words = agent.one(conv.mind, tc.Ref(req_id), "words") or req_id
        key = (str(words), str(proc), int(depth or 0), int(record.claim.object))
        counts[key] = counts.get(key, 0) + 1
    return counts


def redone(conv: "Conversation") -> dict[str, int]:
    """Per goal, how many step executions were repetitions of a step already done for it."""
    out: dict[str, int] = {}
    for (words, _proc, _depth, _pc), n in step_runs(conv).items():
        if n > 1:
            out[words] = out.get(words, 0) + n - 1
    return out
