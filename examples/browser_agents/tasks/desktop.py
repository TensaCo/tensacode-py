"""A desktop chore on a simulated Ubuntu computer, as a mind.

A note on the desktop describes a project to set up. The agent opens Terminal from the
dock, finds and reads the note, parses it with a forgiving grammar, turns it into a
``tc.Plan`` of shell steps, checks the plan's structure, runs each step in the terminal,
checks each step's output, and verifies the result by reading it back.

The machine is a `computerworld <https://github.com/JacobFV/computerworld>`_ world: one
deterministic Rust runtime, no browser and no simulator server. The note is part of the
world definition, so every episode starts from exactly its definition plus its seed and
cannot be polluted by an earlier run. Scoring reads the world through a privileged session
the agent does not hold.
"""

from __future__ import annotations

import random
import time
import re
from dataclasses import dataclass

import tensorcode as tc
from tensorcode.backends.builtin import IN_PROCESS
from tensorcode.cognition import Rule

from ..browser import is_computerworld_terminal_input
from ..mind import BY_PRIORITY, Enter, Escalate, Finish, MindSpec, Note, Press, Wait, controls, knowledge, one
from ..worlds import note_world
from ..worlds.runtime import expand

MACHINE = "dev"
USER = "agent"
HOME = f"/home/{USER}"
V = tc.Var
TASK, PLAN, EPISODE = tc.Ref("task:note"), tc.Ref("plan:current"), tc.Ref("episode:current")
PROMPT = re.compile(r"^(\S+@\S+?):(\S*)\$\s?(.*)$", re.S)


# ------------------------------------------------------------- the world


WORDS = (["ledger", "orbit", "harbor", "signal", "copper", "meadow", "atlas", "ember"], ["tools", "notes", "sync", "lab", "desk", "board"])
TODO = ["draft the schema", "write smoke tests", "wire up CI", "sketch the CLI", "collect sample data", "review naming", "add a changelog"]


NOTES: dict[int, str] = {}  # seed -> a note someone typed into the live viewer


def chore(seed: int) -> dict:
    if seed in NOTES:
        task = read_note(NOTES[seed])
        known = isinstance(task, ProjectTask)
        return {"name": task.name if known else None, "title": task.title if known else None, "items": list(task.items) if known else [],
                "message": task.message if known else None, "note": NOTES[seed], "path": f"{HOME}/Desktop/task-{seed}.txt", "custom": True,
                "root": expand(f"{task.parent}/{task.name}", USER) if known else None}
    r = random.Random(seed)
    name = f"{r.choice(WORDS[0])}-{r.choice(WORDS[1])}-{seed}"
    title = " ".join(w.capitalize() for w in name.split("-")[:2])
    items = r.sample(TODO, r.randint(2, 3))
    message = r.choice(["initial commit", "scaffold project", "start " + name.split("-")[0]])
    note = r.choice([
        f"Set up a project called {name} under ~/Projects. Create README.md containing the line '{title}' and a notes folder with todo.txt listing: {'; '.join(items)}. Make it a git repository and commit everything with the message '{message}'.",
        f"New project please: {name} (in ~/Projects). README.md should say '{title}'. Put a todo.txt in notes/ with these items: {'; '.join(items)}. Then git init and commit it all as '{message}'.",
    ])
    return {"name": name, "title": title, "items": items, "message": message, "note": note,
            "path": f"{HOME}/Desktop/task-{seed}.txt", "root": f"{HOME}/Projects/{name}"}


def world_definition(seed: int) -> dict:
    """This episode's machine: the chore's note sitting on the Desktop."""
    return note_world(chore(seed)["note"], seed)


def score(world, seed: int) -> dict:
    """Read the finished machine through a privileged session: files, then the repository."""
    want = chore(seed)
    if want["name"] is None:  # a typed note the grammar cannot read: nothing to check but that nothing ran
        return {"task": "desktop", "items": 0, "correct": 0, "duplicates": 0, "checks": {}, "want": want}
    root = want["root"]
    readme, todo = world.read(f"{root}/README.md"), world.read(f"{root}/notes/todo.txt")
    commits = world.commits(root)
    checks = {
        "readme": readme is not None and readme.strip() == want["title"],
        "todo": todo is not None and [x.strip() for x in todo.strip().splitlines()] == want["items"],
        # the newest commit carries the message and nothing is left uncommitted
        "commit": bool(commits) and commits[0][0] == want["message"] and not world.pending(root),
    }
    return {"task": "desktop", "items": 3, "correct": sum(checks.values()), "duplicates": max(0, len(commits) - 1),
            "checks": checks, "want": want, "state_hash": world.state_hash()}


# ------------------------------------------------------ language: the note


@dataclass(frozen=True)
class NoteText:
    text: str


@dataclass(frozen=True)
class ProjectTask:
    name: str
    parent: str
    title: str
    items: tuple[str, ...]
    message: str


@tc.implementation("parse", name="project-note-grammar", version="1", accepts=lambda r: isinstance(r.subject, NoteText) and r.target is ProjectTask, profile=IN_PROCESS)
def parse_note(request: tc.Request) -> ProjectTask | tc.Unknown:
    return read_note(request.subject.text)


def read_note(text: str) -> ProjectTask | tc.Unknown:
    """Island grammar: find each slot wherever it appears, tolerate filler and punctuation."""
    t = " ".join(text.split())
    name = re.search(r"(?:project called|project please:)\s+([a-z0-9][\w-]*)", t)
    parent = re.search(r"(?:under|in)\s+(~/[\w/-]+)", t)
    title = re.search(r"README(?:\.md)?\s+(?:containing the line|should say|saying|that says|with)\s+'([^']+)'", t, re.I)
    items = re.search(r"(?:listing|items|todos?|to-?dos?)\s*:\s+(.+?)(?:\.\s|\.?$)", t)
    message = re.search(r"(?:with the message|with message|message|as)\s+'([^']+)'", t, re.I)
    missing = [k for k, m in {"name": name, "parent": parent, "title": title, "items": items, "message": message}.items() if not m]
    if missing:
        return tc.Unknown("note_incomplete", f"could not find: {', '.join(missing)}")
    return ProjectTask(name[1], parent[1], title[1], tuple(x.strip() for x in items[1].split(";") if x.strip()), message[1])


# ------------------------------------------------ program graph: the plan


@tc.action(effect="external", idempotent=False)
@dataclass(frozen=True)
class RunCommand:
    command: str


def plan_for(task: ProjectTask) -> tc.Plan:
    root = f"{task.parent}/{task.name}"
    q = lambda s: "'" + s.replace("'", "") + "'"  # noqa: E731
    steps = [
        tc.Step("mkdir", RunCommand(f"mkdir -p {root}/notes")),
        tc.Step("cd", RunCommand(f"cd {root}"), needs=("mkdir",)),
        tc.Step("readme", RunCommand(f"echo {q(task.title)} > README.md"), needs=("cd",)),
        tc.Step("todo", RunCommand(" && ".join(f"echo {q(item)} {'>' if i == 0 else '>>'} notes/todo.txt" for i, item in enumerate(task.items))), needs=("cd",)),
        tc.Step("init", RunCommand("git init"), needs=("cd",)),
        tc.Step("add", RunCommand("git add ."), needs=("readme", "todo", "init")),
        tc.Step("commit", RunCommand(f"git commit -m {q(task.message)}"), needs=("add",)),
        tc.Step("verify", RunCommand("cat README.md notes/todo.txt && git log --oneline"), needs=("commit",)),
    ]
    return tc.Plan(tuple(steps), rationale=f"set up {root}")




# ------------------------------------------------------ spontaneous thoughts


def transcript(mind: tc.Store) -> list[tuple[str, str]]:
    """(command, output) blocks from the terminal, in screen order."""
    if one(mind, EPISODE, "started") is None:
        return []  # nothing on screen belongs to this episode until the agent's own first command
    lines = sorted((mind.get(r.claim.subject).box[1], r.claim.object) for r in mind.claims(predicate="reads") if r.claim.subject.id.startswith("text:Terminal#"))
    blocks: list[tuple[str, str]] = []
    for _, text in lines:
        if m := PROMPT.match(text):
            blocks.append((m[3].strip(), ""))
        elif blocks:
            blocks[-1] = (blocks[-1][0], (blocks[-1][1] + "\n" + text).strip())
    marker = one(mind, EPISODE, "marker")
    starts = [i for i, (cmd, _) in enumerate(blocks) if cmd == marker]
    # after our own marker; if it scrolled away, everything still visible is from this episode
    return blocks[starts[-1]:] if starts else blocks


def _read_terminal(b, mind):
    claims = []
    for i, (command, output) in enumerate(transcript(mind)):
        if command.startswith("ls ~/Desktop") and (m := re.search(r"task-\d+\.txt", output)):
            claims.append((tc.Claim(TASK, "note_file", f"~/Desktop/{m[0]}"), None))
        if command.startswith("cat ~/Desktop/task-") and output and not one(mind, TASK, "project"):
            task = tc.parse(NoteText(output), ProjectTask)
            if isinstance(task, ProjectTask):
                claims += [(tc.Claim(TASK, k, getattr(task, k)), None) for k in ("name", "parent", "title", "items", "message")] + [(tc.Claim(TASK, "project", task.name), None)]
            else:
                claims.append((tc.Claim(TASK, "unreadable", f"{task.reason}: {task.detail}"), None))
    for r in mind.claims(predicate="command"):  # remember step completions; lines scroll away
        if one(mind, r.claim.subject, "done") is None:
            ran, output = _done(mind, r.claim.object)
            if ran:
                claims += [(tc.Claim(r.claim.subject, "done", True), None), (tc.Claim(r.claim.subject, "output", output or ""), None)]
    if claims:
        yield knowledge(claims, "obs:terminal", "read-terminal")


# one premise (the episode), reacting to any screen change: output appearing *or* a spinner disappearing
RULES = [Rule("read_terminal_transcript", ((EPISODE, "started", True),), _read_terminal, reacts_to="any_change")]


# ------------------------------------------------------------ deliberation

ERROR = re.compile(r"no such file|not found|fatal:|error:|permission denied|nothing to commit|unknown option", re.I)


def _done(mind: tc.Store, command: str) -> tuple[bool, str | None]:
    """Has this command run (a later prompt exists), and what did it print?"""
    blocks = transcript(mind)
    for i in range(len(blocks) - 1, -1, -1):
        if blocks[i][0] == command:
            pending = any("…" == out.strip() for _, out in blocks[i:])  # the terminal's placeholder while a command runs
            return i < len(blocks) - 1 and not pending, None if pending else blocks[i][1]
    return False, None


def intentions(mind: tc.Store) -> list[object]:
    shell = [ref for ref in controls(mind, role="textbox") if is_computerworld_terminal_input(mind.get(ref))]
    if len(shell) != 1:
        shell = []
    if not shell:
        dock = controls(mind, role="button", label="Terminal")
        return [Press(dock[0], "open Terminal from the dock", priority=90)] if dock else [Wait(50, "waiting for the desktop", priority=1)]
    run = lambda cmd, why, p: Enter(shell[0], cmd, why, priority=p, submit=True)  # noqa: E731
    if one(mind, TASK, "unreadable"):
        return [Escalate(f"task note unreadable: {one(mind, TASK, 'unreadable')}")]
    if one(mind, TASK, "note_file") is None:
        if one(mind, EPISODE, "started") is None:
            # a fresh world per episode, so nothing older is on screen to scope out
            marker = "ls ~/Desktop"
            return [Enter(shell[0], marker, "look for a task note on the Desktop", priority=80, submit=True,
                          records=(tc.Claim(EPISODE, "started", True), tc.Claim(EPISODE, "marker", marker), tc.Claim(EPISODE, "issued_at", time.monotonic())))]
        marker = one(mind, EPISODE, "marker")
        ran, _ = _done(mind, marker)
        if ran:
            return [Escalate("no task note on the Desktop")]
        if time.monotonic() - one(mind, EPISODE, "issued_at") > 3 and not _done(mind, marker)[1] and not any(c == marker for c, _ in transcript(mind)):
            return [Enter(shell[0], marker, "terminal did not take the command; type it again", priority=79, submit=True, records=(tc.Claim(EPISODE, "issued_at", time.monotonic() + 1e6),))]
        return [Wait(20, "waiting for ls", priority=2)]
    if one(mind, TASK, "project") is None:
        cmd = f"cat {one(mind, TASK, 'note_file')}"
        ran, _ = _done(mind, cmd)
        return [run(cmd, "read the task note", 75)] if not ran else [Wait(30, "parsing the note", priority=1)]

    task = ProjectTask(*(one(mind, TASK, k) for k in ("name", "parent", "title", "items", "message")))
    plan = plan_for(task)
    if one(mind, PLAN, "checked") is None:
        order = tc.plan_order(plan)
        if isinstance(order, tc.Verdict):
            return [Escalate(f"plan does not hold together: {'; '.join(order.reasons)}")]
        return [Note((tc.Claim(PLAN, "checked", len(plan.steps)),), (), f"plan checked: {len(plan.steps)} steps for {task.parent}/{task.name}", priority=70)]
    for step in plan.steps:
        ref = tc.Ref(f"step:{step.id}")
        if one(mind, ref, "done") is None:
            if one(mind, ref, "command") is not None:
                return [Wait(20, f"waiting for `{step.id}`", priority=2)]
            return [Enter(shell[0], step.action.command, f"step {step.id}: {step.action.command[:70]}", priority=60, submit=True, records=(tc.Claim(ref, "command", step.action.command),))]
        output = one(mind, ref, "output", "")
        if step.id == "commit" and "nothing to commit" in output:
            continue  # already committed identically (a rerun): the read-back below decides
        if ERROR.search(output):
            return [Escalate(f"step {step.id} failed: {output.splitlines()[0][:120]}")]
    shown = one(mind, tc.Ref(f"step:{plan.steps[-1].id}"), "output", "")
    ok = task.title in (shown or "") and all(item in (shown or "") for item in task.items) and task.message in (shown or "")
    return [Finish("verified by reading back files and git log", priority=100)] if ok else [Escalate(f"verification did not match: {shown!r:.120}")]


SPEC = MindSpec("desktop", RULES, intentions, BY_PRIORITY, max_cycles=80)
BINDINGS = [parse_note]
world = world_definition  # the harness builds one machine per episode from this
