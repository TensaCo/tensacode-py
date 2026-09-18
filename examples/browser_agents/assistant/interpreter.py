"""Walk a procedure with its state in the mind, not in Python.

Everything the walk needs is a claim: which procedure a request is running, where it is in
it (``pc``), what it has bound (``env:name``), what each step did (``step:… result``), and
which claims a step's conclusion rests on. So "why did you type that?" is answerable with
``explain``, and a half-finished request survives inspection between cycles.

    begin(...)    a request picks a procedure and gets a frame
    advance(...)  deliver what the body returned, then run steps until the body is needed again

Mental steps (say, compute, focus, remember, call, return, stop) happen inside one advance.
Body steps (run, click, fill, look, open_app, teach) write the same claims the body already
reads (``kind``, ``command``, ``label``, …) and hand control back to the mind loop.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from typing import Callable

import tensacode as tc
from tensacode.cognition import Thought

from ..mind import one
from . import procedure as L
from .memory import derive, note, set_state, unremember

_seq = itertools.count()


@dataclass
class Host:
    """What the interpreter needs from the assistant around it."""

    say: Callable[[tc.Store, tc.Ref, str, int], Thought]
    find: Callable[[str], L.Procedure | None]
    start_teacher: Callable[[str, dict], None] = lambda key, ask: None
    on_finish: Callable[[tc.Store, tc.Ref, dict, int], Thought] = lambda mind, req, env, cycle: Thought()
    changes_since: Callable[[str], dict] = lambda since: {"any": False, "unavailable": "I keep no snapshots to compare"}
    max_mental_steps: int = 400  # a guard against a procedure that loops without touching the body
    chunks: object | None = None  # a tensacode.chunking.Chunks, if this mind automatizes; None = deliberate always


@dataclass
class Frames:
    """Frame identifiers only; all frame *state* is in the store."""

    counter: itertools.count = field(default_factory=itertools.count)


def _frame_ref(req: tc.Ref, depth: int) -> tc.Ref:
    return tc.Ref(f"frame:{req.id}#{depth}")


def _step_ref(frame: tc.Ref, pc: int) -> tc.Ref:
    return tc.Ref(f"step:{frame.id}@{pc}")


# ----------------------------------------------------------------- the frame


def env_of(mind: tc.Store, frame: tc.Ref) -> dict:
    return {r.claim.predicate[4:]: r.claim.object for r in mind.claims(frame) if r.claim.predicate.startswith("env:")}


def set_env(mind: tc.Store, frame: tc.Ref, values: dict, source: str, premises: tuple[str, ...] = ()) -> Thought:
    thought = Thought()
    for name, value in values.items():
        if value is None or isinstance(value, (bool, int, float, str, list, dict, tuple)):
            thought += set_state(mind, frame, f"env:{name}", value, source)
    if premises:  # what this binding rests on, so explain() can walk back to the command
        thought += derive(mind, [tc.Claim(frame, "bound", tuple(sorted(values))) ], source, premises)
    return thought


def begin(mind: tc.Store, req: tc.Ref, proc: L.Procedure, env: dict, cycle: int) -> Thought:
    frame = _frame_ref(req, 0)
    env = {**env, "request_ref": req.id}
    thought = set_state(mind, req, "frame", frame, f"decision:{cycle}")
    thought += set_state(mind, frame, "procedure", proc.id, f"decision:{cycle}")
    thought += set_state(mind, frame, "pc", 0, f"decision:{cycle}")
    thought += set_env(mind, frame, env, f"decision:{cycle}")
    return thought


# --------------------------------------------------------- body value -> env


def bindings_for(kind: str, value: object) -> dict:
    """Turn what the body returned into environment values (storable, and named the same way every time)."""
    if kind == "run":
        text = getattr(value, "text", "") or ""
        facts = L.PRIMITIVES["output_facts"](out=text, timed_out=bool(getattr(value, "timed_out", False)))
        cwd = getattr(value, "cwd", None)
        return {**facts, "cwd_seen": cwd}
    if kind == "look":
        texts = [t for _, t in getattr(value, "texts", ())][:120]
        labels = [lab for _, lab in getattr(value, "controls", ())][:120]
        windows = [w for w, _ in getattr(value, "texts", ())][:120]
        return {"screen_texts": texts, "screen_labels": labels, "screen_windows": windows}
    if kind in ("click", "fill"):
        return {"ok": value is True, "problem": None if value is True else str(value)}
    if kind == "open_app":
        return {"opened": bool(value)}
    if kind == "teach":
        return {"reply": value if isinstance(value, str) else None}
    if kind == "sample":
        return dict(value) if isinstance(value, dict) else {"unavailable": str(value or "no sample")}
    if kind == "ask":
        return {"answer": getattr(value, "words", str(value or "")), "answer_act": getattr(value, "act", None)}
    return {"result": value if isinstance(value, (str, int, float, bool)) else None}


def _prefixed(values: dict, prefix: str | None) -> dict:
    return {f"{prefix}_{k}": v for k, v in values.items()} if prefix else values


# ------------------------------------------------------------------ the walk


def advance(mind: tc.Store, req: tc.Ref, value: object, cycle: int, host: Host) -> Thought:
    """Deliver ``value`` to the step that asked for it, then run until the body is needed or the request is done."""
    thought = Thought()
    frame = one(mind, req, "frame")
    if frame is None:
        return thought
    act = one(mind, req, "doing")
    if act is not None:
        kind = one(mind, act, "kind")
        step_ref = one(mind, act, "step")
        bound = _prefixed(bindings_for(kind, value), one(mind, act, "as"))
        if host.chunks is not None:  # automatic until something goes differently from the runs it learned on
            thought += _check_chunk(mind, req, frame, step_ref, bound, cycle, host)
            thought += _note_mark(mind, frame, step_ref, bound, cycle, host)
        premises = tuple(r.id for r in mind.claims(act) if r.claim.predicate in ("output", "command", "label", "text", "app"))
        thought += set_env(mind, frame, bound, f"step:{cycle}", premises)
        if step_ref is not None:
            thought += set_state(mind, step_ref, "result", _short(bound), f"step:{cycle}")
        thought += set_state(mind, req, "doing", None, f"step:{cycle}")
        if kind == "ask":  # an answer arrived: the request is no longer waiting on the user
            thought += set_state(mind, req, "status", "running", f"step:{cycle}")
            for r in mind.claims(tc.Ref("agent:self"), "awaiting"):
                if r.claim.object == req:
                    thought += set_state(mind, tc.Ref("agent:self"), "awaiting", None, f"step:{cycle}")
        thought += _bump(mind, frame, cycle)

    for _ in range(host.max_mental_steps):
        frame = one(mind, req, "frame")
        proc = host.find(one(mind, frame, "procedure"))
        if proc is None:
            thought += host.say(mind, req, "I lost track of how I was doing that, so I stopped.", cycle)
            return thought + _finish(mind, req, "failed", cycle, host)
        pc = int(one(mind, frame, "pc") or 0)
        if pc >= len(proc.steps):
            done, thought_ = _pop(mind, req, frame, {}, cycle, host)
            thought += thought_
            if done:
                return thought
            continue
        step = proc.steps[pc]
        env = env_of(mind, frame)
        step_ref = _step_ref(frame, pc)
        chunk = host.chunks.chunk_for(proc.id) if host.chunks is not None else None
        if chunk is not None and pc in chunk.skipped:
            thought += _bump(mind, frame, cycle)  # a guard whose answer never varied is no longer asked
            continue
        if chunk is None:
            try:
                passes = L.holds(step.get("when"), env)
            except Exception as exc:  # noqa: BLE001 - a broken guard is a bug in the procedure, reported not crashed
                thought += host.say(mind, req, f"Sorry, a step in “{proc.id}” is malformed ({type(exc).__name__}: {exc}).", cycle)
                return thought + _finish(mind, req, "failed", cycle, host)
            if not passes:
                thought += note(mind, [tc.Claim(step_ref, "skipped", True), tc.Claim(step_ref, "of", frame)], f"step:{cycle}")
                thought += _bump(mind, frame, cycle)
                continue
        do = step["do"]
        if chunk is None:
            thought += note(mind, [tc.Claim(step_ref, "do", do), tc.Claim(step_ref, "index", pc), tc.Claim(step_ref, "of", frame),
                                   tc.Claim(step_ref, "procedure", proc.id)], f"step:{cycle}")
        elif int(one(mind, frame, "chunk_noted") or 0) == 0:  # one claim for the whole run, not one per step
            thought += note(mind, [tc.Claim(frame, "ran_chunk", chunk.shape), tc.Claim(frame, "of_procedure", proc.id)], f"step:{cycle}")
            thought += set_state(mind, frame, "chunk_noted", 1, f"step:{cycle}")
            host.chunks.used(chunk)
        if do in L.BODY_STEPS:
            return thought + _issue(mind, req, frame, step, step_ref, env, cycle, host)
        thought += _mental(mind, req, frame, step, step_ref, env, cycle, host)
        if one(mind, req, "status") in ("done", "failed", "awaiting"):
            return thought
    thought += host.say(mind, req, "That took too many steps without doing anything, so I stopped.", cycle)
    return thought + _finish(mind, req, "failed", cycle, host)


def _short(values: dict) -> dict:
    """A step's result as it is recorded: long text trimmed so the graph stays readable."""
    out = {}
    for k, v in values.items():
        if isinstance(v, str) and len(v) > 200:
            out[k] = v[:200] + "…"
        elif isinstance(v, list) and len(v) > 12:
            out[k] = v[:12] + ["…"]
        elif v is not None:
            out[k] = v
    return out


def _bump(mind: tc.Store, frame: tc.Ref, cycle: int) -> Thought:
    return set_state(mind, frame, "pc", int(one(mind, frame, "pc") or 0) + 1, f"step:{cycle}")


def _issue(mind: tc.Store, req: tc.Ref, frame: tc.Ref, step: dict, step_ref: tc.Ref, env: dict, cycle: int, host: Host) -> Thought:
    """Write the claims the body reads, and hand control back to the mind loop."""
    do = step["do"]
    act = tc.Ref(f"command:{req.id.split(':', 1)[1]}.{next(_seq)}")
    claims = [tc.Claim(act, "kind", do), tc.Claim(act, "for", req), tc.Claim(act, "step", step_ref)]
    if step.get("as"):
        claims.append(tc.Claim(act, "as", step["as"]))
    why = L.render(step.get("why", ""), env) if step.get("why") else ""
    if not why:  # a default in the agent's own words, so nothing it did is unaccounted for
        detail = next((L.render(step[k], env) for k in ("app", "label", "region", "command") if step.get(k)), "")
        why = {"open_app": "open", "click": "click", "fill": "type into", "look": "look at the screen",
               "sample": "look at the pixels of", "run": "run", "teach": "ask the teacher"}.get(do, do)
        why = f"{why} {detail}".strip()
    # what the body is asked to do rests on the step that asked: explain() walks command -> step -> procedure
    premises = tuple(r.id for r in mind.claims(step_ref))
    if do == "run":
        command = L.render(step["command"], env)
        if not command.strip():  # the template's variables were all missing: report the bug, never type nothing
            return say(mind, req, f"A step in “{procedure}” asked me to run an empty command, so I stopped. That's a bug in my own procedure, not something you did.", cycle), None
        thought = derive(mind, [tc.Claim(act, "command", command), tc.Claim(act, "why", why or command[:80])], f"step:{cycle}", premises)
        thought += note(mind, claims, f"decision:{cycle}")
        thought += set_state(mind, act, "phase", "want", f"decision:{cycle}")
    elif do in ("click", "fill"):
        claims += [tc.Claim(act, "label", L.render(step["label"], env)), tc.Claim(act, "why", why)]
        if do == "fill":
            claims += [tc.Claim(act, "text", L.render(step["text"], env)), tc.Claim(act, "submit", bool(step.get("submit")))]
        thought = note(mind, claims, f"decision:{cycle}")
    elif do == "look":
        claims += [tc.Claim(act, "started_at", time.monotonic()), tc.Claim(act, "settle_ms", int(step.get("settle_ms", 350))),
                   tc.Claim(act, "why", why)]
        thought = note(mind, claims, f"decision:{cycle}")
    elif do == "open_app":
        claims += [tc.Claim(act, "app", L.render(step["app"], env)), tc.Claim(act, "why", why)]
        thought = note(mind, claims, f"decision:{cycle}")
    elif do == "sample":
        claims += [tc.Claim(act, "region", L.render(step.get("region", "screen"), env)), tc.Claim(act, "why", why)]
        if step.get("box") is not None:
            claims.append(tc.Claim(act, "box", L.resolve_value(step["box"], env)))
        thought = note(mind, claims, f"decision:{cycle}")
    else:  # teach
        ask = {"system": L.render(step.get("system", ""), env), "user": L.render(step.get("user", ""), env),
               "max_new_tokens": int(step.get("max_new_tokens", 300))}
        claims.append(tc.Claim(act, "why", why or "ask the teacher model"))
        thought = note(mind, claims, f"decision:{cycle}")
        host.start_teacher(act.id, ask)
    thought += set_state(mind, step_ref, "issued", act, f"decision:{cycle}")
    return thought + set_state(mind, req, "doing", act, f"decision:{cycle}")


def _mental(mind: tc.Store, req: tc.Ref, frame: tc.Ref, step: dict, step_ref: tc.Ref, env: dict, cycle: int, host: Host) -> Thought:
    do = step["do"]
    if do == "say":
        thought = host.say(mind, req, L.render(step["text"], env), cycle)
        after = step.get("and")  # sugar: say and leave, instead of repeating the guard on the next line
        if after == "stop":
            return thought + _finish(mind, req, "done", cycle, host)
        if after == "return":
            values = {k: L.resolve_value(v, env) for k, v in (step.get("values") or {}).items()}
            _, popped = _pop(mind, req, frame, values, cycle, host)
            return thought + popped
        return thought + _bump(mind, frame, cycle)
    if do == "ask":
        question = L.render(step["text"], env)
        act = tc.Ref(f"question:{req.id.split(':', 1)[1]}.{next(_seq)}")
        thought = host.say(mind, req, question, cycle)
        thought += note(mind, [tc.Claim(act, "kind", "ask"), tc.Claim(act, "for", req), tc.Claim(act, "step", step_ref),
                               tc.Claim(act, "question", question)], f"decision:{cycle}")
        if step.get("as"):
            thought += note(mind, [tc.Claim(act, "as", step["as"])], f"decision:{cycle}")
        thought += set_state(mind, req, "doing", act, f"decision:{cycle}")
        thought += set_state(mind, req, "status", "awaiting", f"decision:{cycle}")
        return thought + set_state(mind, tc.Ref("agent:self"), "awaiting", req, f"decision:{cycle}")
    if do == "compute":
        args = {k: L.resolve_value(v, env) for k, v in (step.get("args") or {}).items()}
        prim = L.PRIMITIVES[step["prim"]]
        if "store" in __import__("inspect").signature(prim).parameters:
            args["store"] = mind  # a primitive may ask for the mind itself (e.g. common ground)
        try:
            values = prim(**args)
        except Exception as exc:  # noqa: BLE001 - a primitive that fails is reported, not fatal
            thought = host.say(mind, req, f"Sorry, I hit a problem working that out ({type(exc).__name__}: {exc}).", cycle)
            return thought + _finish(mind, req, "failed", cycle, host)
        bound = _prefixed(values, step.get("as"))
        premises = tuple(r.id for r in mind.claims(frame) if r.claim.predicate.startswith("env:"))[:8]
        thought = set_env(mind, frame, bound, f"step:{cycle}", premises)
        thought += set_state(mind, step_ref, "result", _short(bound), f"step:{cycle}")
        return thought + _bump(mind, frame, cycle)
    if do == "remember":
        triples = [(L.render(s, env), L.render(p, env), L.resolve_value(o, env)) for s, p, o in step.get("triples", [])]
        if step.get("mode") == "forget":  # you asked me to forget it: retract, with the reason kept
            thought = Thought()
            for s_, p_, _ in triples:
                if s_:
                    thought += unremember(mind, tc.Ref(s_), p_, "you asked me to forget it")
        elif step.get("mode") == "replace":  # a told fact that changes: the old value is retracted, not left standing
            thought = Thought()
            for s, p, o in triples:
                if s:
                    thought += set_state(mind, tc.Ref(s), p, o, f"utterance:{one(mind, req, 'order') or cycle}")
        else:
            thought = note(mind, [tc.Claim(tc.Ref(s), p, o) for s, p, o in triples if s], f"decision:{cycle}")
        if step.get("mode") != "forget":
            thought += _ground_told(mind, req, triples, cycle)
        return thought + _bump(mind, frame, cycle)
    if do == "recall":
        bound = _recall(mind, step, env)
        thought = set_env(mind, frame, _prefixed(bound, step.get("as")), f"step:{cycle}")
        thought += set_state(mind, step_ref, "result", _short(_prefixed(bound, step.get("as"))), f"step:{cycle}")
        return thought + _bump(mind, frame, cycle)
    if do == "changes":
        bound = host.changes_since(L.render(step.get("since", "last_message"), env))
        thought = set_env(mind, frame, _prefixed(bound, step.get("as")), f"step:{cycle}")
        thought += set_state(mind, step_ref, "result", _short(_prefixed(bound, step.get("as"))), f"step:{cycle}")
        return thought + _bump(mind, frame, cycle)
    if do == "focus":
        path, kind = L.render(step["path"], env), L.render(step.get("kind", "regular file"), env)
        thought = note(mind, [tc.Claim(tc.Ref(f"path:{path}"), "is_a", kind)], f"decision:{cycle}")
        thought += set_state(mind, tc.Ref("agent:self"), "focus", path, f"decision:{cycle}")
        return thought + _bump(mind, frame, cycle)
    if do == "call":
        return _push(mind, req, frame, step, env, cycle, host)
    if do == "return":
        values = {k: L.resolve_value(v, env) for k, v in (step.get("values") or {}).items()}
        done, thought = _pop(mind, req, frame, values, cycle, host)
        return thought
    if do == "stop":
        return _finish(mind, req, "done", cycle, host)
    raise L.ProcedureError(f"unknown mental step {do!r}")


def _ground_told(mind: tc.Store, req: tc.Ref, triples: list, cycle: int) -> Thought:
    """A fact you just told me is shared the moment it is written, not merely believed.

    Grounding has to happen here rather than at parse time: common ground points at the claim,
    and the claim does not exist until the step that records it runs.
    """
    from tensacode.social import YOU_SAID, CommonGround

    order = one(mind, req, "order")
    turn = int(order[0]) if isinstance(order, (list, tuple)) and order else int(cycle)
    ground, thought = CommonGround(mind), Thought()
    for s, p, _o in triples:
        if s != "person:user" or not p:
            continue
        records = mind.claims(tc.Ref(s), p)
        if records:
            thought += ground.add(records[0].id, YOU_SAID, turn, source=f"utterance:{turn}")
    return thought


def _recall(mind: tc.Store, step: dict, env: dict) -> dict:
    """Look in the mind itself: a question is answered by querying beliefs, not by running a command.

    ``{"do": "recall", "pattern": {"subject": "person:user"}, "as": "told"}`` — any field may be
    left out to mean "anything". Screen items come back with their geometry and window, so a
    question about the sidebar or a window can be answered from what was perceived, and each
    item carries when it was believed, which is the provenance the reply can cite.
    """
    pattern = {k: L.resolve_value(v, env) for k, v in (step.get("pattern") or {}).items()}
    subject = pattern.get("subject")
    kwargs: dict = {}
    if subject:
        kwargs["subject"] = tc.Ref(str(subject))
    if pattern.get("predicate"):
        kwargs["predicate"] = str(pattern["predicate"])
    if "object" in pattern and pattern["object"] is not None:
        kwargs["object"] = pattern["object"]
    records = mind.claims(**kwargs)
    prefix = pattern.get("subject_prefix")
    if prefix:
        records = [r for r in records if r.claim.subject.id.startswith(str(prefix))]
    items, seen_refs = [], {}
    for r in records[: int(step.get("limit", 200))]:
        entity = mind.entities.get(r.claim.subject)
        box = getattr(entity, "box", None)
        item = {"id": str(r.id), "subject": r.claim.subject.id, "predicate": r.claim.predicate,
                "object": r.claim.object if isinstance(r.claim.object, (str, int, float, bool)) else str(r.claim.object),
                "when": r.evidence[-1].observed_at.isoformat(timespec="seconds") if r.evidence else None,
                "source": r.evidence[-1].source.id if r.evidence else None}
        if box is not None:
            item["box"] = [int(v) for v in box]
            item["role"] = getattr(entity, "role", None)
            item["section"] = getattr(entity, "section", None)
        items.append(item)
        seen_refs.setdefault(r.claim.subject.id, True)
    return {"items": items, "n": len(items), "objects": [i["object"] for i in items],
            "subjects": list(seen_refs), "any": bool(items)}


# --------------------------------------------------------------- automatization


def _frame_trace(mind: tc.Store, frame: tc.Ref, procedure: str, status: str):
    """The path this frame took, read back off the store: which steps ran and which were skipped."""
    from tensacode.chunking import Trace

    taken, skipped, marks = [], [], []
    prefix = f"step:{frame.id}@"
    for r in mind.claims(predicate="index"):
        if r.claim.subject.id.startswith(prefix):
            taken.append(int(r.claim.object))
    for r in mind.claims(predicate="skipped"):
        if r.claim.subject.id.startswith(prefix):
            skipped.append(int(r.claim.subject.id.rsplit("@", 1)[1]))
    for r in mind.claims(predicate="trouble"):
        if r.claim.subject.id.startswith(prefix):
            marks.append((int(r.claim.subject.id.rsplit("@", 1)[1]), str(r.claim.object)))
    return Trace(procedure, tuple(sorted(taken)), tuple(sorted(skipped)), status, tuple(sorted(marks)))


def _record_run(mind: tc.Store, frame: tc.Ref, status: str, host: Host) -> None:
    """Note how a frame's run went, so a repeated path can compile into a chunk."""
    if host.chunks is None or frame is None:
        return
    if int(one(mind, frame, "chunk_noted") or 0):
        return  # it ran as a chunk already; there is nothing new to learn from it
    procedure = one(mind, frame, "procedure")
    if procedure is None:
        return
    host.chunks.record(_frame_trace(mind, frame, str(procedure), status))


def _note_mark(mind: tc.Store, frame: tc.Ref, step_ref: tc.Ref | None, bound: dict, cycle: int, host: Host) -> Thought:
    """Record that a step showed trouble, so a chunk can know what normal looked like.

    Only written when there *is* a mark, and only while deliberating: a step that goes fine costs
    no claim, and a step under a chunk has nothing to teach.
    """
    from tensacode.chunking import mark

    if step_ref is None or int(one(mind, frame, "chunk_noted") or 0):
        return Thought()
    found = mark(bound)
    return set_state(mind, step_ref, "trouble", found, f"step:{cycle}") if found else Thought()


def _check_chunk(mind: tc.Store, req: tc.Ref, frame: tc.Ref, step_ref: tc.Ref | None, bound: dict, cycle: int, host: Host) -> Thought:
    """A chunk assumes the world keeps behaving; when it does not, deliberate again from here."""
    from tensacode.chunking import divergent

    procedure = one(mind, frame, "procedure")
    chunk = host.chunks.chunk_for(str(procedure)) if procedure is not None else None
    if chunk is None or not int(one(mind, frame, "chunk_noted") or 0):
        return Thought()
    pc = int(step_ref.id.rsplit("@", 1)[1]) if step_ref is not None else -1
    why = divergent(bound, chunk.expects(pc))
    if not why:
        return Thought()
    host.chunks.retire(chunk, why)
    thought = note(mind, [tc.Claim(frame, "chunk_broke", why)], f"step:{cycle}")
    return thought + set_state(mind, frame, "chunk_noted", 0, f"step:{cycle}")


def _push(mind: tc.Store, req: tc.Ref, frame: tc.Ref, step: dict, env: dict, cycle: int, host: Host) -> Thought:
    proc = host.find(step["proc"])
    if proc is None:
        thought = host.say(mind, req, f"I don't have a way to do “{step['proc']}”.", cycle)
        return thought + _finish(mind, req, "failed", cycle, host)
    here = host.find(one(mind, frame, "procedure"))
    tail = here is not None and step["proc"] == here.id and int(one(mind, frame, "pc") or 0) == len(here.steps) - 1
    if tail:  # a procedure calling itself as its last step is a loop: reuse the frame instead of stacking
        passed = {k: L.resolve_value(v, env) for k, v in (step.get("with") or {}).items()}
        thought = set_env(mind, frame, passed, f"decision:{cycle}")
        return thought + set_state(mind, frame, "pc", 0, f"decision:{cycle}")
    depth = int(frame.id.rsplit("#", 1)[1]) + 1
    child = _frame_ref(req, depth)
    passed = {k: L.resolve_value(v, env) for k, v in (step.get("with") or {}).items()}
    thought = set_state(mind, child, "procedure", proc.id, f"decision:{cycle}")
    thought += set_state(mind, child, "pc", 0, f"decision:{cycle}")
    thought += set_state(mind, child, "caller", frame, f"decision:{cycle}")
    thought += set_state(mind, child, "binds", step.get("bind") or {}, f"decision:{cycle}")
    inherited = {k: env.get(k) for k in ("cwd", "focus", "focus_kind", "known", "words", "act", "slot", "request_ref")}
    thought += set_env(mind, child, {**inherited, **passed}, f"decision:{cycle}")
    return thought + set_state(mind, req, "frame", child, f"decision:{cycle}")


def _pop(mind: tc.Store, req: tc.Ref, frame: tc.Ref, values: dict, cycle: int, host: Host) -> tuple[bool, Thought]:
    """Leave a frame. Returning from the outermost one finishes the request."""
    caller = one(mind, frame, "caller")
    if caller is None:
        return True, _finish(mind, req, "done", cycle, host)
    _record_run(mind, frame, "done", host)
    binds = one(mind, frame, "binds") or {}
    bound = {into: values.get(name) for into, name in dict(binds).items()}
    thought = set_env(mind, caller, bound, f"step:{cycle}")
    thought += set_state(mind, req, "frame", caller, f"step:{cycle}")
    thought += _bump(mind, caller, cycle)
    return False, thought


def _finish(mind: tc.Store, req: tc.Ref, status: str, cycle: int, host: Host) -> Thought:
    frame = one(mind, req, "frame")
    _record_run(mind, frame, status, host)
    env = env_of(mind, frame) if frame is not None else {}
    thought = set_state(mind, req, "doing", None, f"decision:{cycle}")
    thought += host.on_finish(mind, req, env, cycle)
    return thought + set_state(mind, req, "status", status, f"decision:{cycle}")


def trace_of(mind: tc.Store, req: tc.Ref) -> list[str]:
    """Readable: what this request's procedure did, step by step (for the viewer and for tests)."""
    rows = []
    for r in mind.claims(predicate="do"):
        step = r.claim.subject
        if not step.id.startswith(f"step:frame:{req.id}"):
            continue
        pc = one(mind, step, "index")
        result = one(mind, step, "result")
        rows.append((one(mind, step, "of").id, pc, f"{one(mind, step, 'procedure')}[{pc}] {r.claim.object}" + (f" -> {result}" if result else "")))
    return [text for _, _, text in sorted(rows)]
