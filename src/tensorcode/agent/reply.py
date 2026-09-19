"""Language out: one reply per message, said from what actually happened.

Each outcome becomes a frame the agent asserts about itself ("I made a folder", "I
can't design a device") and the grammar realizes it — the same grammar that read the
request, run backwards. Only the connective tissue is fixed wording: "because", the
list punctuation, and the reason an outcome carries (which is data from the planner or
the plugin, not prose chosen for a prompt). When realization fails there is one plain
sentence for "I could not say that" — the frame it could not realize goes to the trace,
because a frame printed in a reply is not an answer and was once counted as one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..language import Entity, Frame, realize

if TYPE_CHECKING:
    from .core import Agent, Outcome
    from .understand import Sentence

I = Entity("pronoun", "I", {"person": 1, "number": "singular"})


def say(agent: "Agent", frame: Frame) -> str:
    text = realize(agent.grammar, frame)
    if not text:
        return frame.describe()
    return surface(text)


def said_or_not(agent: "Agent", frame: Frame) -> tuple[str, bool]:
    """The frame as a sentence, and whether the grammar managed to say it.

    A frame the grammar cannot realize used to be printed as data in the reply — a reader
    got ``tell(content=name:Voyager, ...) [modality=can, polarity=negative]``. That is honest
    about the failure but it is not English, and it was read downstream as a confident answer.
    The caller says what to put in its place; the frame itself belongs in the trace.
    """
    text = realize(agent.grammar, frame)
    return (surface(text), True) if text else (frame.describe(), False)


def surface(text: str) -> str:
    """Orthography only: the first-person pronoun and the first letter are capitals."""
    words = ["I" if w == "i" else w for w in text.split(" ")]
    out = " ".join(words)
    return out[:1].upper() + out[1:]


def shift(value: Any) -> Any:
    """Deixis: what the speaker called theirs is the listener's when said back (1 <-> 2)."""
    if isinstance(value, Entity):
        feats = {k: shift(v) for k, v in value.features.items()}
        if value.kind == "description" and value.text:
            # said back in the speaker's own words when the lexicon's reading replaced them
            # (a spelling correction), not when it is only their lemma ("recipes" -> "recipe")
            said = [w for w in value.text.split() if w.lower() not in ("the", "a", "an", "my", "your", "our")]
            noun = str(feats.get("noun") or "")
            if said and not said[-1].lower().startswith(noun.lower()[: max(3, len(noun) - 1)]):
                feats["noun"] = said[-1]
        if feats.get("possessor") in (1, 2):
            feats["possessor"] = 3 - feats["possessor"]
        if value.kind == "pronoun" and value.features.get("person") in (1, 2):
            feats["person"] = 3 - value.features["person"]
        return Entity(value.kind, value.text, feats, value.ref, value.candidates)
    if isinstance(value, Frame):
        return Frame(value.predicate, {k: shift(v) for k, v in value.roles.items()}, value.features)
    if isinstance(value, tuple):
        return tuple(shift(v) for v in value)
    return value


def about(agent: "Agent", o: "Outcome", **features: Any) -> Frame:
    """The request's own frame, re-said with the agent as subject and the perspective shifted."""
    f = o.act.frame
    roles = {k: shift(v) for k, v in f.roles.items() if k != "subject"}
    return Frame(f.predicate, {"subject": I, **roles}, {"mood": "declarative", **features})


def clause(agent: "Agent", o: "Outcome") -> str | None:
    if o.status == "done":
        text, said = said_or_not(agent, about(agent, o, tense="past"))
        return f"{text}, and checked that it worked" if said else "I did that, and checked that it worked"
    if o.status == "unverified":
        text, said = said_or_not(agent, about(agent, o, tense="past"))
        return (f"{text}, but couldn't check it ({o.reason})" if said
                else f"I did that, but couldn't check it ({o.reason})")
    if o.status == "failed":
        text, said = said_or_not(agent, about(agent, o, tense="past", polarity="negative"))
        return f"{text} ({o.reason})" if said else f"I tried, and it didn't work ({o.reason})"
    if o.status == "declined":
        text, said = said_or_not(agent, about(agent, o, modality="can", polarity="negative"))
        return f"{text}: {o.reason}" if said else f"I can't do that: {o.reason}"
    if o.status == "unknown" and o.act.kind == "request":
        return f"I don't know what \"{o.act.frame.predicate}\" should achieve ({o.reason})"
    if o.status == "unknown":
        return f"I don't know ({o.reason})"
    if o.status == "answered":
        return answer_text(agent, o)
    if o.status == "not_understood":
        return None  # reported per sentence, below
    return None


def answer_text(agent: "Agent", o: "Outcome") -> str:
    names = []
    for v in o.answer or []:
        shown = next((d for p in agent.plugins if (d := p.display(v))), None)
        names.append(shown or str(getattr(v, "id", v)).split(":", 1)[-1])
    names = list(dict.fromkeys(names))
    if not names:
        return "There is nothing there."
    if len(names) == 1:
        return names[0] + "."
    return ", ".join(names[:-1]) + " and " + names[-1] + "."


def compose(agent: "Agent", sentences: list["Sentence"], outcomes: list["Outcome"]) -> str:
    parts: list[str] = []
    counts: dict[str, int] = {}
    for o in outcomes:
        c = clause(agent, o)
        if c:
            c = c if c.endswith((".", "?", "!")) else c + "."
            if c not in counts:
                parts.append(c)
            counts[c] = counts.get(c, 0) + 1
    parts = [p if counts[p] == 1 else p[:-1] + f" ({counts[p]} times)." for p in parts]
    mentioned = sum(1 for o in outcomes if o.status == "mentioned")
    if mentioned:
        parts.append(f"I read the quoted part as something you were describing, not as {'a request' if mentioned == 1 else 'requests'} to me.")
    noted = sum(1 for o in outcomes if o.status == "noted")
    if noted:
        parts.append(f"I noted {noted} thing{'s' if noted != 1 else ''} you told me.")
    unread = [s for s in sentences if s.coverage < 1.0 or not s.acts]
    if unread:
        shown = unread[:3]
        missed = "; ".join(f"“{s.text}” (not {', '.join(w for w in s.skipped if any(ch.isalnum() for ch in w)) or 'parsed'})" for s in shown)
        more = f" and {len(unread) - len(shown)} more" if len(unread) > len(shown) else ""
        parts.append(f"I didn't fully follow {missed}{more}, so I didn't act on {'it' if len(unread) == 1 else 'those'}.")
    # a sentence the reader covered but the agent could not record is not "nothing to do":
    # say which part of it was not followed, or the next turn asks about something the
    # store never received
    said = {s.text for s in unread}
    misread = [o.reason for o in outcomes if o.status == "not_understood" and o.reason and o.act.frame is not None
               and o.act.frame.describe() not in said]
    if misread and not any("didn't fully follow" in p for p in parts):
        parts.append(f"I didn't follow all of that: {misread[0]}.")
    if not parts:
        return "I read that, but it didn't ask me to do or answer anything."
    return " ".join(parts)
