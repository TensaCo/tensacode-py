"""Symbolic language understanding and generation: one grammar, both directions.

    from tensorcode.language import ENGLISH, understand, realize, Context, resolve, to_claims

    got = understand(ENGLISH, "Anem said the north field failed")
    got.meanings          # (Frame('say', {speaker/subject, content: Frame('fail', ...)}),)
    got.skipped           # words no constituent claimed, reported not guessed
    got.ambiguous         # two readings of equal score survive as two readings

    to_claims(frame, source=Ref("agent:anem"))   # reported speech lands in its own scope
    realize(ENGLISH, frame)                      # say it again with the same grammar

No model is called anywhere in this package, and it has no dependencies outside
the standard library.
"""

from .chart import Chart, Node, Reading, Understanding, build_chart, cover, tokenize, understand, unquote
from .discourse import Context, resolve, unresolved
from .english import ENGLISH, ENGLISH_LEXICON
from .features import FVar, ground, unify
from .generate import realize, round_trip
from .grammar import (
    Ask, Attach, Build, Cat, Coord, Ent, Entry, Grammar, Head, Lexicon, Lit, Locative, Merge, Order, Production,
    Qualify, Terminal, inflect, production, words,
)
from .grammar import OpenClass, guess_entries
from .semantics import Entity, Frame, Question, Request, to_claims

__all__ = [
    "Ask", "Attach", "Build", "Cat", "Chart", "Context", "Coord", "ENGLISH", "ENGLISH_LEXICON", "Ent", "Entity",
    "Entry", "FVar", "Frame", "Grammar", "Head", "Lexicon", "Lit", "Locative", "Merge", "Node", "Order",
    "OpenClass", "Production", "Qualify", "Question", "Reading", "Request", "Terminal", "Understanding",
    "build_chart", "guess_entries",
    "cover", "ground", "inflect", "production", "realize", "resolve", "round_trip", "to_claims", "tokenize",
    "unify", "unquote", "unresolved", "understand", "words",
]
