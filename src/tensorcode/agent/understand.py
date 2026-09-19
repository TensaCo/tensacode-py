"""Grammar and learned readers propose interpretations while retaining source evidence.

Sentence splitting and quotation handling are authored preprocessing conventions.
Quotation metadata records when a supported enclosing span is treated as mentioned
language; malformed or mixed delimiters remain in the input. Reader scores and
speech-act proposals do not authorize dispatch: the workspace retains alternatives
for an explicit interpretation decision or deferral.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from ..language import Entity, Frame, Grammar, Question, Request, understand
from ..language.chart import Reading, tokenize
from ..language.conventions import RequestConventions, RequestInterpretation, interpret_request, request_conventions

#: Sentence-final punctuation. Brackets nest and double quotes or backticks toggle, and
#: anything inside them stays in one sentence; an apostrophe ("don't") is not a quote.
ENDERS = ".!?"
BRACKETS = {"(": ")", "[": "]", "{": "}"}
TOGGLES = "\"`“”"


def sentences(text: str) -> list[str]:
    """Split a message into sentences: at . ! ? followed by a space or end, and at line breaks.

    Each line of a list ("the artifact itself" under "design:") is its own sentence;
    the agent still reads them as one message and answers once.
    """
    out: list[str] = []
    buf: list[str] = []
    closing: list[str] = []
    quoted = False

    def flush() -> None:
        s = "".join(buf).strip()
        if s:
            out.append(s)
        buf.clear()

    for i, ch in enumerate(text):
        if ch in TOGGLES:
            quoted = not quoted
        elif not quoted and ch in BRACKETS:
            closing.append(BRACKETS[ch])
        elif not quoted and closing and ch == closing[-1]:
            closing.pop()
        inside = quoted or bool(closing)
        if ch == "\n" and not inside:
            flush()
            continue
        buf.append(ch)
        if ch in ENDERS and not inside and (i + 1 == len(text) or text[i + 1].isspace()):
            flush()
    flush()
    return out


@dataclass(frozen=True)
class Act:
    """What one meaning in a sentence does."""

    kind: str          # "request" | "question" | "tell" | "fragment" | "mention"
    meaning: Any       # Request | Question | Frame | other constituent
    frame: Frame | None
    interpretation: RequestInterpretation | None = None

    def describe(self) -> str:
        m = self.meaning
        return f"{self.kind}: " + (m.describe() if hasattr(m, "describe") else repr(m))


@dataclass(frozen=True)
class SentenceAlternative:
    """A reader's proposed interpretation, not an established meaning.

    Order and any ``reading.score`` come from the reader, not calibrated belief
    probabilities. A singleton only records what the reader returned. In particular,
    composed and learned readings do not enumerate all possible interpretations.
    """

    reading: Reading | None
    acts: tuple[Act, ...]
    skipped: tuple[str, ...] = ()
    guessed: tuple[tuple[str, str], ...] = ()
    provenance: str = "grammar"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Sentence:
    text: str
    tokens: tuple[str, ...]
    reading: Reading | None
    acts: tuple[Act, ...]
    skipped: tuple[str, ...] = field(default=())
    guessed: tuple[tuple[str, str], ...] = field(default=())
    parse_ms: float = 0.0
    alternatives: tuple[SentenceAlternative, ...] = ()

    @property
    def coverage(self) -> float:
        """Share of word tokens the parse used (punctuation does not count against it)."""
        words = [t for t in self.tokens if any(c.isalnum() for c in t)]
        missed = [t for t in self.skipped if any(c.isalnum() for c in t)]
        return 1.0 if not words else round(1 - len(missed) / len(words), 3)


def indirect_request(q: Question, conventions: RequestConventions | None = None) -> Request | None:
    """Compatibility wrapper for the language-owned, defeasible request convention."""
    interpretation = interpret_request(q, conventions)
    return interpretation.request if interpretation is not None else None


def acts_of(meaning: Any, conventions: RequestConventions | None = None) -> list[Act]:
    """One act per meaning; coordinated requests ("do x and do y") are several."""
    if isinstance(meaning, tuple):
        return [a for m in meaning for a in acts_of(m, conventions)]
    return [act_of(meaning, conventions)]


def act_of(meaning: Any, conventions: RequestConventions | None = None) -> Act:
    if isinstance(meaning, Request):
        frame = meaning.frame
        return Act("request", Request(frame), frame)
    if isinstance(meaning, Question):
        interpretation = interpret_request(meaning, conventions)
        if interpretation is not None:
            request = interpretation.request
            return Act("request", request, request.frame, interpretation)
        return Act("question", meaning, meaning.frame)
    if isinstance(meaning, Frame):
        mood = meaning.mood
        if mood == "imperative":
            return Act("request", Request(meaning), meaning)
        return Act("tell", meaning, meaning)
    return Act("fragment", meaning, None)


QUOTES = {'"': '"', "“": "”", "'": "'", "‘": "’", "`": "`"}


@dataclass(frozen=True)
class QuotationEnvelope:
    """Source spans for one structurally supported enclosing quotation."""
    status: str
    content_span: tuple[int, int] | None = None
    delimiter_spans: tuple[tuple[int, int], ...] = ()
    suffix_span: tuple[int, int] | None = None
    reason: str | None = None


def quotation_envelope(s: str) -> QuotationEnvelope:
    """Inspect delimiters without stripping content or inferring speaker intent.

    Apostrophes between word characters belong to the source word, including
    contractions. Other ambiguous apostrophes are not silently treated as an
    enclosing quotation. Escaped delimiters stay in the source unchanged.
    """
    start = len(s) - len(s.lstrip())
    end = len(s.rstrip())
    if start >= end or s[start] not in QUOTES:
        return QuotationEnvelope("none")
    opening, closing = s[start], QUOTES[s[start]]
    close = None
    for index in range(start + 1, end):
        if s[index] != closing:
            continue
        slashes = 0
        cursor = index - 1
        while cursor >= start and s[cursor] == "\\":
            slashes += 1
            cursor -= 1
        if slashes % 2:
            continue
        previous = s[index - 1] if index else ""
        following = s[index + 1] if index + 1 < end else ""
        if closing in ("'", "’") and previous.isalnum() and following.isalnum():
            continue
        if closing in ("'", "’") and following.isalnum() and not previous.isalnum():
            return QuotationEnvelope("ambiguous", reason="apostrophe or nested quote delimiter")
        close = index
        break
    if close is None:
        return QuotationEnvelope("unmatched", reason="no unambiguous closing delimiter")
    tail = s[close + 1:end]
    if any(not character.isspace() and character not in ENDERS for character in tail):
        delimiters = sum(character in (opening, closing) for character in tail)
        status = "unmatched" if delimiters % 2 else "multiple" if delimiters else "mixed"
        return QuotationEnvelope(status, reason="material remains after the first closing delimiter")
    return QuotationEnvelope("whole", (start + 1, close),
                             ((start, start + 1), (close, close + 1)), (close + 1, len(s)))


def quoted(s: str) -> str | None:
    """Exact interior of a supported whole quotation, including empty content."""
    envelope = quotation_envelope(s)
    return s[slice(*envelope.content_span)] if envelope.content_span is not None else None


def _quotation_metadata(s: str, offset: int = 0) -> dict[str, Any]:
    envelope = quotation_envelope(s)
    def absolute(span):
        return (offset + span[0], offset + span[1]) if span is not None else None
    return {"status": envelope.status, "reason": envelope.reason,
            "convention": "authored:whole-quotation-as-mention",
            "applied": envelope.status == "whole",
            "source_span": (offset, offset + len(s)),
            "content_span": absolute(envelope.content_span),
            "delimiter_spans": tuple(absolute(span) for span in envelope.delimiter_spans),
            "suffix_span": absolute(envelope.suffix_span)}


def parse_one(grammar: Grammar, s: str, *, mention: bool = False, conventions: RequestConventions | None = None) -> Sentence:
    u = understand(grammar, s)
    alternatives = []
    for r in u.readings:
        acts = tuple(a for m in r.meanings for a in acts_of(m, conventions))
        if mention:
            # Every candidate is mentioned language, including unselected requests.
            acts = tuple(Act("mention", a.meaning, a.frame, a.interpretation) for a in acts)
        alternatives.append(SentenceAlternative(r, acts, tuple(w for _, w in r.skipped), r.guessed))
    first = alternatives[0] if alternatives else None
    return Sentence(s, tuple(u.tokens), first.reading if first else None, first.acts if first else (),
                    first.skipped if first else tuple(tokenize(s)), first.guessed if first else (),
                    round(u.ms, 1), tuple(alternatives))


def read(grammar: Grammar, text: str, *, conventions: RequestConventions | None = None) -> list[Sentence]:
    """Every sentence of ``text``, parsed, with its acts in order.

    Two pieces of text structure are read, both general:

    * a sentence that is wholly a quotation is parsed as language, and its acts are
      *mentions* — the use/mention distinction;
    * a sentence ending in a colon, followed by lines that are noun phrases, is one
      request whose object is those lines ("design: the artifact itself / its bom").
    """
    conventions = request_conventions(conventions)
    parts = sentences(text)
    part_starts = []
    cursor = 0
    for part in parts:
        start = text.find(part, cursor)
        if start < 0:
            raise ValueError("sentence cannot be anchored in its source")
        part_starts.append(start)
        cursor = start + len(part)
    out: list[Sentence] = []
    i = 0
    while i < len(parts):
        s = parts[i]
        inner = quoted(s)
        if inner is not None:
            quote_metadata = _quotation_metadata(s, part_starts[i])
            if not tokenize(inner):
                alternative = SentenceAlternative(None, (), provenance="quoted-source-unresolved", metadata={
                    "quotation": quote_metadata, "syntax_complete": False,
                    "unresolved": "quoted source has no lexical content"})
                out.append(Sentence(s, (), None, (), alternatives=(alternative,)))
            else:
                parsed = parse_one(grammar, inner, mention=True, conventions=conventions)
                out.append(replace(parsed, text=s, alternatives=tuple(
                    replace(alternative, metadata={**alternative.metadata, "quotation": quote_metadata})
                    for alternative in parsed.alternatives)))
            i += 1
            continue
        if s.endswith(":") and i + 1 < len(parts):
            head = parse_one(grammar, s[:-1].strip(), conventions=conventions)
            items, j = [], i + 1
            while j < len(parts) and not parts[j].endswith(":"):
                np = understand(grammar, parts[j], starts=("NP",))
                r = np.readings[0] if np.readings else None
                if r is None or len(r.meanings) != 1 or r.skipped and len([w for _, w in r.skipped if w.isalnum()]) > len(np.tokens) // 2:
                    break
                items.append((parts[j], r, tuple(np.tokens)))
                j += 1
            request = next((a for a in head.acts if a.kind == "request" and a.frame is not None and "object" not in a.frame.roles), None)
            if items and request is not None:
                objects = tuple(r.meanings[0] for _, r, _ in items)
                frame = request.frame.filled(object=objects if len(objects) > 1 else objects[0])
                skipped = tuple(w for _, r, _ in items for _, w in r.skipped)
                guessed = tuple(g for _, r, _ in items for g in r.guessed)
                text_all = s + " " + "; ".join(t for t, _, _ in items)
                tokens = head.tokens + tuple(tok for _, _, toks in items for tok in toks)
                acts = (act_of(Request(frame), conventions),)
                skipped, guessed = head.skipped + skipped, head.guessed + guessed
                # The head reading is retained for compatibility, but is not a
                # parse of the entire composition. Do not present it as one.
                alternative = SentenceAlternative(None, acts, skipped, guessed, "colon-composition-selected")
                out.append(Sentence(text_all, tokens, head.reading, acts,
                                    skipped, guessed, head.parse_ms, (alternative,)))
                i = j
                continue
        parsed = parse_one(grammar, s, conventions=conventions)
        out.append(replace(parsed, alternatives=tuple(
            replace(alternative, metadata={**alternative.metadata, "quotation": _quotation_metadata(s, part_starts[i])})
            for alternative in parsed.alternatives)))
        i += 1
    return out


# ------------------------------------------------------------------ the learned reader


class LearnedReader:
    """Bounded learned syntactic proposals, without selecting an interpretation.

    Tag and dependency scores remain separate, uncalibrated model scores. Complete
    syntax is not established semantic understanding: the semantic adapter is still
    authored. The compatibility ``Sentence.acts`` field exposes the first retained
    proposal; the interpretation workspace retains every alternative and its default
    policy defers. Empty searches retain an unresolved source rather than a repaired
    dependency root or an invented complete interpretation.
    """

    def __init__(self, model_path=None, *, conventions: RequestConventions | None = None,
                 tag_beam_width: int = 4, tag_max_candidates: int = 4,
                 parse_beam_width: int = 8, parse_max_candidates: int = 4,
                 max_expansions: int = 100000, max_alternatives: int = 16,
                 max_sentence_expansions: int = 600000, parse_ranking: str = "local_margin",
                 semantic_max_candidates: int = 4, semantic_max_expansions: int = 64,
                 max_sentence_semantic_expansions: int = 2048) -> None:
        import hashlib
        from pathlib import Path

        from ..language.deps_semantics import Reader
        from ..language.learned_parser import load_model
        from ..language.treebank import lemma_table, lemmatize, load as load_treebank

        limits = (tag_beam_width, tag_max_candidates, parse_beam_width,
                  parse_max_candidates, max_expansions, max_alternatives, max_sentence_expansions,
                  semantic_max_candidates, semantic_max_expansions, max_sentence_semantic_expansions)
        if any(not isinstance(value, int) or isinstance(value, bool) or value < 1 for value in limits):
            raise ValueError("learned reader search bounds must be positive integers")
        if parse_ranking not in ("raw", "local_margin"):
            raise ValueError("parse ranking must be raw or local_margin")
        self.semantic_max_candidates = semantic_max_candidates
        self.semantic_max_expansions = semantic_max_expansions
        self.max_sentence_semantic_expansions = max_sentence_semantic_expansions
        self.parse_ranking = parse_ranking
        self.max_sentence_expansions = max_sentence_expansions
        path = Path(model_path or Path.home() / ".cache" / "tensorcode" / "models" / "ud_ewt_parser.pickle")
        got = load_model(path)
        if got is None:
            raise FileNotFoundError(f"no trained parser at {path}: run eval/parsing/train_ud.py")
        self.tagger, self.parser = got
        self.model_artifact = {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        self.table = lemma_table(load_treebank("train"))
        self.lemmatize = lemmatize
        self.reader = Reader()
        self.conventions = request_conventions(conventions)
        self.tag_beam_width, self.tag_max_candidates = tag_beam_width, tag_max_candidates
        self.parse_beam_width, self.parse_max_candidates = parse_beam_width, parse_max_candidates
        self.max_expansions, self.max_alternatives = max_expansions, max_alternatives

    @staticmethod
    def _search_metadata(search, beam_width, max_candidates, max_expansions):
        return {"beam_width": beam_width, "max_candidates": max_candidates,
                "max_expansions": max_expansions, "expansions": search.expansions,
                "complete": search.complete, "truncated": search.truncated, "reason": search.reason}

    def read(self, text: str) -> list[Sentence]:
        from copy import deepcopy
        from dataclasses import asdict
        from ..language.deps_semantics import SemanticReadCandidates
        import time

        out = []
        message_cursor = 0
        for raw in sentences(text):
            raw_start = text.find(raw, message_cursor)
            if raw_start < 0:
                raise ValueError("sentence cannot be anchored in its source")
            message_cursor = raw_start + len(raw)
            inner = quoted(raw)
            t0 = time.perf_counter()
            quote_metadata = _quotation_metadata(raw, raw_start)
            words = list(tokenize(inner if inner is not None else raw))
            if not words:
                alternative = SentenceAlternative(None, (), provenance="learned-reader-unresolved", metadata={
                    "model_artifact": self.model_artifact, "sentence_span": (raw_start, message_cursor),
                    "token_anchors": (), "quotation": quote_metadata, "syntax_complete": False,
                    "semantic_projection_complete": False, "unresolved": "quoted source has no lexical content"})
                out.append(Sentence(raw, (), None, (), alternatives=(alternative,)))
                continue
            anchors = []
            token_cursor = quotation_envelope(raw).content_span[0] if inner is not None else 0
            for index, word in enumerate(words, 1):
                start = raw.find(word, token_cursor)
                if start < 0:
                    raise ValueError("token cannot be anchored in its source")
                token_cursor = start + len(word)
                anchors.append({"index": index, "token": word,
                                "char_span": (raw_start + start, raw_start + token_cursor)})
            remaining_expansions = self.max_sentence_expansions
            tag_budget = min(self.max_expansions, remaining_expansions)
            tag_search = self.tagger.tag_candidates(words, beam_width=self.tag_beam_width,
                                                    max_candidates=self.tag_max_candidates,
                                                    max_expansions=tag_budget)
            remaining_expansions -= tag_search.expansions
            tag_metadata = self._search_metadata(tag_search, self.tag_beam_width,
                                                 self.tag_max_candidates, tag_budget)
            common = {"model_artifact": self.model_artifact, "sentence_span": (raw_start, message_cursor),
                      "quotation": quote_metadata, "token_anchors": tuple(anchors), "tag_search": tag_metadata,
                      "semantic_adapter": "authored:deps_semantics.Reader",
                      "semantic_projection_complete": None, "coverage_basis": "syntactic attachment only"}
            greedy_tagged = self.tagger.greedy_candidate(words)
            tagged_proposals = ([greedy_tagged] if greedy_tagged is not None else []) + list(tag_search.candidates)
            tagged_unique = {}
            tag_paths = {}
            for tagged in tagged_proposals:
                signature = tuple(tagged.tags)
                tagged_unique.setdefault(signature, tagged)
                tag_paths.setdefault(signature, []).append({
                    "provenance": tuple(getattr(tagged, "provenance", ("unspecified",))),
                    "lexical_positions_zero_based": tuple(getattr(tagged, "lexical_positions", ())),
                    "score": {"value": tagged.score, "kind": "uncalibrated"}})
            common["tag_proposal_policy"] = "retain learned greedy tag path alongside bounded alternatives"
            batches = []
            parse_searches = []
            semantic_cache = {}
            remaining_semantic = self.max_sentence_semantic_expansions
            for lane_index, (tags, tagged) in enumerate(tagged_unique.items()):
                lemmas = tuple(self.lemmatize(w, t, self.table) for w, t in zip(words, tags))
                greedy_search = None
                step_budget = min(remaining_expansions, self.max_expansions, 4 * len(words) + 10)
                if greedy_tagged is not None and tags == tuple(greedy_tagged.tags):
                    greedy_search = self.parser.greedy_search(words, tags, max_steps=step_budget)
                    remaining_expansions -= greedy_search.expansions
                lane_budget = min(self.max_expansions, remaining_expansions // (len(tagged_unique) - lane_index))
                parsed = self.parser.parse_candidates(words, tags, beam_width=self.parse_beam_width,
                                                      max_candidates=self.parse_max_candidates,
                                                      max_expansions=lane_budget, ranking=self.parse_ranking)
                remaining_expansions -= parsed.expansions
                parse_metadata = self._search_metadata(parsed, self.parse_beam_width,
                                                       self.parse_max_candidates, lane_budget)
                greedy_metadata = ({"max_steps": step_budget, "expansions": greedy_search.expansions,
                                    "complete": greedy_search.complete, "truncated": greedy_search.truncated,
                                    "reason": greedy_search.reason} if greedy_search is not None else None)
                parse_searches.append({"tags": tags, "tag_score": {"value": tagged.score, "kind": "uncalibrated"},
                                       "tag_proposals": tuple(tag_paths[tags]), "greedy_search": greedy_metadata,
                                       **parse_metadata})
                proposals = ([(candidate, "greedy-unrepaired") for candidate in greedy_search.candidates]
                             if greedy_search is not None else [])
                proposals += [(candidate, "bounded-search") for candidate in parsed.candidates]
                batch = []
                for candidate, method in proposals:
                    syntax_key = (tags, tuple(sorted(candidate.heads.items())), tuple(sorted(candidate.labels.items())))
                    if syntax_key not in semantic_cache:
                        semantic_budget = min(self.semantic_max_expansions, remaining_semantic)
                        semantics = (self.reader.read_candidates(words, tags, lemmas, candidate.heads, candidate.labels,
                                     max_candidates=self.semantic_max_candidates, max_expansions=semantic_budget)
                                     if semantic_budget else SemanticReadCandidates((), True, 0, 1))
                        remaining_semantic -= semantics.explored
                        semantic_cache[syntax_key] = (semantics, semantic_budget)
                    semantics, semantic_budget = semantic_cache[syntax_key]
                    for semantic_index, semantic in enumerate(semantics.candidates or (None,)):
                        meanings = semantic.meanings if semantic is not None else ()
                        acts = tuple(a for m in meanings for a in acts_of(m, self.conventions))
                        if inner is not None:
                            acts = tuple(Act("mention", a.meaning, a.frame, a.interpretation) for a in acts)
                        metadata = {**common, "tags": tags, "lemmas": lemmas,
                                    "heads": dict(candidate.heads), "labels": dict(candidate.labels),
                                    "transitions": tuple(candidate.transitions),
                                    "tag_score": {"value": tagged.score, "kind": "uncalibrated"},
                                    "parser_score": {"value": candidate.score, "kind": "uncalibrated"},
                                    "parse_search": parse_metadata, "greedy_search": greedy_metadata,
                                    "tag_proposals": tuple(tag_paths[tags]), "parser_method": method,
                                    "parser_ranking": getattr(candidate, "ranking", "raw"),
                                    "parser_provenance": tuple(getattr(candidate, "provenance", (method,))),
                                    "parser_search_score": getattr(candidate, "search_score", None),
                                    "syntax_complete": True}
                        metadata.update({"semantic_candidate_index": semantic_index,
                                         "semantic_choices": tuple(asdict(c) for c in semantic.choices) if semantic is not None else (),
                                         "semantic_unresolved": tuple(asdict(c) for c in semantic.unresolved) if semantic is not None else (),
                                         "semantic_search": {"max_candidates": self.semantic_max_candidates,
                                             "max_expansions": semantic_budget, "explored": semantics.explored,
                                             "pending": semantics.pending, "truncated": semantics.truncated,
                                             "reason": getattr(semantics, "reason", None)}})
                        if not acts:
                            metadata["unresolved"] = "semantic projection unresolved or exhausted"
                        batch.append(SentenceAlternative(None, acts, () if acts else tuple(words),
                                                        provenance="learned-reader-candidate", metadata=metadata))
                batches.append(batch)
            # Round-robin keeps multiple tag hypotheses represented under a cap.
            # This is a retention policy, not a combined semantic-confidence rank.
            alternatives = []
            seen = {}
            for position in range(max((len(batch) for batch in batches), default=0)):
                for batch in batches:
                    if position >= len(batch):
                        continue
                    alternative = batch[position]
                    metadata = alternative.metadata
                    signature = (metadata["tags"], tuple(sorted(metadata["heads"].items())),
                                 tuple(sorted(metadata["labels"].items())), metadata["semantic_candidate_index"])
                    path = {"method": metadata["parser_method"], "score": metadata["parser_score"],
                            "search_score": metadata["parser_search_score"], "ranking": metadata["parser_ranking"],
                            "transitions": metadata["transitions"]}
                    if signature not in seen:
                        seen[signature] = alternative
                        alternative.metadata["decoder_proposals"] = (path,)
                        alternatives.append(alternative)
                    else:
                        retained = seen[signature]
                        retained.metadata["decoder_proposals"] += (path,)
            discarded = max(0, len(alternatives) - self.max_alternatives)
            alternatives = alternatives[:self.max_alternatives]
            if not alternatives:
                alternatives = [SentenceAlternative(None, (), tuple(words), provenance="learned-reader-unresolved",
                                                    metadata={**common, "syntax_complete": False,
                                                              "unresolved": "no complete learned dependency candidate"})]
            for alternative in alternatives:
                alternative.metadata.update({"parse_searches": tuple(parse_searches),
                                             "sentence_semantic_budget": self.max_sentence_semantic_expansions,
                                             "sentence_semantic_expansions": self.max_sentence_semantic_expansions - remaining_semantic,
                                             "sentence_search_budget": self.max_sentence_expansions,
                                             "sentence_search_expansions": self.max_sentence_expansions - remaining_expansions,
                                             "greedy_tagging_tokens": len(words),
                                             "budget_policy": "per-search cap and fair remaining budget across tag lanes",
                                             "proposal_limit": self.max_alternatives,
                                             "proposals_discarded": discarded,
                                             "proposal_retention": "validated greedy proposal first; round-robin across tag candidates",
                                             "search_truncated": bool(discarded or tag_search.truncated or
                                                                      any(search.truncated for search, _ in semantic_cache.values()) or
                                                                      any(p["truncated"] or
                                                                          (p["greedy_search"] is not None and p["greedy_search"]["truncated"])
                                                                          for p in parse_searches))})
            alternatives = tuple(deepcopy(alternative) for alternative in alternatives)
            first = alternatives[0]
            out.append(Sentence(raw, tuple(words), None, first.acts, first.skipped, (),
                                round((time.perf_counter() - t0) * 1000, 1), alternatives))
        return out
