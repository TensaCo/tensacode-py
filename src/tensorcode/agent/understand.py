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
    continuation: SentenceContinuation | None = field(default=None, compare=False, repr=False)

    @property
    def coverage(self) -> float:
        """Share of word tokens the parse used (punctuation does not count against it)."""
        if not self.tokens and self.text.strip():
            return 0.0
        words = [t for t in self.tokens if any(c.isalnum() for c in t)]
        missed = [t for t in self.skipped if any(c.isalnum() for c in t)]
        return 1.0 if not words else round(1 - len(missed) / len(words), 3)


@dataclass(frozen=True)
class ContinuationBatch:
    """New proposals and work performed; none is an interpretation decision."""

    alternatives: tuple[SentenceAlternative, ...]
    explored: int
    pending: int


class SentenceContinuation:
    """Detached in-memory semantic work over already generated syntax.

    Pending counts branches and unstarted families, not possible meanings.
    Round-robin one-expansion quanta are an authored scheduling policy. Syntax
    pruning remains outside this continuation; no decoding or dispatch occurs.
    """

    def __init__(self, raw, reader, states, deferred):
        from copy import deepcopy
        self.raw = raw
        self.reader = deepcopy(reader)
        self.states = deepcopy([{key: value for key, value in state.items() if key != "outputs"}
                                for state in states])
        for family in deepcopy(deferred):
            self.states.append(dict(family=family, cursor=None, explored=0, emitted=0,
                                    pending=1, deferred=True))
        self.position = 0
        self._ready = []

    @property
    def pending(self):
        return sum(state["pending"] for state in self.states) + len(self._ready)

    def advance(self, *, max_expansions: int, max_candidates: int) -> ContinuationBatch:
        from copy import deepcopy
        for value in (max_expansions, max_candidates):
            if type(value) is not int or value < 0:
                raise ValueError("continuation budgets must be nonnegative integers")
        if not max_expansions or not max_candidates:
            return ContinuationBatch((), 0, self.pending)
        # Projection, convention application and result copying can each fail.
        # Publish both cursor movement and proposals only after all succeed.
        working = deepcopy(self)
        batch = working._advance(max_expansions=max_expansions, max_candidates=max_candidates)
        self.__dict__.update(working.__dict__)
        return batch

    def _advance(self, *, max_expansions: int, max_candidates: int) -> ContinuationBatch:
        from copy import deepcopy
        from dataclasses import asdict
        explored = 0
        stalled = 0
        while self.states and len(self._ready) < max_candidates:
            state = self.states[self.position]
            self.position = (self.position + 1) % len(self.states)
            if not state["pending"] or explored >= max_expansions:
                stalled += 1
                if stalled >= len(self.states):
                    break
                continue
            family = state["family"]
            metadata = deepcopy(family.metadata)
            if state.get("deferred"):
                if metadata.get("syntax_complete"):
                    state["cursor"] = self.reader.start_candidates(metadata["tokens"], metadata["tags"],
                        metadata["lemmas"], metadata["heads"], metadata["labels"])
                else:
                    state["pending"] = 0
                    self._ready.append(deepcopy(family))
                state["deferred"] = False
            if state["cursor"] is None:
                continue
            result = state["cursor"].advance(max_expansions=1, max_candidates=1)
            delta = result.explored - state["explored"]
            explored += delta
            state.update(explored=result.explored, pending=result.pending)
            stalled = 0 if delta or result.candidates else stalled + 1
            for semantic in result.candidates:
                acts = tuple(a for meaning in semantic.meanings for a in neutral_acts(meaning))
                metadata.pop("unresolved", None)
                metadata.update(semantic_candidate_index=state["emitted"],
                    semantic_choices=tuple(asdict(choice) for choice in semantic.choices),
                    semantic_unresolved=tuple(asdict(issue) for issue in semantic.unresolved),
                    semantic_projection_complete=False if not acts else None,
                    continuation_proposal=True)
                if not acts:
                    metadata["unresolved"] = "semantic projection unresolved"
                state["emitted"] += 1
                frontier = dict(explored=state["explored"], emitted=state["emitted"], pending=state["pending"],
                                truncated=bool(state["pending"]), reason="continuation budget" if state["pending"] else None)
                metadata.update(semantic_search=dict(frontier), semantic_frontier=dict(frontier),
                                search_truncated=bool(state["pending"] or metadata.get("search_truncated")))
                self._ready.append(SentenceAlternative(None, acts, () if acts else tuple(metadata["tokens"]),
                                                       provenance=family.provenance, metadata=metadata))
            if stalled >= len(self.states):
                break
        # Retain completed prefixes until the entire call succeeds, so a failure
        # in another family cannot silently lose completed proposals.
        delivered = tuple(deepcopy(self._ready[:max_candidates]))
        del self._ready[:max_candidates]
        return ContinuationBatch(delivered, explored, self.pending)


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


def neutral_acts(meaning):
    """Retain provisional structures without authored speech-act authority."""
    if isinstance(meaning, (Request, Question, Frame)):
        raise TypeError('learned syntax projection must supply ProvisionalMeaning, not semantic speech acts')
    return (Act('unresolved', meaning, None),)


class LearnedReader:
    """Bounded learned syntactic proposals, without selecting an interpretation.

    Tag and dependency scores remain separate, uncalibrated model scores. Complete
    syntax is not established semantic understanding: the semantic adapter is still
    authored and provisional. Speech-act authority requires an explicitly admitted
    learned model in the agent workspace. ``Sentence.acts`` exposes the first retained
    proposal; the interpretation workspace retains every alternative and its default
    policy defers. Empty searches retain an unresolved source rather than a repaired
    dependency root or an invented complete interpretation.
    """

    def __init__(self, model_path=None, *, tag_beam_width: int = 4, tag_max_candidates: int = 4,
                 parse_beam_width: int = 8, parse_max_candidates: int = 4,
                 max_expansions: int = 100000, max_alternatives: int = 16,
                 max_sentence_expansions: int = 600000, parse_ranking: str = "local_margin",
                 semantic_max_candidates: int = 4, semantic_max_expansions: int = 64,
                 max_sentence_semantic_expansions: int = 2048,
                 segmentation_model_path=None, segmentation_beam_width: int = 4,
                 segmentation_max_candidates: int = 2, segmentation_max_expansions: int = 100000) -> None:
        import hashlib
        from pathlib import Path

        from ..language.deps_semantics import Reader
        from ..language.learned_parser import load_model
        from ..language.treebank import lemma_table, lemmatize, load as load_treebank

        limits = (tag_beam_width, tag_max_candidates, parse_beam_width,
                  parse_max_candidates, max_expansions, max_alternatives, max_sentence_expansions,
                  semantic_max_candidates, semantic_max_expansions, max_sentence_semantic_expansions,
                  segmentation_beam_width, segmentation_max_candidates, segmentation_max_expansions)
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
        self.tag_beam_width, self.tag_max_candidates = tag_beam_width, tag_max_candidates
        self.parse_beam_width, self.parse_max_candidates = parse_beam_width, parse_max_candidates
        self.max_expansions, self.max_alternatives = max_expansions, max_alternatives
        from ..language.segmentation import DEFAULT_MODEL_PATH, load_model as load_segmenter
        segment_path = Path(segmentation_model_path or DEFAULT_MODEL_PATH)
        self.segmentation_beam_width = segmentation_beam_width
        self.segmentation_max_candidates = segmentation_max_candidates
        self.segmentation_max_expansions = segmentation_max_expansions
        self.segmentation_error = None
        try:
            self.segmenter = load_segmenter(segment_path)
            self.segmentation_artifact = dict(self.segmenter.metadata)
        except (OSError, ValueError) as error:
            self.segmenter = None
            self.segmentation_artifact = {"path": str(segment_path.resolve())}
            self.segmentation_error = f"{type(error).__name__}: {error}"

    @staticmethod
    def _search_metadata(search, beam_width, max_candidates, max_expansions):
        return {"beam_width": beam_width, "max_candidates": max_candidates,
                "max_expansions": max_expansions, "expansions": search.expansions,
                "complete": search.complete, "truncated": search.truncated, "reason": search.reason}

    def _decode_segment(self, raw, raw_start, words, anchors, *, search_budget,
                        semantic_total_budget, segmentation_metadata) -> Sentence:
        from copy import deepcopy
        import time

        t0 = time.perf_counter()
        inner = quoted(raw)
        quote_metadata = _quotation_metadata(raw, raw_start)
        message_cursor = raw_start + len(raw)
        remaining_expansions = search_budget
        tag_budget = min(self.max_expansions, remaining_expansions)
        tag_search = self.tagger.tag_candidates(words, beam_width=self.tag_beam_width,
                                                max_candidates=self.tag_max_candidates,
                                                max_expansions=tag_budget)
        remaining_expansions -= tag_search.expansions
        tag_metadata = self._search_metadata(tag_search, self.tag_beam_width,
                                             self.tag_max_candidates, tag_budget)
        common = {"model_artifact": self.model_artifact, "sentence_span": (raw_start, message_cursor),
                  "quotation": quote_metadata, "tokens": tuple(words), "token_anchors": tuple(anchors), "tag_search": tag_metadata,
                  **segmentation_metadata,
                  "semantic_adapter": "authored:deps_semantics.Reader provisional frames",
                  "speech_act_status": "unresolved",
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
        seen = {}
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
                path = {"method": method, "score": {"value": candidate.score, "kind": "uncalibrated"},
                        "search_score": getattr(candidate, "search_score", None),
                        "ranking": getattr(candidate, "ranking", "raw"), "transitions": tuple(candidate.transitions)}
                if syntax_key in seen:
                    seen[syntax_key].metadata["decoder_proposals"] += (path,)
                    continue
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
                            "decoder_proposals": (path,), "syntax_complete": True,
                            "semantic_projection_complete": False,
                            "unresolved": "semantic projection not scheduled"}
                family = SentenceAlternative(None, (), tuple(words),
                                             provenance="learned-reader-candidate", metadata=metadata)
                seen[syntax_key] = family
                batch.append(family)
            batches.append(batch)
        # Syntax breadth only: semantic variants cannot consume another tree's slot.
        alternatives = [batch[position] for position in range(max(map(len, batches), default=0))
                        for batch in batches if position < len(batch)]
        if not alternatives:
            alternatives = [SentenceAlternative(None, (), tuple(words), provenance="learned-reader-unresolved",
                                                metadata={**common, "syntax_complete": False, "semantic_projection_complete": False,
                                                          "unresolved": "no complete learned dependency candidate"})]
        for alternative in alternatives:
            alternative.metadata.update({"parse_searches": tuple(parse_searches),
                                         "sentence_semantic_budget": semantic_total_budget,
                                         "sentence_semantic_expansions": 0,
                                         "sentence_search_budget": search_budget,
                                         "sentence_search_expansions": search_budget - remaining_expansions,
                                         "greedy_tagging_tokens": len(words),
                                         "budget_policy": "per-search cap and fair remaining budget across tag lanes",
                                         "proposal_limit": self.max_alternatives,
                                         "proposals_discarded": 0,
                                         "proposal_retention": "unique syntax; round-robin across tag candidates",
                                         "search_truncated": bool(tag_search.truncated or
                                                                  any(p["truncated"] or
                                                                      (p["greedy_search"] is not None and p["greedy_search"]["truncated"])
                                                                      for p in parse_searches))})
        alternatives = tuple(deepcopy(alternative) for alternative in alternatives)
        first = alternatives[0]
        return Sentence(raw, tuple(words), None, first.acts, first.skipped, (),
                        round((time.perf_counter() - t0) * 1000, 1), alternatives)

    def _project_families(self, raw, families, *, capture=None):
        """Reserve syntax breadth, then emit additional semantic variants fairly.

        A slot is a retained hypothesis, never authority to execute. Unfinished
        projection is explicitly unresolved; queued semantic branches are not
        silently represented as exhausted or as additional fabricated readings.
        """
        from dataclasses import asdict

        remaining = self.max_sentence_semantic_expansions
        states = []
        for family in families:
            metadata = family.metadata
            cursor = (self.reader.start_candidates(metadata["tokens"], metadata["tags"],
                       metadata["lemmas"], metadata["heads"], metadata["labels"])
                      if metadata.get("syntax_complete") else None)
            states.append({"family": family, "cursor": cursor, "explored": 0,
                           "emitted": 0, "pending": int(cursor is not None), "outputs": []})
        outputs = []

        def advance(state, allowance):
            nonlocal remaining
            before = state["explored"]
            result = state["cursor"].advance(max_expansions=allowance, max_candidates=1)
            state.update(explored=result.explored, pending=result.pending)
            remaining -= result.explored - before
            if not result.candidates:
                return None
            semantic, = result.candidates
            family = state["family"]
            metadata = dict(family.metadata)
            acts = tuple(a for meaning in semantic.meanings for a in neutral_acts(meaning))
            metadata.update(semantic_candidate_index=state["emitted"],
                            semantic_choices=tuple(asdict(choice) for choice in semantic.choices),
                            semantic_unresolved=tuple(asdict(issue) for issue in semantic.unresolved))
            metadata.pop("unresolved", None)
            metadata["semantic_projection_complete"] = False if not acts else None
            if not acts:
                metadata["unresolved"] = "semantic projection unresolved"
            state["emitted"] += 1
            alternative = SentenceAlternative(None, acts, () if acts else tuple(metadata["tokens"]),
                                             provenance=family.provenance, metadata=metadata)
            state["outputs"].append(alternative)
            return alternative

        # Every family owns a first slot, even when its projection cannot finish.
        for index, state in enumerate(states):
            allowance = min(self.semantic_max_expansions, remaining // (len(states) - index))
            alternative = advance(state, allowance) if state["cursor"] is not None else None
            if alternative is None:
                family = state["family"]
                metadata = dict(family.metadata)
                metadata.update(semantic_projection_complete=False, semantic_candidate_index=None,
                                semantic_choices=(), semantic_unresolved=())
                metadata.setdefault("unresolved", "semantic projection not scheduled")
                alternative = SentenceAlternative(None, (), tuple(metadata["tokens"]),
                                                 provenance=family.provenance, metadata=metadata)
                state["outputs"].append(alternative)
            outputs.append(alternative)
        # Additional meanings never displace another retained syntax family.
        while remaining:
            progress = False
            for state in states:
                if not remaining:
                    break
                placeholder_pending = state["outputs"][0].metadata["semantic_candidate_index"] is None
                if len(outputs) >= self.max_alternatives and not placeholder_pending:
                    continue
                allowance = min(remaining, self.semantic_max_expansions - state["explored"])
                if not state["pending"] or allowance <= 0 or state["emitted"] >= self.semantic_max_candidates:
                    continue
                before = state["explored"]
                alternative = advance(state, allowance)
                progress |= state["explored"] > before
                if alternative is not None:
                    if state["outputs"][0].metadata["semantic_candidate_index"] is None:
                        placeholder = state["outputs"].pop(0)
                        outputs[next(i for i, value in enumerate(outputs) if value is placeholder)] = alternative
                    else:
                        outputs.append(alternative)
            if not progress:
                break
        for state in states:
            reason = ("sentence_semantic_budget" if not remaining else
                      "family_expansion_limit" if state["explored"] >= self.semantic_max_expansions else
                      "family_candidate_limit" if state["emitted"] >= self.semantic_max_candidates else
                      "global_retention_limit") if state["pending"] else None
            frontier = {"explored": state["explored"], "emitted": state["emitted"],
                        "pending": state["pending"], "reason": reason,
                        "truncated": bool(state["pending"]),
                        "max_candidates": self.semantic_max_candidates,
                        "max_expansions": self.semantic_max_expansions}
            for alternative in state["outputs"]:
                alternative.metadata.update(semantic_search=dict(frontier), semantic_frontier=dict(frontier))
                alternative.metadata["search_truncated"] = bool(state["pending"] or
                                                               alternative.metadata.get("search_truncated"))
        if capture is not None:
            capture.extend(states)
        return outputs, {"semantic_explored": self.max_sentence_semantic_expansions - remaining,
                         "semantic_emitted": sum(state["emitted"] for state in states),
                         "semantic_pending": sum(state["pending"] for state in states),
                         "semantic_unexpanded_families": sum(bool(state["cursor"] is not None and not state["explored"])
                                                             for state in states),
                         "semantic_discarded": 0}

    @staticmethod
    def _segmented_words(raw, spans):
        words, anchors = [], []
        previous = 0
        for index, span in enumerate(spans, 1):
            if (not isinstance(span, (tuple, list)) or len(span) != 2
                    or any(type(value) is not int for value in span)):
                raise ValueError("segmentation requires integer character-span pairs")
            start, end = span
            if not previous <= start < end <= len(raw) or any(not c.isspace() for c in raw[previous:start]):
                raise ValueError("segmentation spans overlap or omit source characters")
            token = raw[start:end]
            if any(c.isspace() for c in token):
                raise ValueError("segmentation token crosses a whitespace gap")
            words.append(token)
            anchors.append({"index": index, "token": token, "char_span": (start, end)})
            previous = end
        if any(not c.isspace() for c in raw[previous:]):
            raise ValueError("segmentation leaves non-whitespace source uncovered")
        return words, anchors

    def read(self, text: str) -> list[Sentence]:
        """Segment full source with a learned model, then retain bounded readings.

        Sentence splitting and whole-quotation mention treatment remain explicit
        authored conventions. Token boundaries come exclusively from segmentation
        proposals; unavailable models or exhausted searches do not invoke the chart
        tokenizer. Every alternative owns its tokens and exact source anchors.
        """
        from copy import deepcopy
        import time

        out, cursor = [], 0
        for raw in sentences(text):
            started = time.perf_counter()
            start = text.find(raw, cursor)
            if start < 0:
                raise ValueError("sentence cannot be anchored in its source")
            cursor = start + len(raw)
            common = {"model_artifact": self.model_artifact, "segmentation_artifact": self.segmentation_artifact,
                      "sentence_span": (start, cursor), "quotation": _quotation_metadata(raw, start),
                      "sentence_boundary_policy": "authored punctuation/newline splitter",
                      "tokens": (), "token_anchors": (), "syntax_complete": False,
                      "semantic_projection_complete": None}
            remaining = self.max_sentence_expansions
            remaining_semantic = self.max_sentence_semantic_expansions
            segmentation_budget = min(self.segmentation_max_expansions, remaining)
            segmentation_started = time.perf_counter()
            search = None
            unavailable = self.segmentation_error
            if self.segmenter is not None:
                try:
                    search = self.segmenter.segment(raw, beam_width=self.segmentation_beam_width,
                        max_candidates=self.segmentation_max_candidates, max_expansions=segmentation_budget)
                except (ValueError, TypeError) as error:
                    unavailable = f"segmentation failed: {type(error).__name__}: {error}"
            search_metadata = (self._search_metadata(search, self.segmentation_beam_width,
                               self.segmentation_max_candidates, segmentation_budget) if search is not None else
                               {"reason": unavailable or "segmentation model unavailable", "expansions": 0,
                                "truncated": False, "complete": False, "max_expansions": segmentation_budget})
            if search is not None:
                remaining -= search.expansions
            common["segmentation_search"] = search_metadata
            candidates = search.candidates if search is not None else ()
            segmentation_ms = (time.perf_counter() - segmentation_started) * 1000
            syntax_started = time.perf_counter()
            branches, branch_stats = [], []
            for index, segmentation in enumerate(candidates):
                metadata = {**common, "segmentation_index": index,
                            "segmentation_score": {"value": segmentation.score, "kind": "uncalibrated"},
                            "segmentation_provenance": tuple(segmentation.provenance),
                            "segmentation_spans": tuple(segmentation.spans)}
                reason = None
                try:
                    words, anchors = self._segmented_words(raw, segmentation.spans)
                    anchors = tuple({**anchor, "char_span": tuple(start + p for p in anchor["char_span"])} for anchor in anchors)
                except ValueError as error:
                    words, anchors, reason = [], (), str(error)
                metadata.update(tokens=tuple(words), token_anchors=anchors)
                lane_budget = remaining // (len(candidates) - index)
                semantic_budget = remaining_semantic // (len(candidates) - index)
                inner = quoted(raw)
                if inner is not None and not inner.strip():
                    reason = "quoted source has no lexical content"
                if not words:
                    reason = reason or "segmentation returned no source tokens"
                if lane_budget <= 0:
                    reason = reason or "sentence search budget exhausted before dependency decoding"
                spent = semantic_spent = 0
                if reason is not None:
                    alternatives = (SentenceAlternative(None, (), tuple(words), provenance="learned-reader-unresolved",
                                    metadata={**metadata, "unresolved": reason, "semantic_projection_complete": False}),)
                else:
                    sentence = self._decode_segment(raw, start, words, anchors, search_budget=lane_budget,
                                    semantic_total_budget=semantic_budget, segmentation_metadata=metadata)
                    alternatives = sentence.alternatives
                    spent = alternatives[0].metadata["sentence_search_expansions"]
                    semantic_spent = alternatives[0].metadata["sentence_semantic_expansions"]
                    remaining -= spent
                    remaining_semantic -= semantic_spent
                branch_stats.append({"segmentation_index": index, "search_budget": lane_budget,
                                     "search_expansions": spent, "semantic_budget": semantic_budget,
                                     "semantic_expansions": semantic_spent})
                branches.append(alternatives)
            syntax_ms = (time.perf_counter() - syntax_started) * 1000
            retention_started = time.perf_counter()
            families = [branch[position] for position in range(max(map(len, branches), default=0))
                        for branch in branches if position < len(branch)]
            # One sentence-wide cap, before any semantic projection. Distinct
            # segmentation boundaries remain distinct source interpretations.
            unique = {}
            for family in families:
                metadata = family.metadata
                signature = (tuple(metadata.get("segmentation_spans", ())), tuple(metadata.get("tags", ())),
                             tuple(sorted(metadata.get("heads", {}).items())),
                             tuple(sorted(metadata.get("labels", {}).items())), metadata.get("unresolved"))
                if signature not in unique:
                    unique[signature] = family
                else:
                    retained = unique[signature].metadata
                    retained["decoder_proposals"] = (retained.get("decoder_proposals", ()) +
                                                     metadata.get("decoder_proposals", ()))
            families = list(unique.values())
            retained = families[:self.max_alternatives]
            deferred = families[self.max_alternatives:]
            pending_syntax = tuple({**family.metadata, "semantic_projection_complete": False,
                                    "unresolved": "global syntax retention limit; semantic projection not run"}
                                   for family in deferred)
            discarded = len(deferred)
            retention_ms = (time.perf_counter() - retention_started) * 1000
            semantic_started = time.perf_counter()
            continuation_states = []
            alternatives, retention_stats = self._project_families(raw, retained, capture=continuation_states)
            continuation = SentenceContinuation(raw, self.reader, continuation_states, deferred)
            if not continuation.pending:
                continuation = None
            semantic_ms = (time.perf_counter() - semantic_started) * 1000
            remaining_semantic -= retention_stats["semantic_explored"]
            retention_stats.update(syntax_generated=sum(bool(f.metadata.get("syntax_complete")) for f in families),
                                   syntax_retained=sum(bool(f.metadata.get("syntax_complete")) for f in retained),
                                   syntax_discarded=sum(bool(f.metadata.get("syntax_complete")) for f in deferred))
            if not alternatives:
                alternatives = [SentenceAlternative(None, (), provenance="learned-reader-unresolved", metadata={
                    **common, "semantic_projection_complete": False,
                    "unresolved": unavailable or "no complete learned segmentation candidate"})]
            for alternative in alternatives:
                alternative.metadata.update({"sentence_search_budget": self.max_sentence_expansions,
                    "sentence_search_expansions": self.max_sentence_expansions - remaining,
                    "sentence_semantic_budget": self.max_sentence_semantic_expansions,
                    "sentence_semantic_expansions": self.max_sentence_semantic_expansions - remaining_semantic,
                    "segmentation_branches": tuple(branch_stats), "segment_proposals_discarded": 0,
                    "proposals_discarded": discarded, "pending_syntax_families": pending_syntax,
                    "retention_stats": retention_stats,
                    "reader_phase_ms": {"segmentation": segmentation_ms, "syntax": syntax_ms,
                                        "semantics": semantic_ms, "retention": retention_ms},
                    "proposal_retention": "global syntax breadth before additional semantic variants",
                    "segmentation_retention_policy": "round-robin across learned segmentations",
                    "search_truncated": bool(discarded or search_metadata["truncated"] or
                                             alternative.metadata.get("search_truncated", False))})
            alternatives = tuple(deepcopy(alternative) for alternative in alternatives)
            first = alternatives[0]
            out.append(Sentence(raw, first.metadata["tokens"], None, first.acts, first.skipped, (),
                                round((time.perf_counter() - started) * 1000, 1), alternatives, continuation))
        return out
