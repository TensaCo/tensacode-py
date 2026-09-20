"""Reader proposals survive the boundary without acquiring certainty or authority."""

from types import SimpleNamespace
from dependency_meaning_fixtures import neutral_fixture

from tensorcode.agent.understand import LearnedReader, Sentence, parse_one, read
from tensorcode.language import ENGLISH, Entity, Frame, Request, understand


def test_ambiguous_grammar_preserves_reader_order_coverage_and_distinct_acts():
    text = "make a python hello world project"
    source = understand(ENGLISH, text)
    sentence = parse_one(ENGLISH, text)
    assert len(sentence.alternatives) == len(source.readings) > 1
    assert [a.reading for a in sentence.alternatives] == list(source.readings)
    assert len({tuple(a.kind for a in c.acts) for c in sentence.alternatives}) > 1
    for candidate, reading in zip(sentence.alternatives, source.readings):
        assert candidate.skipped == tuple(word for _, word in reading.skipped)
        assert candidate.guessed == reading.guessed
        assert candidate.provenance == "grammar"
    first = sentence.alternatives[0]
    assert (sentence.reading, sentence.acts, sentence.skipped, sentence.guessed) == (
        first.reading, first.acts, first.skipped, first.guessed)


def test_quotation_cannot_expose_executable_acts_through_alternatives():
    sentence, = read(ENGLISH, '"open the file"')
    assert len(sentence.alternatives) > 1
    assert sentence.acts
    assert all(act.kind == "mention" for candidate in sentence.alternatives for act in candidate.acts)
    assert all(act.kind == "mention" for act in sentence.acts)


def test_colon_composition_does_not_misrepresent_head_as_full_candidate_parse():
    sentence, = read(ENGLISH, "design:\nthe artifact itself\nits bom")
    candidate, = sentence.alternatives
    assert candidate.provenance == "colon-composition-selected"
    assert candidate.reading is None
    assert candidate.acts == sentence.acts
    assert len(candidate.acts[0].frame.roles["object"]) == 2
    assert candidate.skipped == sentence.skipped
    assert candidate.guessed == sentence.guessed


def test_existing_positional_sentence_constructor_remains_valid():
    sentence = Sentence("opaque", ("opaque",), None, (), ("opaque",), (), 3.2)
    assert sentence.alternatives == ()
    assert sentence.parse_ms == 3.2
    assert sentence.coverage == 0


def test_learned_reader_retains_a_single_returned_candidate_and_preserves_quotation():
    # Exercise the boundary without requiring a downloaded treebank/model. This
    # parser returns one hypothesis; that fact must not masquerade as certainty.
    reader = LearnedReader.__new__(LearnedReader)
    from segmentation_fixtures import install_segmentation_fixture
    install_segmentation_fixture(reader)
    def candidates(rows):
        return SimpleNamespace(candidates=tuple(rows), complete=False, truncated=True,
                               expansions=3, reason="candidate_limit")

    reader.tagger = SimpleNamespace(greedy_candidate=lambda words: None, tag_candidates=lambda words, **bounds: candidates([
        SimpleNamespace(tags=tuple("X" for _ in words), score=0.0)]))
    reader.parser = SimpleNamespace(parse_candidates=lambda words, tags, **bounds: candidates([
        SimpleNamespace(heads={i: 0 if i == 1 else 1 for i in range(1, len(words) + 1)},
                        labels={i: "root" if i == 1 else "dep" for i in range(1, len(words) + 1)},
                        score=0.0, transitions=())]))
    reader.model_artifact = {"path": "fixture", "sha256": "authored-fixture"}
    reader.tag_beam_width = reader.tag_max_candidates = 1
    reader.parse_beam_width = reader.parse_max_candidates = 1
    reader.max_expansions = 10
    reader.max_alternatives = 1
    reader.max_sentence_expansions = 600000
    reader.parse_ranking = "local_margin"
    reader.semantic_max_candidates = 4
    reader.semantic_max_expansions = 64
    reader.max_sentence_semantic_expansions = 2048
    reader.lemmatize = lambda word, tag, table: word
    reader.table = {}
    frame = Frame("open", {"object": Entity("file", "file")}, {"mood": "imperative"})
    from tensorcode.language.deps_semantics import SemanticReadCandidate, SemanticReadCandidates
    class SuppliedFrontier:
        """Authored semantic output isolates reader retention, not inference."""
        explored = 0

        def __init__(self, args):
            self.neutral = neutral_fixture(frame, *args)

        def advance(self, *, max_expansions, max_candidates):
            if not self.explored and max_expansions and max_candidates:
                self.explored = 1
                return SemanticReadCandidates((SemanticReadCandidate((self.neutral,)),), False, 1, 0)
            return SemanticReadCandidates((), not self.explored, self.explored, int(not self.explored))

    reader.reader = SimpleNamespace(start_candidates=lambda *args, **kwargs: SuppliedFrontier(args))
    for text in ("open the file", '"open the file"'):
        sentence, = reader.read(text)
        candidate, = sentence.alternatives
        assert candidate.provenance == "learned-reader-candidate"
        assert candidate.metadata["search_truncated"]
        assert candidate.metadata["semantic_projection_complete"] is None
        assert candidate.reading is None
        assert candidate.acts == sentence.acts
        assert [act.kind for act in candidate.acts] == ["unresolved"]
        assert candidate.acts[0].frame is None
        assert candidate.acts[0].meaning.frame == frame
