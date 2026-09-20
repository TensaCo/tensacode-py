"""Explicit segmentation fixtures isolate source anchoring and shared decoder budgets."""
from types import MethodType, SimpleNamespace
from dependency_meaning_fixtures import neutral_fixture

from tensorcode.agent.understand import Act, LearnedReader, Sentence, SentenceAlternative
from tensorcode.language import Frame, Request


class SuppliedSegmenter:
    """Authored test proposals; no learned language capability is claimed here."""
    def __init__(self, spans):
        self.spans, self.calls = spans, []

    def segment(self, text, **budget):
        self.calls.append((text, budget))
        spent = min(3, budget["max_expansions"])
        candidates = tuple(SimpleNamespace(spans=spans, score=float(-i), provenance=("authored-test-proposal",))
                           for i, spans in enumerate(self.spans)) if spent == 3 else ()
        return SimpleNamespace(candidates=candidates, expansions=spent, truncated=True,
                               complete=False, reason="fixture-budget")


def supplied_reader(spans):
    reader = object.__new__(LearnedReader)
    reader.segmenter = SuppliedSegmenter(spans)
    reader.segmentation_artifact = {"path": "test-only", "sha256": "supplied-fixture"}
    reader.segmentation_error = None
    reader.segmentation_beam_width, reader.segmentation_max_candidates = 4, 2
    reader.segmentation_max_expansions = 100
    reader.max_sentence_expansions, reader.max_sentence_semantic_expansions = 20, 10
    reader.max_alternatives = 16
    reader.semantic_max_candidates, reader.semantic_max_expansions = 4, 64
    reader.model_artifact = {"sha256": "authored-decoder-fixture"}
    reader.decoded = []

    def decode(self, raw, raw_start, words, anchors, *, search_budget, semantic_total_budget, segmentation_metadata):
        self.decoded.append((raw, tuple(words), anchors, search_budget, semantic_total_budget))
        alternative = SentenceAlternative(None, (), provenance="authored-decoder-fixture", metadata={
            **segmentation_metadata, "tokens": tuple(words), "syntax_complete": True,
            "tags": tuple("X" for _ in words), "lemmas": tuple(words),
            "heads": {i: 0 if i == 1 else 1 for i in range(1, len(words) + 1)},
            "labels": {i: "root" if i == 1 else "dep" for i in range(1, len(words) + 1)},
            "sentence_search_expansions": min(search_budget, 5), "sentence_semantic_expansions": 0,
            "search_truncated": False})
        return Sentence(raw, tuple(words), None, (), alternatives=(alternative,))
    reader._decode_segment = MethodType(decode, reader)
    from tensorcode.language.deps_semantics import SemanticReadCandidate, SemanticReadCandidates
    frame = Frame("fixture_action", {}, {"mood": "imperative"})

    class SuppliedFrontier:
        explored = 0

        def __init__(self, args):
            self.neutral = neutral_fixture(frame, *args)

        def advance(self, *, max_expansions, max_candidates):
            if not self.explored and max_expansions and max_candidates:
                self.explored = 1
                return SemanticReadCandidates((SemanticReadCandidate((self.neutral,)),), False, 1, 0)
            return SemanticReadCandidates((), not self.explored, self.explored, int(not self.explored))

    reader.reader = SimpleNamespace(start_candidates=lambda *args, **kwargs: SuppliedFrontier(args))
    return reader


def test_each_segmentation_retains_its_own_tokens_and_exact_original_spans(monkeypatch):
    import tensorcode.agent.understand as boundary

    monkeypatch.setattr(boundary, "tokenize", lambda text: (_ for _ in ()).throw(AssertionError("legacy tokenizer used")))
    reader = supplied_reader((((0, 2), (2, 5), (6, 8), (8, 9)), ((0, 5), (6, 8), (8, 9))))
    text = "  can't go.\ncan't go."
    sentences = reader.read(text)
    assert len(sentences) == 2
    for sentence in sentences:
        assert len(sentence.alternatives) == 2
        assert sentence.tokens == ("ca", "n't", "go", ".")
        assert sentence.alternatives[1].metadata["tokens"] == ("can't", "go", ".")
        for alternative in sentence.alternatives:
            metadata = alternative.metadata
            assert metadata["segmentation_score"]["kind"] == "uncalibrated"
            assert metadata["segmentation_artifact"] == reader.segmentation_artifact
            for anchor in metadata["token_anchors"]:
                start, end = anchor["char_span"]
                assert text[start:end] == anchor["token"]


def test_segmentation_and_decode_lanes_share_one_finite_expansion_budget():
    reader = supplied_reader((((0, 2), (2, 5), (6, 8), (8, 9)), ((0, 5), (6, 8), (8, 9))))
    sentence, = reader.read("can't go.")
    metadata = sentence.alternatives[0].metadata
    branches = metadata["segmentation_branches"]
    spent = metadata["segmentation_search"]["expansions"] + sum(branch["search_expansions"] for branch in branches)
    assert metadata["sentence_search_expansions"] == spent <= 20
    assert all(branch["search_expansions"] <= branch["search_budget"] for branch in branches)
    assert len(branches) == 2 and all(branch["search_budget"] > 0 for branch in branches)


def test_full_quote_source_reaches_segmenter_and_mention_boundary_remains_explicit():
    reader = supplied_reader((((0, 1), (1, 3), (3, 4)),))
    sentence, = reader.read('"go"')
    assert reader.segmenter.calls[0][0] == '"go"'
    assert sentence.tokens == ('"', "go", '"')
    assert sentence.acts[0].kind == "unresolved"
    assert sentence.acts[0].frame is None
    metadata = sentence.alternatives[0].metadata
    assert metadata["quotation"]["applied"]
    assert metadata["token_anchors"][0]["char_span"] == (0, 1)


def test_empty_quotation_is_segmented_but_does_not_produce_invented_acts():
    reader = supplied_reader((((0, 1), (1, 2)),))
    sentence, = reader.read('""')
    assert reader.segmenter.calls[0][0] == '""'
    assert not reader.decoded and not sentence.acts
    assert sentence.tokens == ('"', '"')
    assert "no lexical content" in sentence.alternatives[0].metadata["unresolved"]


def test_unavailable_segmentation_retains_unresolved_input_without_decoder_fallback():
    reader = supplied_reader(())
    reader.segmenter = None
    reader.segmentation_error = "FileNotFoundError: missing learned segmenter"
    sentence, = reader.read("unseen input")
    assert not reader.decoded and not sentence.acts and sentence.tokens == ()
    assert sentence.coverage == 0
    candidate, = sentence.alternatives
    assert candidate.metadata["tokens"] == () and candidate.metadata["token_anchors"] == ()
    assert candidate.metadata["sentence_span"] == (0, 12)
    assert "missing learned segmenter" in candidate.metadata["unresolved"]


def test_exhausted_segmentation_and_invalid_spans_never_become_complete_readings():
    reader = supplied_reader((((0, 2),),))
    reader.max_sentence_expansions = 1
    exhausted, = reader.read("raw")
    assert not reader.decoded and not exhausted.acts
    assert exhausted.coverage == 0
    assert exhausted.alternatives[0].metadata["sentence_search_expansions"] == 1
    reader.max_sentence_expansions = 20
    invalid, = reader.read("raw")
    assert not reader.decoded and not invalid.acts
    assert "uncovered" in invalid.alternatives[0].metadata["unresolved"]


def test_missing_or_corrupt_segmenter_artifact_is_retained_as_unresolved(tmp_path, monkeypatch):
    import tensorcode.language.deps_semantics as semantics
    import tensorcode.language.learned_parser as parsing
    import tensorcode.language.treebank as treebank

    # Isolate segmenter loading from the independent cached parser requirement.
    parser_path = tmp_path / "supplied-parser.fixture"
    parser_path.write_bytes(b"explicit test fixture")
    monkeypatch.setattr(parsing, "load_model", lambda path: (SimpleNamespace(), SimpleNamespace()))
    monkeypatch.setattr(treebank, "load", lambda split: [])
    monkeypatch.setattr(semantics, "Reader", lambda: SimpleNamespace())
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("not valid JSON")
    for path in (tmp_path / "absent.json", corrupt):
        reader = LearnedReader(parser_path, segmentation_model_path=path)
        sentence, = reader.read("retain this source")
        assert reader.segmenter is None and not sentence.acts
        assert sentence.coverage == 0
        candidate, = sentence.alternatives
        assert candidate.metadata["segmentation_artifact"]["path"] == str(path)
        assert candidate.metadata["tokens"] == ()
        assert candidate.metadata["unresolved"]
        assert candidate.metadata["sentence_search_expansions"] == 0
