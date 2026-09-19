"""Learned syntax proposals remain alternatives with source and search evidence."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from tensorcode.agent.understand import LearnedReader
from tensorcode.language.deps_semantics import Reader
from tensorcode.language.conventions import request_conventions


MODEL = Path.home() / ".cache/tensorcode/models/ud_ewt_parser.pickle"
SEGMENTATION_MODEL = MODEL.with_name("ud_ewt_segmenter.json")


def search(candidates, *, reason=None):
    return SimpleNamespace(candidates=tuple(candidates), complete=False, truncated=True,
                           expansions=7, reason=reason)


class FixtureTagger:
    """Authored decoder output isolates retention; no linguistic inference claimed."""
    def tag_candidates(self, words, **bounds):
        self.bounds = bounds
        first = tuple("VERB" if index == 1 else "NOUN" for index, _ in enumerate(words))
        return search([SimpleNamespace(tags=first, score=9.0),
                       SimpleNamespace(tags=tuple("NOUN" for _ in words), score=4.0)])

    def greedy_candidate(self, words):
        return None

    def tag(self, words):
        raise AssertionError("greedy tagger must not be used")


class FixtureParser:
    def parse_candidates(self, words, tags, **bounds):
        self.bounds = bounds
        n = len(words)
        rows = []
        for root in (2, 1):
            heads = {i: (0 if i == root else root) for i in range(1, n + 1)}
            labels = {i: ("root" if i == root else "nsubj") for i in range(1, n + 1)}
            rows.append(SimpleNamespace(heads=heads, labels=labels, score=float(root), transitions=("fixture",)))
        return search(rows)

    def parse(self, words, tags):
        raise AssertionError("repaired greedy parser must not be used")


def fixture_reader(limit=16):
    reader = object.__new__(LearnedReader)
    from segmentation_fixtures import install_segmentation_fixture
    install_segmentation_fixture(reader)
    reader.tagger, reader.parser = FixtureTagger(), FixtureParser()
    reader.table = {}
    reader.lemmatize = lambda word, tag, table: word.lower()
    reader.reader = Reader()
    reader.conventions = request_conventions()
    reader.model_artifact = {"path": "fixture:decoder", "sha256": "authored-test-fixture"}
    reader.tag_beam_width, reader.tag_max_candidates = 4, 4
    reader.parse_beam_width, reader.parse_max_candidates = 8, 4
    reader.max_expansions, reader.max_alternatives = 100, limit
    reader.max_sentence_expansions = 600000
    reader.parse_ranking = "local_margin"
    reader.semantic_max_candidates = 4
    reader.semantic_max_expansions = 64
    reader.max_sentence_semantic_expansions = 2048
    return reader


def test_bounded_candidates_keep_distinct_syntax_and_separate_scores():
    reader = fixture_reader(limit=3)
    sentence, = reader.read("birds fly.")
    assert len(sentence.alternatives) == 3
    signatures = {(a.metadata["tags"], tuple(a.metadata["heads"].items())) for a in sentence.alternatives}
    assert len(signatures) == 3
    assert len({a.metadata["tags"] for a in sentence.alternatives}) == 2
    for alternative in sentence.alternatives:
        metadata = alternative.metadata
        assert metadata["tag_score"]["kind"] == metadata["parser_score"]["kind"] == "uncalibrated"
        assert "score" not in metadata  # there is no invented combined confidence
        assert metadata["semantic_projection_complete"] is (None if alternative.acts else False)
        assert metadata["proposals_discarded"] == 1 and metadata["search_truncated"]
        assert metadata["model_artifact"] == reader.model_artifact
        assert metadata["tag_search"]["max_expansions"] == 100
        assert metadata["parse_search"]["beam_width"] == 8
    assert sentence.acts == sentence.alternatives[0].acts


def test_token_anchors_point_into_original_message_including_quotes_and_repetition():
    text = '  "birds fly."\nbirds fly.'
    first, second = fixture_reader().read(text)
    for sentence in (first, second):
        for candidate in sentence.alternatives:
            metadata = candidate.metadata
            start, end = metadata["sentence_span"]
            assert text[start:end] == sentence.text
            for anchor in metadata["token_anchors"]:
                lo, hi = anchor["char_span"]
                assert text[lo:hi] == anchor["token"]
    assert all(act.kind == "mention" for alternative in first.alternatives for act in alternative.acts)
    assert first.alternatives[0].metadata["sentence_span"][1] < second.alternatives[0].metadata["sentence_span"][0]


def test_no_complete_candidate_retains_unresolved_source_without_repaired_root():
    reader = fixture_reader()
    reader.parser.parse_candidates = lambda *a, **k: search([], reason="budget_exhausted")
    sentence, = reader.read("birds fly.")
    assert not sentence.acts and sentence.skipped == sentence.tokens and sentence.coverage == 0
    alternative, = sentence.alternatives
    assert not alternative.metadata["syntax_complete"]
    assert "heads" not in alternative.metadata
    assert all(p["reason"] == "budget_exhausted" for p in alternative.metadata["parse_searches"])


def test_no_tag_candidate_retains_search_budget_and_source():
    reader = fixture_reader()
    reader.tagger.tag_candidates = lambda *a, **k: search([], reason="budget_exhausted")
    sentence, = reader.read("birds fly.")
    alternative, = sentence.alternatives
    assert alternative.metadata["tag_search"]["reason"] == "budget_exhausted"
    assert alternative.metadata["parse_searches"] == ()
    assert not alternative.acts and sentence.coverage == 0


def test_workspace_retains_metadata_without_default_selection_or_aliasing():
    from tensorcode.agent.core import Agent

    reader = fixture_reader()
    agent = Agent([], reader=reader)
    message = agent.interpret("birds fly.")
    group, = [agent.interpretations.get(i) for i in message.group_ids]
    assert group.selected_id is None and len(group.candidates) == 4
    source = agent.interpretations.get_source(group.source_id)
    assert source.text == "birds fly."
    candidate = group.candidates[0]
    candidate.payload.metadata["heads"][1] = 999
    assert agent.interpretations.get(group.id).candidates[0].payload.metadata["heads"][1] != 999


@pytest.mark.skipif(not MODEL.exists() or not SEGMENTATION_MODEL.exists(), reason="requires trained UD parser and segmenter artifacts")
def test_actual_learned_model_proposes_multiple_complete_syntactic_alternatives():
    reader = LearnedReader(MODEL)
    sentence, = reader.read("make a python hello world project")
    assert len(sentence.alternatives) > 1
    assert len(reader.model_artifact["sha256"]) == 64
    for alternative in sentence.alternatives:
        metadata = alternative.metadata
        assert metadata["syntax_complete"]
        assert set(metadata["heads"]) == set(range(1, len(metadata["tokens"]) + 1))
        assert metadata["tag_score"]["kind"] == "uncalibrated"
        assert metadata["parser_score"]["kind"] == "uncalibrated"
        assert metadata["semantic_projection_complete"] is (None if alternative.acts else False)
        assert metadata["sentence_search_expansions"] <= metadata["sentence_search_budget"]
    reader.max_sentence_expansions = 1
    exhausted, = reader.read("make a python hello world project")
    assert exhausted.coverage == 0 and not exhausted.acts
    assert exhausted.alternatives[0].metadata["sentence_search_expansions"] <= 1
    assert not exhausted.alternatives[0].metadata["syntax_complete"]


def test_validated_greedy_lane_is_retained_with_provenance_without_hiding_alternatives():
    reader = fixture_reader()
    reader.tagger.greedy_candidate = lambda words: SimpleNamespace(
        tags=tuple("VERB" if i == 1 else "NOUN" for i, _ in enumerate(words)), score=9.0,
        provenance=("training_lexicon", "perceptron_uncovered"), lexical_positions=(0,))
    reader.parser.greedy_search = lambda words, tags, **bounds: search([
        reader.parser.parse_candidates(words, tags).candidates[0]])
    sentence, = reader.read("birds fly.")
    assert len(sentence.alternatives) == 4
    first = sentence.alternatives[0]
    assert first.metadata["parser_method"] == "greedy-unrepaired"
    assert first.metadata["tag_proposals"][0]["lexical_positions_zero_based"] == (0,)
    assert {p["method"] for p in first.metadata["decoder_proposals"]} == {"greedy-unrepaired", "bounded-search"}
    assert first.metadata["greedy_search"]["max_steps"] == 22
    assert len({a.metadata["tags"] for a in sentence.alternatives}) == 2


def test_failed_unrepaired_greedy_lane_is_explicit_and_does_not_remove_search_candidates():
    reader = fixture_reader()
    reader.tagger.greedy_candidate = lambda words: reader.tagger.tag_candidates(words).candidates[0]
    reader.parser.greedy_search = lambda *a, **k: search([], reason="invalid_greedy_transition")
    sentence, = reader.read("birds fly.")
    assert len(sentence.alternatives) == 4
    first = sentence.alternatives[0]
    assert first.metadata["parser_method"] == "bounded-search"
    assert first.metadata["greedy_search"]["reason"] == "invalid_greedy_transition"


def preposition_fixture(priors):
    reader = fixture_reader()
    tags = ("NOUN", "VERB", "ADP", "NOUN")
    reader.tagger.tag_candidates = lambda *a, **k: search([SimpleNamespace(tags=tags, score=2.0)])
    reader.parser.parse_candidates = lambda *a, **k: search([SimpleNamespace(
        heads={1: 2, 2: 0, 3: 4, 4: 2}, labels={1: "nsubj", 2: "root", 3: "case", 4: "obl"},
        score=3.0, transitions=())])
    reader.reader = Reader(prepositions=priors, preposition_provenance="authored:reader-fixture")
    return reader


def test_semantic_role_alternatives_survive_identical_syntax_with_occurrence_evidence():
    reader = preposition_fixture({"with": [("instrument", -0.2), ("manner", -0.8)]})
    sentence, = reader.read("birds fly with wings")
    assert len(sentence.alternatives) == 2
    roles = {next(iter(a.metadata["semantic_choices"]))["role"] for a in sentence.alternatives}
    assert roles == {"instrument", "manner"}
    assert all(a.metadata["semantic_choices"][0]["dependent_token"] == 4 for a in sentence.alternatives)
    assert all(a.metadata["semantic_choices"][0]["provenance"] == "authored:reader-fixture" for a in sentence.alternatives)
    assert {tuple(a.acts[0].frame.roles) for a in sentence.alternatives} == {("subject", "instrument"), ("subject", "manner")}


def test_unknown_semantic_relation_preserves_tree_without_executable_meaning():
    reader = preposition_fixture({})
    sentence, = reader.read("birds fly with wings")
    candidate, = sentence.alternatives
    assert not candidate.acts and sentence.coverage == 0
    assert candidate.metadata["syntax_complete"]
    assert candidate.metadata["semantic_unresolved"][0]["dependent_token"] == 4
    assert candidate.metadata["heads"] == {1: 2, 2: 0, 3: 4, 4: 2}


def test_semantic_expansion_budget_is_aggregate_and_exhaustion_is_retained():
    reader = preposition_fixture({"with": [("instrument", -0.2), ("manner", -0.8)]})
    reader.max_sentence_semantic_expansions = 1
    sentence, = reader.read("birds fly with wings")
    candidate, = sentence.alternatives
    assert not candidate.acts
    assert candidate.metadata["sentence_semantic_expansions"] <= 1
    assert candidate.metadata["semantic_search"]["truncated"]
    assert candidate.metadata["search_truncated"]
