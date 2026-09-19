"""Learned beams retain alternatives without inventing complete dependency trees."""
from pathlib import Path

import pytest

from tensorcode.language.learned_parser import (
    LEFT, REDUCE, RIGHT, SHIFT, Parser, Perceptron, State, Tagger,
    _candidate_moves, _complete_tree, load_model, train,
)
from tensorcode.language.treebank import Token


def assert_tree(candidate, count):
    assert set(candidate.heads) == set(candidate.labels) == set(range(1, count + 1))
    if count:
        assert list(candidate.heads.values()).count(0) == 1
    for dependent, head in candidate.heads.items():
        assert head in range(count + 1) and head != dependent
        assert (head == 0) == (candidate.labels[dependent] == "root")
        seen = set()
        current = dependent
        while current:
            assert current not in seen
            seen.add(current)
            current = candidate.heads[current]
    assert candidate.score_kind == "uncalibrated"


def tiny_parser():
    return Parser(moves=(SHIFT, REDUCE, "right|root", "right|dep", "left|dep"))


def test_tag_beam_keeps_binary_branch_that_wins_after_later_context():
    model = Perceptron()
    model.weights["w=one"] = {"A": 2.0, "B": 1.0}
    model.weights["t-1 w=B two"] = {"A": 5.0}
    tagger = Tagger(model, ("A", "B", "A"), known={"one": "A", "two": "B"})
    result = tagger.tag_candidates(["one", "two"], beam_width=4, max_candidates=4)
    assert result.candidates[0].tags == ("B", "A")
    assert result.candidates[0].score == 6.0
    assert len({candidate.tags for candidate in result.candidates}) == 4
    assert result.complete and not result.truncated
    assert result.expansions == 6
    assert all(candidate.score_kind == "uncalibrated" for candidate in result.candidates)
    # The legacy learned-word shortcut stays confined to the greedy baseline.
    assert tagger.tag(["one", "two"]) == ["A", "B"]


def test_parser_returns_both_structural_roots_without_duplicate_trees():
    parser = tiny_parser()
    parser.moves += ("right|root", "left|dep")
    result = parser.parse_candidates(["a", "b"], ["X", "X"], beam_width=100, max_candidates=100)
    assert result.complete and not result.truncated
    assert {tuple(sorted(candidate.heads.items())) for candidate in result.candidates} == {
        ((1, 0), (2, 1)), ((1, 2), (2, 0)),
    }
    for candidate in result.candidates:
        assert_tree(candidate, 2)
    again = parser.parse_candidates(["a", "b"], ["X", "X"], beam_width=100, max_candidates=100)
    assert again == result  # Tie order is reproducible, not a semantic selection.


def test_parser_uses_learned_weights_and_can_keep_a_lower_scoring_initial_branch():
    parser = tiny_parser()
    parser.model.weights["stack=1"] = {"right|root": 1.0}
    parser.model.weights["s0w=a"] = {"left|dep": 10.0}
    result = parser.parse_candidates(["a", "b"], ["X", "X"], beam_width=100, max_candidates=100)
    assert result.candidates[0].heads == {1: 2, 2: 0}
    assert result.candidates[0].score > result.candidates[1].score
    assert "left|dep" in result.candidates[0].transitions


def test_small_treebank_training_produces_ranked_alternatives():
    corpus = [
        [Token(1, "can", "can", "VERB", (), 0, "root"), Token(2, "fish", "fish", "NOUN", (), 1, "obj")],
        [Token(1, "fish", "fish", "VERB", (), 0, "root"), Token(2, "can", "can", "NOUN", (), 1, "obj")],
    ] * 4
    tagger, parser = train(corpus, tag_epochs=5, parse_epochs=5)
    tags = tagger.tag_candidates(["can", "fish"], beam_width=8, max_candidates=8)
    assert tags.candidates[0].tags == ("VERB", "NOUN")
    assert tags.candidates[0].score > tags.candidates[-1].score
    parses = parser.parse_candidates(["can", "fish"], tags.candidates[0].tags, beam_width=64, max_candidates=8)
    assert parses.candidates
    assert parses.candidates[0].heads == {1: 0, 2: 1}
    assert parses.candidates[0].labels == {1: "root", 2: "obj"}
    for candidate in parses.candidates:
        assert_tree(candidate, 2)


@pytest.mark.parametrize("inventory", [(), (SHIFT,), (REDUCE,), ("right|dep",), ("left|root",), ("invented",)])
def test_missing_or_illegal_moves_never_fabricate_root_attachments(inventory):
    result = Parser(moves=inventory).parse_candidates(["a"], ["X"], beam_width=8)
    assert not result.candidates
    assert result.complete and not result.truncated
    assert result.reason == "no_complete_candidates"


def test_exhausted_buffer_has_no_shift_fallback_or_phantom_token():
    state = State(1, stack=[0, 1], next=2)
    assert _candidate_moves(state, (SHIFT, REDUCE, "left|dep", "right|root")) == ()
    assert not _complete_tree(state)
    state = State(2, stack=[0], next=3, heads={1: 2, 2: 1}, labels={1: "dep", 2: "dep"})
    assert not _complete_tree(state)
    state = State(1, stack=[0], next=2, heads={1: 0, 2: 1}, labels={1: "root", 2: "dep"})
    assert not _complete_tree(state)


def test_budgets_never_present_incomplete_states_as_complete_candidates():
    tagger = Tagger(tags=("A", "B"))
    tags = tagger.tag_candidates(["a", "b"], max_expansions=1)
    assert tags.candidates == () and tags.truncated and not tags.complete
    assert tags.reason == "budget_exhausted" and tags.expansions == 1
    parser = tiny_parser()
    for budget in (0, 1, 2):
        result = parser.parse_candidates(["a", "b"], ["X", "X"], max_expansions=budget)
        assert result.expansions <= budget
        assert result.truncated and result.reason == "budget_exhausted"
        assert not result.complete
        for candidate in result.candidates:
            assert_tree(candidate, 2)
    assert parser.parse_candidates(["a", "b"], ["X", "X"], max_expansions=1).candidates == ()


def test_beam_and_output_truncation_are_explicit():
    tagger = Tagger(tags=("A", "B"))
    assert tagger.tag_candidates(["x"], beam_width=1).truncated
    assert tagger.tag_candidates(["x"], beam_width=4, max_candidates=1).reason == "candidate_limit"
    parser = tiny_parser()
    result = parser.parse_candidates(["a", "b"], ["X", "X"], beam_width=100, max_candidates=1)
    assert len(result.candidates) == 1
    assert result.truncated and not result.complete and result.reason == "candidate_limit"


def test_empty_input_is_a_complete_empty_structure_without_expansion():
    tagged = Tagger().tag_candidates([], max_expansions=0)
    parsed = Parser().parse_candidates([], [], max_expansions=0)
    assert tagged.candidates[0].tags == ()
    assert parsed.candidates[0].heads == parsed.candidates[0].labels == {}
    assert parsed.candidates[0].transitions == ()
    assert tagged.complete and parsed.complete and not tagged.truncated and not parsed.truncated
    assert tagged.expansions == parsed.expansions == 0


@pytest.mark.parametrize("options", [{"beam_width": 0}, {"max_candidates": 0}, {"max_expansions": -1}, {"beam_width": True}, {"max_expansions": 1.5}])
def test_search_controls_validate_without_coercion(options):
    with pytest.raises(ValueError):
        Tagger().tag_candidates([], **options)
    with pytest.raises(ValueError):
        Parser().parse_candidates([], [], **options)
    with pytest.raises(ValueError, match="same length"):
        Parser().parse_candidates(["a"], [])


def test_cached_model_optional_smoke_has_only_legal_complete_candidates():
    path = Path.home() / ".cache/tensorcode/models/ud_ewt_parser.pickle"
    if not path.exists():
        pytest.skip("no cached learned model")
    tagger, parser = load_model(path)
    words = ["Birds", "fly", "."]
    tags = tagger.tag_candidates(words, beam_width=4, max_candidates=2, max_expansions=500)
    assert tags.candidates
    result = parser.parse_candidates(words, tags.candidates[0].tags, beam_width=8, max_candidates=4, max_expansions=2000)
    assert result.expansions <= 2000
    for candidate in result.candidates:
        assert_tree(candidate, len(words))


def test_empirical_lexical_proposal_retains_evidence_without_filtering_weighted_tags():
    tagger = Tagger(tags=("A", "B"), known={"familiar": "A"})
    tagger.model.weights["w=familiar"] = {"B": 10.0}
    lexical = tagger.greedy_candidate(["familiar", "new"])
    assert lexical.tags == tuple(tagger.tag(["familiar", "new"])) == ("A", "B")
    assert lexical.lexical_positions == (0,)
    assert "training_lexicon" in lexical.provenance
    weighted = tagger.tag_candidates(["familiar", "new"], beam_width=4)
    assert any(candidate.tags[0] == "B" for candidate in weighted.candidates)
    assert tagger.lexical_candidate(["new"]) is None
    assert tagger.greedy_candidate(["new"]).tags == ("B",)


def test_unrepaired_greedy_lane_never_changes_illegal_decisions_to_produce_a_tree():
    parser = tiny_parser()
    # Historical greedy chooses shift and later repairs the missing head. The
    # audited lane rejects that sequence, even though another legal tree exists.
    parser.model.weights["b"] = {SHIFT: 10.0}
    audit = parser.greedy_search(["a"], ["X"])
    assert audit.candidates == ()
    assert audit.reason == "no_complete_candidates"
    assert parser.greedy_candidate(["a"], ["X"]) is None
    assert tiny_parser().parse_candidates(["a"], ["X"]).candidates


def test_valid_greedy_lane_matches_existing_model_decisions_without_repairs():
    parser = tiny_parser()
    parser.model.weights["b"] = {"right|root": 10.0, REDUCE: 9.0}
    candidate = parser.greedy_candidate(["a"], ["X"])
    assert candidate.heads == {1: 0}
    assert candidate.provenance == ("greedy-learned-unrepaired",)
    assert candidate.ranking == "greedy-local"
    assert_tree(candidate, 1)
    audit = parser.greedy_search(["a"], ["X"], max_steps=0)
    assert not audit.candidates and audit.reason == "budget_exhausted"


def test_margin_ranking_keeps_raw_scores_separate_and_is_explicit():
    parser = tiny_parser()
    parser.model.weights["stack=1"] = {"right|root": 1.0}
    parser.model.weights["s0w=a"] = {"left|dep": 10.0}
    raw = parser.parse_candidates(["a", "b"], ["X", "X"], beam_width=100, ranking="raw")
    margin = parser.parse_candidates(["a", "b"], ["X", "X"], beam_width=100, ranking="local_margin")
    assert raw.candidates[0].heads == {1: 2, 2: 0}
    assert margin.candidates[0].heads == {1: 0, 2: 1}
    assert all(candidate.ranking == "local_margin" and candidate.search_score <= 0 for candidate in margin.candidates)
    raw_scores = {tuple(sorted(candidate.heads.items())): candidate.score for candidate in raw.candidates}
    assert all(raw_scores[tuple(sorted(candidate.heads.items()))] == candidate.score for candidate in margin.candidates)
    with pytest.raises(ValueError, match="ranking"):
        parser.parse_candidates([], [], ranking="calibrated-probability")


def test_production_has_no_repaired_decoder_or_illegal_move_fallback():
    parser = Parser(moves=(SHIFT,))
    assert not hasattr(parser, "parse")
    dead_end = State(1, stack=[0, 1], next=2)
    assert dead_end.legal() == []
    assert parser._labels(dead_end) == []
    assert parser.greedy_candidate(["x"], ["X"]) is None
