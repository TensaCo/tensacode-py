"""Quotation preprocessing preserves literal source and exposes its convention."""
from pathlib import Path

import pytest

from tensorcode.agent.understand import LearnedReader, quotation_envelope, quoted, read
from tensorcode.language import ENGLISH


@pytest.mark.parametrize("raw,interior", [
    ('""', ''), ("''", ''), ('" "', ' '), ('"  delete file  "', '  delete file  '),
    ("'don't move'", "don't move"), ("‘I’m here’", "I’m here"),
    ('"hello?"!?', 'hello?'), ('  "hello".  ', 'hello'),
    ('"say \\"hello\\""', 'say \\"hello\\"'),
])
def test_true_envelope_preserves_exact_interior(raw, interior):
    assert quoted(raw) == interior
    envelope = quotation_envelope(raw)
    assert envelope.status == "whole"
    assert raw[slice(*envelope.content_span)] == interior
    assert len(envelope.delimiter_spans) == 2


@pytest.mark.parametrize("raw,status", [
    ('"delete" then "move"', 'multiple'), ("'delete' then 'move'", 'multiple'),
    ('"x""', 'unmatched'), ('"unterminated', 'unmatched'),
    ('"hello" then move', 'mixed'), ("'I 'm here'", 'ambiguous'),
    ("don't move", 'none'),
])
def test_multiple_malformed_or_uncertain_quotes_are_never_stripped(raw, status):
    assert quoted(raw) is None
    assert quotation_envelope(raw).status == status


@pytest.mark.parametrize("raw", ['""', '" "', "''"])
def test_grammar_keeps_empty_quotation_as_unresolved_source(raw):
    [sentence] = read(ENGLISH, raw)
    assert sentence.text == raw
    assert sentence.acts == ()
    assert len(sentence.alternatives) == 1
    metadata = sentence.alternatives[0].metadata
    assert metadata["quotation"]["applied"]
    assert metadata["quotation"]["convention"] == "authored:whole-quotation-as-mention"
    assert metadata["unresolved"] == "quoted source has no lexical content"


def test_grammar_preserves_raw_quotation_and_original_message_offsets():
    message = 'hello.\n  "  delete file  "'
    parsed = read(ENGLISH, message)
    quoted_sentence = parsed[-1]
    assert quoted_sentence.text == '"  delete file  "'
    assert all(act.kind == "mention" for alternative in quoted_sentence.alternatives for act in alternative.acts)
    metadata = quoted_sentence.alternatives[0].metadata["quotation"]
    assert message[slice(*metadata["source_span"])] == quoted_sentence.text
    assert message[slice(*metadata["content_span"])] == '  delete file  '


@pytest.fixture(scope="module")
def learned():
    path = Path.home() / ".cache/tensorcode/models/ud_ewt_parser.pickle"
    if not path.exists() or not path.with_name("ud_ewt_segmenter.json").exists():
        pytest.skip("requires cached learned parser and segmenter")
    return LearnedReader(tag_beam_width=1, tag_max_candidates=1,
                         parse_beam_width=2, parse_max_candidates=1, max_expansions=2000,
                         max_sentence_expansions=8000, max_alternatives=4,
                         semantic_max_candidates=2)


@pytest.mark.parametrize("raw", ['"delete" then "move"', '"x""'])
def test_actual_learned_reader_keeps_multiple_and_unmatched_quote_tokens(learned, raw):
    [sentence] = learned.read(raw)
    assert sentence.text == raw
    assert sentence.tokens != ('delete', '" then "', 'move')
    for alternative in sentence.alternatives:
        assert alternative.metadata["quotation"]["applied"] is False
        covered = set()
        for anchor in alternative.metadata["token_anchors"]:
            assert raw[slice(*anchor["char_span"])] == anchor["token"]
            covered.update(range(*anchor["char_span"]))
        assert covered == {i for i, char in enumerate(raw) if not char.isspace()}


@pytest.mark.parametrize("raw", ['""', '" "'])
def test_actual_learned_reader_retains_empty_quote_without_fabricated_meaning(learned, raw):
    [sentence] = learned.read(raw)
    assert sentence.text == raw and sentence.acts == ()
    assert sentence.alternatives
    for alternative in sentence.alternatives:
        metadata = alternative.metadata
        assert metadata["syntax_complete"] is False
        assert metadata["semantic_projection_complete"] is False
        assert metadata["quotation"]["applied"] is True
        assert raw[slice(*metadata["quotation"]["content_span"])] == raw[1:-1]


def test_actual_learned_reader_retains_repeated_quote_offsets_and_contraction(learned):
    message = '"don\'t move"\n"don\'t move"'
    parsed = learned.read(message)
    assert len(parsed) == 2
    starts = []
    for sentence in parsed:
        metadata = sentence.alternatives[0].metadata
        quote = metadata["quotation"]
        starts.append(quote["source_span"][0])
        assert quote["applied"]
        assert message[slice(*quote["content_span"])] == "don't move"
        for anchor in metadata["token_anchors"]:
            assert message[slice(*anchor["char_span"])] == anchor["token"]
        assert all(act.kind == "mention" for alternative in sentence.alternatives for act in alternative.acts)
    assert starts == [0, len('"don\'t move"\n')]
