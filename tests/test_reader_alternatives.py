"""Reader proposals survive the boundary without acquiring certainty or authority."""

from types import SimpleNamespace

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


def test_learned_reader_reports_single_proposal_and_preserves_quotation():
    # Exercise the boundary without requiring a downloaded treebank/model. This
    # parser returns one hypothesis; that fact must not masquerade as certainty.
    reader = LearnedReader.__new__(LearnedReader)
    reader.tagger = SimpleNamespace(tag=lambda words: ["X"] * len(words))
    reader.parser = SimpleNamespace(parse=lambda words, tags: ([0] * len(words), ["root"] * len(words)))
    reader.lemmatize = lambda word, tag, table: word
    reader.table = {}
    reader.conventions = ()
    frame = Frame("open", {"object": Entity("file", "file")}, {"mood": "imperative"})
    reader.reader = SimpleNamespace(read=lambda *args: (Request(frame),))
    for text, kind in (("open the file", "request"), ('"open the file"', "mention")):
        sentence, = reader.read(text)
        candidate, = sentence.alternatives
        assert candidate.provenance == "learned-reader-single"
        assert candidate.reading is None
        assert candidate.acts == sentence.acts
        assert [act.kind for act in candidate.acts] == [kind]
