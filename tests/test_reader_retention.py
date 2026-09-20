"""Authored proposal fixtures test compute allocation, never linguistic accuracy."""
from types import SimpleNamespace

from tensorcode.language import Frame
from tensorcode.language.deps_semantics import SemanticReadCandidate, SemanticReadCandidates
from test_learned_reader_alternatives import fixture_reader
from dependency_meaning_fixtures import neutral_fixture


class CountingSemantics:
    def __init__(self, variants=3):
        self.variants = variants
        self.started = []
        self.advances = []

    def start_candidates(self, words, tags, lemmas, heads, labels):
        key = (tuple(words), tuple(tags), tuple(heads.items()))
        self.started.append(key)
        owner = self

        class Cursor:
            explored = 0

            def advance(self, *, max_expansions, max_candidates):
                owner.advances.append((key, max_expansions, max_candidates))
                candidates = ()
                if max_expansions and max_candidates and self.explored < owner.variants:
                    candidates = (SemanticReadCandidate((neutral_fixture(Frame('authored', {}, {'variant': self.explored}), words, tags, lemmas, heads, labels),), ()),)
                    self.explored += 1
                pending = owner.variants - self.explored
                return SemanticReadCandidates(candidates, bool(pending), self.explored, pending,
                                              'authored_pending' if pending else None)
        return Cursor()


def reader_with_counter(limit):
    reader = fixture_reader(limit)
    reader.reader = CountingSemantics()
    return reader


def test_global_cap_precedes_semantics_and_preserves_distinct_trees():
    reader = reader_with_counter(3)
    sentence, = reader.read('birds fly.')
    assert len(sentence.alternatives) == len(reader.reader.started) == 3
    assert len(set(reader.reader.started)) == 3
    assert len(reader.reader.advances) == 3
    metadata = sentence.alternatives[0].metadata
    assert metadata['retention_stats'] == {
        'syntax_generated': 4, 'syntax_retained': 3, 'syntax_discarded': 1,
        'semantic_explored': 3, 'semantic_emitted': 3, 'semantic_pending': 6,
        'semantic_unexpanded_families': 0, 'semantic_discarded': 0}
    deferred, = metadata['pending_syntax_families']
    assert deferred['heads'] and deferred['tokens'] and deferred['token_anchors']
    assert deferred['semantic_projection_complete'] is False
    assert 'not run' in deferred['unresolved']
    assert all(a.metadata['semantic_candidate_index'] == 0 for a in sentence.alternatives)


def test_second_semantic_variants_follow_every_distinct_syntax_family():
    reader = reader_with_counter(6)
    sentence, = reader.read('birds fly.')
    assert len(reader.reader.started) == 4
    assert len(reader.reader.advances) == 6
    indices = [a.metadata['semantic_candidate_index'] for a in sentence.alternatives]
    assert indices == [0, 0, 0, 0, 1, 1]
    assert sentence.alternatives[0].metadata['retention_stats']['semantic_discarded'] == 0
    assert sentence.alternatives[0].metadata['retention_stats']['semantic_pending'] == 6
    assert all(a.metadata['search_truncated'] for a in sentence.alternatives)


def test_shared_semantic_budget_retains_unexpanded_families_without_acts():
    reader = reader_with_counter(4)
    reader.max_sentence_semantic_expansions = 1
    sentence, = reader.read('birds fly.')
    assert len(sentence.alternatives) == 4
    metadata = sentence.alternatives[0].metadata
    assert metadata['sentence_semantic_expansions'] == 1
    assert metadata['retention_stats']['semantic_unexpanded_families'] == 3
    unresolved = [a for a in sentence.alternatives if a.metadata['semantic_candidate_index'] is None]
    assert len(unresolved) == 3
    assert all(not a.acts and a.skipped == a.metadata['tokens'] for a in unresolved)
    assert all(a.metadata['semantic_frontier']['pending'] for a in unresolved)
    assert all(a.metadata['semantic_projection_complete'] is False for a in unresolved)


def test_duplicate_decoder_paths_share_one_projection_and_keep_provenance():
    reader = reader_with_counter(4)
    original = reader.parser.parse_candidates

    def duplicate(*args, **kwargs):
        search = original(*args, **kwargs)
        search.candidates = tuple(candidate for candidate in search.candidates for _ in range(2))
        return search

    reader.parser.parse_candidates = duplicate
    sentence, = reader.read('birds fly.')
    assert len(reader.reader.started) == 4
    assert all(len(a.metadata['decoder_proposals']) == 2 for a in sentence.alternatives)
    assert sentence.alternatives[0].metadata['retention_stats']['syntax_generated'] == 4


def test_full_cap_can_resume_reserved_placeholder_using_unspent_budget():
    reader = reader_with_counter(2)
    reader.max_sentence_semantic_expansions = 6
    owner = reader.reader
    started = []

    def start(*args):
        needed = 5 if not started else 1
        started.append(needed)

        class Cursor:
            explored = 0

            def advance(self, *, max_expansions, max_candidates):
                old = self.explored
                self.explored += min(max_expansions, needed - self.explored)
                complete = self.explored == needed
                candidates = ((SemanticReadCandidate((neutral_fixture(Frame('authored', {}, {}), *args),), ()),)
                              if complete and old < needed else ())
                return SemanticReadCandidates(candidates, not complete, self.explored,
                                              int(not complete), None)
        return Cursor()

    owner.start_candidates = start
    sentence, = reader.read('birds fly.')
    assert started == [5, 1]
    assert len(sentence.alternatives) == 2
    assert all(a.acts and a.metadata['semantic_candidate_index'] == 0 for a in sentence.alternatives)
    assert sentence.alternatives[0].metadata['retention_stats']['semantic_explored'] == 6
    assert sentence.alternatives[0].metadata['retention_stats']['semantic_pending'] == 0


def test_global_cap_interleaves_segmentation_families_before_projection():
    reader = reader_with_counter(3)
    reader.segmenter = SimpleNamespace(segment=lambda text, **bounds: SimpleNamespace(
        candidates=(SimpleNamespace(spans=((0, 5), (6, 9), (9, 10)), score=0.0, provenance=('fixture',)),
                    SimpleNamespace(spans=((0, 2), (2, 5), (6, 9), (9, 10)), score=0.0, provenance=('fixture',))),
        expansions=2, complete=True, truncated=False, reason=None))
    sentence, = reader.read('birds fly.')
    assert len(reader.reader.started) == 3
    assert [a.metadata['segmentation_index'] for a in sentence.alternatives] == [0, 1, 0]
    stats = sentence.alternatives[0].metadata['retention_stats']
    assert stats['syntax_generated'] == 8 and stats['syntax_retained'] == 3
    assert stats['syntax_discarded'] == 5 and stats['semantic_explored'] == 3
    assert len(sentence.alternatives[0].metadata['pending_syntax_families']) == 5
