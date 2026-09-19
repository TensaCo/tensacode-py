"""Measurement guards must fail when resumed work secretly decodes source again."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from eval.parsing.evaluate_continuation import candidate_record, forbid_decoding


def test_decoder_guard_catches_detached_model_and_restores_method():
    class Model:
        def segment(self, text):
            return text
    model = Model()
    copy = deepcopy(model)
    with forbid_decoding(SimpleNamespace(segmenter=model, tagger=None, parser=None)) as counts:
        with pytest.raises(AssertionError, match='continuation invoked decoder'):
            copy.segment('source')
        assert counts['Model.segment'] == 1
    assert copy.segment('source') == 'source'


def test_candidate_evidence_audit_rejects_wrong_source_anchor():
    metadata = dict(syntax_complete=True, sentence_span=(0, 4), tokens=('Bird',),
                    token_anchors=[dict(index=1, token='Fish', char_span=(0, 4))],
                    heads={1: 0}, labels={1: 'root'})
    candidate = SimpleNamespace(id='reading:test', provenance=('fixture',),
                                payload=SimpleNamespace(metadata=metadata, acts=()))
    result = candidate_record(candidate, 'Bird')
    assert result['anchor_errors'] == ['ValueError: token anchor does not match source']
