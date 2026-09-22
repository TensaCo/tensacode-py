import json
import pytest
from tensorcode._internal.cognition.state import CognitiveState, Evidence, Hypothesis, Assessment, Goal, Plan, Observation


def base():
    return CognitiveState().add_evidence([Evidence('e', 'source words', 'document')]).add_hypotheses([Hypothesis('h', 'interpretation', model_provenance='weights-a')])


def test_revision_preserves_source_and_contradiction_and_invalidates_selection():
    s = base().assess([Assessment('e', 'h', {'entailed': .2, 'contradicted': .7, 'unknown': .1}, 'judge-a')]).select(['h'])
    changed = s.add_hypotheses([Hypothesis('h', 'revised', model_provenance='weights-b')])
    assert changed.evidence[0].text == 'source words'
    assert changed.assessments[0].scores['contradicted'] == .7
    assert changed.is_stale(changed.assessments[0])
    assert changed.selection_stale
    assert s.hypotheses[0].text == 'interpretation'
    assert not s.is_stale(s.assessments[0])
    assert not changed.observations


def test_failed_batch_is_atomic_and_states_independent():
    s = base()
    with pytest.raises(ValueError):
        s.add_evidence([Evidence('ok','new','src'), Evidence('e','overwrite','src')])
    with pytest.raises(ValueError):
        s.assess([Assessment('e','h',{'yes': .5},'judge'), Assessment('missing','h',{'yes': 1},'judge')])
    with pytest.raises(ValueError):
        s.assess([Assessment('e','h',{'yes': float('nan')},'judge')])
    assert len(s.evidence) == 1
    assert s.assessments == ()
    assert CognitiveState().evidence == ()


def test_bounded_state_and_strict_session_roundtrip(tmp_path):
    s = base().add_goals([Goal('g','investigate','user')]).add_plans([Plan('p', ('inspect',), ('new evidence',))]).observe([Observation('o','actual response','receipt-1')])
    p = tmp_path / 'session.json'
    s.save(p)
    loaded = CognitiveState.load(p)
    assert loaded.to_dict() == s.to_dict()
    with pytest.raises(ValueError):
        CognitiveState(max_records=1).add_evidence([Evidence('a','a','s'),Evidence('b','b','s')])
    data = json.loads(p.read_text()); data['unexpected'] = True; p.write_text(json.dumps(data))
    with pytest.raises(ValueError): CognitiveState.load(p)


def test_scores_cannot_be_mutated_after_capture():
    scores = {'unknown': 1.0}
    s = base().assess([Assessment('e','h',scores,'judge')])
    scores['unknown'] = 0
    assert s.assessments[0].scores['unknown'] == 1
    with pytest.raises(TypeError): s.assessments[0].scores['unknown'] = 0


def test_history_roundtrip_and_future_revision_rejected(tmp_path):
    s = base().assess([Assessment('e','h',{'supports': 1,'conflicts': -1},'judge')]).select(['h'])
    s = s.add_evidence([Evidence('e2','contrary source','second')])
    p = tmp_path / 'state.json'; s.save(p)
    loaded = CognitiveState.load(p)
    assert loaded.is_stale(loaded.assessments[0]) and loaded.selection_stale
    assert loaded.evidence[1].source_id == 'second'
    data = loaded.to_dict(); data['assessments'][0]['revision'] = 999
    with pytest.raises(ValueError): CognitiveState.from_dict(data)
    with pytest.raises(ValueError): loaded.select(['missing'])


def test_capacity_rejects_assessment_without_changing_state():
    s = CognitiveState(max_records=2).add_evidence([Evidence('e','E','source')]).add_hypotheses([Hypothesis('h','H',model_provenance='model')])
    with pytest.raises(ValueError): s.assess([Assessment('e','h',{'unknown':1},'judge')])
    assert s.assessments == ()


def test_reassessment_marks_previous_model_scores_stale_without_content_change():
    s = base().assess([Assessment('e','h',{'unknown':1},'model-v1')])
    updated = s.assess([Assessment('e','h',{'support':1},'model-v2')])
    assert updated.is_stale(updated.assessments[0])
    assert not updated.is_stale(updated.assessments[1])


def test_hypothesis_requires_stated_provenance():
    with pytest.raises(TypeError):
        Hypothesis('h', 'interpretation')
    with pytest.raises(ValueError, match='model_provenance'):
        Hypothesis('h', 'interpretation', model_provenance='')
    assert Hypothesis('h', 'text', 'supplied', model_provenance='caller-supplied').origin == 'supplied'
