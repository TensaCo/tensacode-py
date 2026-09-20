"""Explicit labels and source-anchored neutral syntax authorize learned proposals."""
from dataclasses import replace
import pytest

from tensorcode.agent import Agent
from tensorcode.agent.understand import Act, SentenceAlternative
from tensorcode.agent.speech_act_learning import (
    retain_speech_act_example, fit_speech_act_model, admit_speech_act_model,
    get_speech_act_model, propose_speech_acts, speech_act_dependencies,
)
from tensorcode.agent.scene_grounding import grounding_dependencies
from tensorcode.agent.grounding import propose_grounding, MentionBinding
from tensorcode.learning.speech_act import SpeechActLabel
from tensorcode.outcomes import Unknown
from tensorcode.records import Ref
from test_speech_act_learning import example


def neutral(agent, noun, *, prefix=''):
    meaning = example(noun).meaning
    text = prefix + ' '.join(meaning.words)
    offset = len(prefix)
    anchors = []
    for index, word in enumerate(meaning.words, 1):
        anchors.append({'index': index, 'token': word, 'char_span': (offset, offset + len(word))})
        offset += len(word) + 1
    source = agent.interpretations.add_source(text, provider='authored syntax fixture')
    group = agent.interpretations.create_group(source.id)
    candidate = agent.interpretations.propose(group.id, SentenceAlternative(None,
        (Act('unresolved', meaning, None),), metadata={'tokens': meaning.words, 'token_anchors': tuple(anchors)}))
    return group.id, candidate.id


def fitted():
    agent = Agent()
    examples = []
    for noun in ('files', 'folders', 'windows'):
        gid, cid = neutral(agent, noun)
        record = retain_speech_act_example(agent, gid, cid, 0, SpeechActLabel('request'), basis=('explicit authored label',))
        assert not isinstance(record, Unknown), record
        examples.append(record)
    handle = fit_speech_act_model(agent, examples[:2], examples[2:])
    assert not isinstance(handle, Unknown), handle
    admitted = admit_speech_act_model(agent, handle, reason='explicit test model admission')
    assert not isinstance(admitted, Unknown), admitted
    return agent, admitted, examples


def test_source_anchored_teaching_and_explicit_model_admission():
    agent = Agent()
    gid, cid = neutral(agent, 'files', prefix='Prior source. ')
    record = retain_speech_act_example(agent, gid, cid, 0, SpeechActLabel('request'), basis=('explicit teacher',))
    assert not isinstance(record, Unknown), record
    assert record.example.text == 'open files'
    assert record.language.source.text == 'Prior source. open files'
    assert agent.interpretations.get(gid).selected_id is None
    agent, model, records = fitted()
    assert not isinstance(get_speech_act_model(agent, model), Unknown)
    unadmitted = replace(model, dependency=None)
    assert isinstance(get_speech_act_model(agent, unadmitted), Unknown)


def test_joint_publication_retains_neutral_parent_and_authenticates_grounding_descendants():
    agent, model, _ = fitted()
    gid, cid = neutral(agent, 'pictures')
    report = propose_speech_acts(agent, model, gid, cid)
    assert not isinstance(report, Unknown), report
    assert report.complete and len(report.candidate_ids) == 1
    group = agent.interpretations.get(gid)
    assert group.selected_id is None and group.candidates[0].id == cid
    child = next(c for c in group.candidates if c.id in report.candidate_ids)
    assert child.payload.acts[0].kind == 'request' and child.payload.acts[0].frame.predicate == 'open'
    assert grounding_dependencies(agent, gid, child.id) == (model.dependency,)
    bound = propose_grounding(agent.interpretations, gid, child.id,
        (MentionBinding(('acts', 0, 'frame', 'roles', 'object'), Ref('node:pictures'), (report.source_id,), 'explicit fixture binding'),))
    assert grounding_dependencies(agent, gid, bound.id) == (model.dependency,)
    agent.interpretations.unset(model.group_id, reason='withdraw speech model')
    assert isinstance(speech_act_dependencies(agent, gid, child.id), Unknown)
    assert isinstance(grounding_dependencies(agent, gid, bound.id), Unknown)


@pytest.mark.parametrize('tamper', ['source', 'candidate', 'report', 'model'])
def test_changed_speech_support_cannot_authorize_reading(tamper):
    agent, model, _ = fitted()
    gid, cid = neutral(agent, 'pictures')
    report = propose_speech_acts(agent, model, gid, cid)
    child_id, = report.candidate_ids
    if tamper == 'source':
        source = agent.interpretations.get(gid).source_id
        agent.interpretations._sources[source].metadata['changed'] = True
    elif tamper == 'candidate':
        # Internal mutation represents compromised retained evidence, not a normal editing API.
        agent._speech_act_children[(gid, child_id)][2].payload.metadata['changed'] = True
    elif tamper == 'report':
        agent.interpretations._sources[report.source_id].metadata['changed'] = True
    else:
        agent.interpretations.unset(model.group_id, reason='withdraw')
    assert isinstance(grounding_dependencies(agent, gid, child_id), Unknown)


def test_teaching_anchor_mismatch_and_overlapping_sources_refuse_fit():
    agent = Agent()
    gid, cid = neutral(agent, 'files')
    group = agent.interpretations.get(gid)
    agent.interpretations._sources[group.source_id] = replace(agent.interpretations._sources[group.source_id], text='other source')
    assert isinstance(retain_speech_act_example(agent, gid, cid, 0, SpeechActLabel('request'), basis=('explicit',)), Unknown)
    agent, _, records = fitted()
    assert isinstance(fit_speech_act_model(agent, records[:2], records[:1]), Unknown)


def test_refit_requires_readmission_and_invalidates_old_speech_children():
    agent, model, records = fitted()
    gid, cid = neutral(agent, 'pictures')
    report = propose_speech_acts(agent, model, gid, cid)
    newer = fit_speech_act_model(agent, records[:2], records[2:], group_id=model.group_id)
    assert not isinstance(newer, Unknown), newer
    assert newer.dependency is None and agent.interpretations.get(model.group_id).selected_id is None
    assert isinstance(grounding_dependencies(agent, gid, report.candidate_ids[0]), Unknown)


def test_joint_budget_refuses_partial_actionable_publication():
    agent, model, _ = fitted()
    gid, cid = neutral(agent, 'pictures')
    # An unresolved non-clausal sibling must not disappear while a request executes.
    original = agent.interpretations.get(gid).candidates[0].payload
    sibling = agent.interpretations.propose(gid, replace(original,
        acts=(*original.acts, Act('unresolved', 'unknown fragment', None))))
    report = propose_speech_acts(agent, model, gid, sibling.id)
    assert isinstance(report, Unknown)
    assert not any(c.payload.provenance == 'learned-speech-acts' for c in agent.interpretations.get(gid).candidates)


@pytest.mark.parametrize('mutation', ['candidate', 'model'])
def test_publication_callback_cannot_substitute_speech_act_or_withdraw_model(monkeypatch, mutation):
    agent, model, _ = fitted()
    gid, cid = neutral(agent, 'pictures')
    original = agent.interpretations.propose
    def publish(group_id, payload, **kwargs):
        if group_id == gid:
            if mutation == 'candidate':
                payload = replace(payload, acts=(replace(payload.acts[0], kind='tell'),))
            else:
                agent.interpretations.unset(model.group_id, reason='withdraw during publication')
        return original(group_id, payload, **kwargs)
    monkeypatch.setattr(agent.interpretations, 'propose', publish)
    result = propose_speech_acts(agent, model, gid, cid)
    assert isinstance(result, Unknown), result
    children = [c for c in agent.interpretations.get(gid).candidates if c.id != cid]
    assert children and all(c.rejected for c in children)
    assert all(isinstance(grounding_dependencies(agent, gid, c.id), Unknown) for c in children)


def test_unmatched_provisional_clause_prevents_partial_request_publication():
    agent, model, _ = fitted()
    gid, cid = neutral(agent, 'pictures')
    original = agent.interpretations.get(gid).candidates[0].payload
    meaning = original.acts[0].meaning
    sibling = replace(meaning, frame=replace(meaning.frame, predicate='unsupported'), frame_index=1)
    combined = agent.interpretations.propose(gid, replace(original,
        acts=(*original.acts, Act('unresolved', sibling, None))))
    report = propose_speech_acts(agent, model, gid, combined.id)
    assert not isinstance(report, Unknown), report
    assert not report.candidate_ids
    assert not any(c.payload.provenance == 'learned-speech-acts' for c in agent.interpretations.get(gid).candidates)
