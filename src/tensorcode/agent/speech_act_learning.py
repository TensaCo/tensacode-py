"""Source-anchored explicit speech-act teaching and admitted learned alternatives.

Syntax and provisional frame projection never authorize request/question/assertion
routing. Teachers supply labels; heldout correspondences propose joint readings;
callers still explicitly select a reading and its model dependency.
"""
from copy import deepcopy
from dataclasses import dataclass, replace
from itertools import product
from uuid import uuid4

from ..learning.experience import _same
from ..language.deps_semantics import ProvisionalMeaning
from ..outcomes import Unknown
from .understand import Act, SentenceAlternative
from .scene_grounding import _capture, _candidate, _validate_snapshot, _final_comparisons
from .task_dependencies import capture_dependency, validate_dependencies, _expected_basis


@dataclass(frozen=True)
class RetainedSpeechActExample:
    example: object
    language: object
    act_index: int
    evidence_source_id: str


@dataclass(frozen=True)
class SpeechActModelHandle:
    group_id: str
    candidate_id: str
    model_id: str
    evidence_source_id: str
    dependency: object = None


@dataclass(frozen=True)
class SpeechActReport:
    candidate_ids: tuple[str, ...]
    source_id: str
    complete: bool
    unresolved: tuple


@dataclass(frozen=True)
class _Version:
    snapshot: dict
    evidence_source_id: str


def _registry(agent, name):
    if not hasattr(agent, name):
        setattr(agent, name, {})
    return getattr(agent, name)


def _meaning(snapshot, index):
    alternative = _candidate(snapshot).payload
    if type(index) is not int or not 0 <= index < len(alternative.acts):
        raise ValueError('explicit provisional act index required')
    act = alternative.acts[index]
    if act.kind != 'unresolved' or act.frame is not None or type(act.meaning) is not ProvisionalMeaning:
        raise ValueError('teaching requires a neutral provisional meaning')
    meaning = act.meaning
    anchors = alternative.metadata.get('token_anchors')
    if tuple(alternative.metadata.get('tokens', ())) != meaning.words or type(anchors) not in (tuple, list) or len(anchors) != len(meaning.words):
        raise ValueError('syntax tokens require complete source anchors')
    previous = -1
    for ordinal, (word, anchor) in enumerate(zip(meaning.words, anchors), 1):
        if type(anchor) is not dict or anchor.get('index') != ordinal or anchor.get('token') != word:
            raise ValueError('token anchor identity mismatch')
        span = anchor.get('char_span')
        if (type(span) not in (tuple, list) or len(span) != 2 or any(type(x) is not int for x in span)
                or not 0 <= span[0] < span[1] <= len(snapshot.source.text) or span[0] < previous
                or snapshot.source.text[span[0]:span[1]] != word):
            raise ValueError('provisional tokens do not match retained source')
        previous = span[1]
    return deepcopy(meaning)


def retain_speech_act_example(agent, group_id, candidate_id, act_index, label, *, basis):
    from ..learning.speech_act import SpeechActExample, SpeechActLabel
    try:
        if type(basis) is not tuple or not basis or any(type(x) is not str or not x.strip() for x in basis):
            raise ValueError('speech-act teaching requires an explicit basis')
        if type(label) is not SpeechActLabel:
            raise ValueError('an explicit SpeechActLabel is required')
        label.__post_init__()
        language = _capture(agent.interpretations, group_id, candidate_id, SentenceAlternative)
        meaning = _meaning(language, act_index)
        anchors = _candidate(language).payload.metadata['token_anchors']
        teaching_text = language.source.text[anchors[0]['char_span'][0]:anchors[-1]['char_span'][1]]
        example = SpeechActExample('speech-example:' + uuid4().hex, language.source.id,
                                   teaching_text, meaning, deepcopy(label), basis)
        evidence = agent.interpretations.add_source('Explicit provisional syntax to speech-act teaching',
            modality='teaching', provider='explicit-speech-act-teaching', payload=deepcopy(example),
            metadata={'language': language, 'act_index': act_index})
        result = RetainedSpeechActExample(example, language, act_index, evidence.id)
        cached, returned = deepcopy((result, evidence)), deepcopy(result)
        _validate_snapshot(agent.interpretations, language, SentenceAlternative)
        _final_comparisons(agent.interpretations, (language,))
        _registry(agent, '_speech_act_examples')[evidence.id] = cached
        return returned
    except Exception as error:
        return Unknown('speech_act_teaching_unavailable', f'{type(error).__name__}: {error}')


def _validate_examples(agent, records):
    for record in records:
        cached = _registry(agent, '_speech_act_examples').get(record.evidence_source_id)
        if cached is None or not _same(cached[0], record) or not _same(agent.interpretations.get_source(record.evidence_source_id), cached[1]):
            raise ValueError('speech-act teaching content changed')
        _validate_snapshot(agent.interpretations, record.language, SentenceAlternative)
        if not _same(_meaning(record.language, record.act_index), record.example.meaning):
            raise ValueError('speech-act teaching meaning changed')
    _final_comparisons(agent.interpretations, tuple(record.language for record in records))


def _model_state(agent, group_id):
    saved = _registry(agent, '_speech_act_models').get(group_id)
    if saved is None:
        raise ValueError('unrecognized speech-act model group')
    source, provenance, versions = saved
    workspace = agent.interpretations
    comparison = workspace.comparison_basis(group_id)
    group = workspace.get(group_id)
    if group.source_id != source.id or group.provenance != provenance or not _same(workspace.get_source(source.id), source):
        raise ValueError('speech-act model source changed')
    if tuple(c.id for c in group.candidates) != tuple(v[0] for v in versions):
        raise ValueError('speech-act model versions changed')
    for candidate, (_, version, evidence, model) in zip(group.candidates, versions):
        if (not _same(candidate.payload, version) or not _same(workspace.get_source(evidence.id), evidence)
                or not _same(vars(model), version.snapshot)):
            raise ValueError('speech-act model content changed')
    if workspace.comparison_basis(group_id) != comparison:
        raise ValueError('speech-act model comparison changed during validation')
    return group, comparison, versions


def fit_speech_act_model(agent, training_records, validation_records, *, group_id=None, max_pairs=256):
    from ..learning.speech_act import fit_speech_acts
    published = None
    try:
        training, validation = tuple(training_records), tuple(validation_records)
        records = (*training, *validation)
        _validate_examples(agent, records)
        if {r.example.source_id for r in training} & {r.example.source_id for r in validation}:
            raise ValueError('speech-act training and heldout sources overlap')
        previous = _model_state(agent, group_id) if group_id is not None else None
        model = fit_speech_acts(tuple(r.example for r in training), tuple(r.example for r in validation), max_pairs=max_pairs)
        evidence = agent.interpretations.add_source('Historical fitted speech-act correspondences',
            modality='speech-act-model', provider='learned-speech-act-correspondence', payload=deepcopy(vars(model)),
            metadata={'training': training, 'validation': validation, 'max_pairs': max_pairs})
        version = _Version(deepcopy(vars(model)), evidence.id)
        cached = deepcopy((version, evidence, model))
        _validate_examples(agent, records)
        workspace = agent.interpretations
        if previous is None:
            group = workspace.create_group(evidence.id, provenance=('speech-act model versions',))
            saved = (deepcopy(evidence), group.provenance, ())
        else:
            group, comparison, _ = _model_state(agent, group_id)
            if comparison != previous[1]:
                raise ValueError('speech-act model admission changed during refit')
            saved = _registry(agent, '_speech_act_models')[group.id]
        candidate = workspace.propose(group.id, version, provenance=('retained-speech-act-model',))
        _registry(agent, '_speech_act_models')[group.id] = (*saved[:2], (*saved[2], (candidate.id, *cached)))
        published = (group.id, candidate.id)
        workspace.unset(group.id, reason='new speech-act fit requires explicit admission')
        _validate_examples(agent, records)
        _model_state(agent, group.id)
        return SpeechActModelHandle(group.id, candidate.id, model.id, evidence.id)
    except Exception as error:
        if published:
            agent.interpretations.reject(*published, reason='speech-act fit evidence changed')
        return Unknown('speech_act_fit_unavailable', f'{type(error).__name__}: {error}')


def admit_speech_act_model(agent, handle, *, reason):
    try:
        if type(handle) is not SpeechActModelHandle or type(reason) is not str or not reason.strip():
            raise ValueError('speech-act admission requires authentic handle and reason')
        group, comparison, versions = _model_state(agent, handle.group_id)
        rows = [row for row in versions if row[0] == handle.candidate_id]
        if len(rows) != 1 or rows[0][3].id != handle.model_id or rows[0][2].id != handle.evidence_source_id or not rows[0][3].complete:
            raise ValueError('speech-act model incomplete or handle mismatched')
        candidate = next(c for c in group.candidates if c.id == handle.candidate_id)
        if candidate.rejected:
            raise ValueError('speech-act model version rejected')
        expected = (comparison[0], group.revision + 1, candidate.id, comparison[3], False, comparison[5], comparison[6])
        agent.interpretations.select(group.id, candidate.id, reason=reason, evidence_ids=(handle.evidence_source_id,))
        dependency = capture_dependency(agent.interpretations, group.id,
            basis=('explicit speech-act model admission', reason), evidence_ids=(handle.evidence_source_id,))
        result = replace(handle, dependency=dependency)
        _model_state(agent, group.id)
        if agent.interpretations.comparison_basis(group.id) != expected or validate_dependencies(agent.interpretations, (dependency,)) is not True:
            raise ValueError('speech-act admission changed')
        _registry(agent, '_speech_act_admissions')[(group.id, candidate.id)] = deepcopy(result)
        return result
    except Exception as error:
        return Unknown('speech_act_admission_unavailable', f'{type(error).__name__}: {error}')


def get_speech_act_model(agent, handle):
    try:
        if type(handle) is not SpeechActModelHandle or handle.dependency is None:
            raise ValueError('speech-act model not admitted')
        cached = _registry(agent, '_speech_act_admissions').get((handle.group_id, handle.candidate_id))
        if not _same(handle, cached):
            raise ValueError('speech-act model admission changed')
        group, comparison, versions = _model_state(agent, handle.group_id)
        if group.selected_id != handle.candidate_id or validate_dependencies(agent.interpretations, (handle.dependency,)) is not True:
            raise ValueError('speech-act model dependency withdrawn')
        model = deepcopy(next(v[3] for v in versions if v[0] == handle.candidate_id))
        _model_state(agent, group.id)
        if agent.interpretations.comparison_basis(group.id) != comparison:
            raise ValueError('speech-act model changed during retrieval')
        return model
    except Exception as error:
        return Unknown('speech_act_model_unavailable', f'{type(error).__name__}: {error}')


def speech_act_dependencies(agent, group_id, candidate_id):
    """Authenticate only registered learned children; graph lineage calls recursively."""
    retained = _registry(agent, '_speech_act_children').get((group_id, candidate_id))
    if retained is None:
        return ()
    try:
        source, parent, child, model_handle, report, provenance = retained
        workspace = agent.interpretations
        comparison = workspace.comparison_basis(group_id)
        group = workspace.get(group_id)
        if (group.source_id != source.id or group.provenance != provenance
                or not _same(workspace.get_source(source.id), source)
                or not _same(next(c for c in group.candidates if c.id == parent.id), parent)
                or not _same(next(c for c in group.candidates if c.id == child.id), child)
                or not _same(workspace.get_source(report.id), report)):
            raise ValueError('learned speech-act source, parent or child changed')
        if isinstance(get_speech_act_model(agent, model_handle), Unknown):
            raise ValueError('learned speech-act model withdrawn or changed')
        if workspace.comparison_basis(group_id) != comparison:
            raise ValueError('speech-act comparison changed during validation')
        return (model_handle.dependency,)
    except Exception as error:
        return Unknown('speech_act_dependency_changed', f'{type(error).__name__}: {error}')


def propose_speech_acts(agent, admitted_handle, group_id, candidate_id, *, max_alternatives=256):
    published = []
    try:
        if type(max_alternatives) is not int or max_alternatives < 1:
            raise ValueError('speech-act alternatives require a positive bound')
        model = get_speech_act_model(agent, admitted_handle)
        if isinstance(model, Unknown):
            raise ValueError(model.detail)
        language = _capture(agent.interpretations, group_id, candidate_id, SentenceAlternative)
        parent = _candidate(language)
        meanings = tuple(_meaning(language, index) for index in range(len(parent.payload.acts)))
        if not meanings:
            raise ValueError('no provisional meanings in this candidate')
        batches = tuple(model.propose(meaning) for meaning in meanings)
        unresolved = tuple(reason for batch in batches for reason in batch.unresolved)
        complete = all(batch.complete and batch.proposals for batch in batches)
        count = 1
        for batch in batches:
            count *= len(batch.proposals)
        if count > max_alternatives:
            complete = False
            unresolved = (*unresolved, 'joint_speech_act_budget')
        evidence = agent.interpretations.add_source('Learned joint speech-act alternatives',
            modality='speech-act-proposals', provider='learned-speech-act-correspondence', payload=deepcopy(batches),
            metadata={'parent': language, 'model': admitted_handle, 'joint_count': count,
                      'complete': complete, 'unresolved': unresolved})
        expected_evidence = deepcopy(evidence)
        _validate_snapshot(agent.interpretations, language, SentenceAlternative)
        if not complete or unresolved:
            return SpeechActReport((), evidence.id, bool(complete), unresolved)
        for combination in product(*(batch.proposals for batch in batches)):
            acts = tuple(Act({'statement': 'tell', 'request': 'request', 'question': 'question', 'unresolved': 'unresolved'}[p.label.kind],
                             deepcopy(p.meaning), deepcopy(getattr(p.meaning, 'frame', p.frame)) if p.label.kind != 'unresolved' else None) for p in combination)
            actionable = all(a.kind != 'unresolved' for a in acts)
            alternative = replace(parent.payload, acts=acts, provenance='learned-speech-acts',
                metadata={**deepcopy(parent.payload.metadata), 'speech_act_complete': actionable,
                          'speech_act_status': 'proposed' if actionable else 'unresolved',
                          'speech_act_evidence_source_id': evidence.id,
                          'speech_act_proposals': deepcopy(combination)})
            expected_alternative = deepcopy(alternative)
            provenance = ('learned-speech-acts', f'parent:{candidate_id}', f'evidence:{evidence.id}')
            candidate = agent.interpretations.propose(group_id, alternative, provenance=provenance)
            expected_candidate = replace(candidate, group_id=group_id, payload=expected_alternative, provenance=provenance)
            published.append(candidate.id)
            _registry(agent, '_speech_act_children')[(group_id, candidate.id)] = deepcopy((
                language.source, parent, expected_candidate, admitted_handle, expected_evidence, language.group.provenance))
        current = agent.interpretations.get(group_id)
        expected_ids = (*language.comparison[3], *published)
        if (tuple(c.id for c in current.candidates) != expected_ids
                or not _same(current.candidates[:len(language.group.candidates)], language.group.candidates)
                or not _same(agent.interpretations.get_source(evidence.id), expected_evidence)
                or not _same(agent.interpretations.get_source(language.source.id), language.source)
                or isinstance(get_speech_act_model(agent, admitted_handle), Unknown)):
            raise ValueError('speech-act support changed during publication')
        expected = (*language.comparison[:3], expected_ids, *language.comparison[4:])
        if agent.interpretations.comparison_basis(group_id) != expected:
            raise ValueError('speech-act frontier changed during publication')
        for ident in published:
            result = speech_act_dependencies(agent, group_id, ident)
            if isinstance(result, Unknown):
                raise ValueError(result.detail)
        return SpeechActReport(tuple(published), evidence.id, True, ())
    except Exception as error:
        for ident in published:
            agent.interpretations.reject(group_id, ident, reason='speech-act support changed during publication')
        return Unknown('speech_act_projection_unavailable', f'{type(error).__name__}: {error}')


def project_speech_act_group(agent, handle, group_id):
    """Project each newly retained neutral family once per admitted model version."""
    projected = _registry(agent, '_speech_act_projected')
    results = []
    for candidate in agent.interpretations.get(group_id).candidates:
        key = (group_id, candidate.id, handle.group_id, handle.candidate_id, handle.dependency)
        if key in projected or type(candidate.payload) is not SentenceAlternative:
            continue
        if not candidate.payload.acts or not all(a.kind == 'unresolved' and type(a.meaning) is ProvisionalMeaning for a in candidate.payload.acts):
            continue
        result = propose_speech_acts(agent, handle, group_id, candidate.id)
        projected[key] = result
        results.append(result)
    return tuple(results)
