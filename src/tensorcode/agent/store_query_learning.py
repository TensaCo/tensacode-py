"""Source-anchored learned store queries with explicit selection and support validation."""
from copy import deepcopy
from dataclasses import dataclass, replace
from uuid import uuid4

from ..learning.experience import _same
from ..language import Question
from ..outcomes import Unknown
from .understand import SentenceAlternative
from .scene_grounding import _capture, _candidate, _validate_snapshot, _final_comparisons, grounding_dependencies
from .task_dependencies import capture_dependency, validate_dependencies, _expected_basis


@dataclass(frozen=True)
class RetainedStoreQueryExample:
    example: object
    language: object
    act_index: int
    evidence_source_id: str


@dataclass(frozen=True)
class StoreQueryModelHandle:
    group_id: str
    candidate_id: str
    model_id: str
    evidence_source_id: str
    dependency: object = None


@dataclass(frozen=True)
class StoreQueryReport:
    group_id: str
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


def _question(snapshot, index):
    candidate = _candidate(snapshot)
    if type(index) is not int or not 0 <= index < len(candidate.payload.acts):
        raise ValueError('an explicit retained question act index is required')
    act = candidate.payload.acts[index]
    if act.kind != 'question' or type(act.meaning) is not Question or not _same(act.meaning.frame, act.frame):
        raise ValueError('teaching requires a retained full question and matching frame')
    return deepcopy(act.meaning)


def retain_store_query_example(agent, group_id, candidate_id, act_index, plan, *, basis):
    from ..learning.store_query import StoreQueryExample, StoreQueryPlan
    try:
        if type(plan) is not StoreQueryPlan:
            raise ValueError('an explicit StoreQueryPlan is required')
        if type(basis) is not tuple or not basis or any(type(x) is not str or not x.strip() for x in basis):
            raise ValueError('store_query teaching requires an explicit basis')
        language = _capture(agent.interpretations, group_id, candidate_id, SentenceAlternative)
        question = _question(language, act_index)
        example = StoreQueryExample('store_query-example:' + uuid4().hex, language.source.id,
            language.source.text, question, deepcopy(plan), basis)
        support = grounding_dependencies(agent, group_id, candidate_id)
        if isinstance(support, Unknown):
            raise ValueError(support.detail)
        evidence = agent.interpretations.add_source('Explicit full-question store_query teaching',
            modality='teaching', provider='explicit-store_query-teaching', payload=deepcopy(example),
            metadata={'language': language, 'act_index': act_index, 'dependencies': support})
        result = RetainedStoreQueryExample(example, language, act_index, evidence.id)
        cached, returned = deepcopy((result, evidence)), deepcopy(result)
        _validate_snapshot(agent.interpretations, language, SentenceAlternative)
        if validate_dependencies(agent.interpretations, support) is not True:
            raise ValueError('store_query teaching support changed')
        _final_comparisons(agent.interpretations, (language,))
        _registry(agent, '_store_query_examples')[evidence.id] = cached
        return returned
    except Exception as error:
        return Unknown('store_query_teaching_unavailable', f'{type(error).__name__}: {error}')


def _validate_examples(agent, records):
    for record in records:
        cached = _registry(agent, '_store_query_examples').get(record.evidence_source_id)
        if cached is None or not _same(cached[0], record) or not _same(agent.interpretations.get_source(record.evidence_source_id), cached[1]):
            raise ValueError('store_query teaching content changed')
        _validate_snapshot(agent.interpretations, record.language, SentenceAlternative)
        if not _same(_question(record.language, record.act_index), record.example.question):
            raise ValueError('store_query teaching question changed')
        support = grounding_dependencies(agent, record.language.group_id, record.language.candidate_id)
        if isinstance(support, Unknown) or support != cached[1].metadata['dependencies']:
            raise ValueError('store_query teaching support changed')
        if validate_dependencies(agent.interpretations, support) is not True:
            raise ValueError('store_query teaching dependency changed')
    _final_comparisons(agent.interpretations, tuple(record.language for record in records))


def _model_state(agent, group_id):
    saved = _registry(agent, '_store_query_models').get(group_id)
    if saved is None:
        raise ValueError('unrecognized store_query model group')
    source, provenance, versions = saved
    workspace = agent.interpretations
    comparison = workspace.comparison_basis(group_id)
    group = workspace.get(group_id)
    if group.source_id != source.id or group.provenance != provenance or not _same(workspace.get_source(source.id), source):
        raise ValueError('store_query model source changed')
    if tuple(c.id for c in group.candidates) != tuple(v[0] for v in versions):
        raise ValueError('store_query model versions changed')
    for candidate, (_, version, evidence, model) in zip(group.candidates, versions):
        if (not _same(candidate.payload, version) or not _same(workspace.get_source(evidence.id), evidence)
                or not _same(vars(model), version.snapshot)):
            raise ValueError('store_query model content changed')
    if workspace.comparison_basis(group_id) != comparison:
        raise ValueError('store_query model comparison changed during validation')
    return group, comparison, versions


def fit_store_query_model(agent, training_records, validation_records, *, group_id=None, max_pairs=256):
    from ..learning.store_query import fit_store_queries
    published = None
    try:
        training, validation = tuple(training_records), tuple(validation_records)
        records = (*training, *validation)
        _validate_examples(agent, records)
        if {r.example.source_id for r in training} & {r.example.source_id for r in validation}:
            raise ValueError('store_query training and heldout sources overlap')
        previous = _model_state(agent, group_id) if group_id is not None else None
        model = fit_store_queries(tuple(r.example for r in training), tuple(r.example for r in validation), max_pairs=max_pairs)
        evidence = agent.interpretations.add_source('Historical fitted store_query correspondences',
            modality='store_query-model', provider='learned-store_query-correspondence', payload=deepcopy(vars(model)),
            metadata={'training': training, 'validation': validation, 'max_pairs': max_pairs})
        version = _Version(deepcopy(vars(model)), evidence.id)
        cached = deepcopy((version, evidence, model))
        _validate_examples(agent, records)
        workspace = agent.interpretations
        if previous is None:
            group = workspace.create_group(evidence.id, provenance=('store_query model versions',))
            saved = (deepcopy(evidence), group.provenance, ())
        else:
            group, comparison, _ = _model_state(agent, group_id)
            if comparison != previous[1]:
                raise ValueError('store_query model admission changed during refit')
            saved = _registry(agent, '_store_query_models')[group.id]
        candidate = workspace.propose(group.id, version, provenance=('retained-store_query-model',))
        _registry(agent, '_store_query_models')[group.id] = (*saved[:2], (*saved[2], (candidate.id, *cached)))
        published = (group.id, candidate.id)
        workspace.unset(group.id, reason='new store_query fit requires explicit admission')
        _validate_examples(agent, records)
        _model_state(agent, group.id)
        return StoreQueryModelHandle(group.id, candidate.id, model.id, evidence.id)
    except Exception as error:
        if published:
            agent.interpretations.reject(*published, reason='store_query fit evidence changed')
        return Unknown('store_query_fit_unavailable', f'{type(error).__name__}: {error}')


def admit_store_query_model(agent, handle, *, reason):
    try:
        if type(handle) is not StoreQueryModelHandle or type(reason) is not str or not reason.strip():
            raise ValueError('store_query admission requires authentic handle and reason')
        group, comparison, versions = _model_state(agent, handle.group_id)
        rows = [row for row in versions if row[0] == handle.candidate_id]
        if len(rows) != 1 or rows[0][3].id != handle.model_id or rows[0][2].id != handle.evidence_source_id or not rows[0][3].complete:
            raise ValueError('store_query model incomplete or handle mismatched')
        candidate = next(c for c in group.candidates if c.id == handle.candidate_id)
        if candidate.rejected:
            raise ValueError('store_query model version rejected')
        expected = (comparison[0], group.revision + 1, candidate.id, comparison[3], False, comparison[5], comparison[6])
        agent.interpretations.select(group.id, candidate.id, reason=reason, evidence_ids=(handle.evidence_source_id,))
        dependency = capture_dependency(agent.interpretations, group.id,
            basis=('explicit store_query model admission', reason), evidence_ids=(handle.evidence_source_id,))
        result = replace(handle, dependency=dependency)
        _model_state(agent, group.id)
        if agent.interpretations.comparison_basis(group.id) != expected or validate_dependencies(agent.interpretations, (dependency,)) is not True:
            raise ValueError('store_query admission changed')
        _registry(agent, '_store_query_admissions')[(group.id, candidate.id)] = deepcopy(result)
        return result
    except Exception as error:
        return Unknown('store_query_admission_unavailable', f'{type(error).__name__}: {error}')


def get_store_query_model(agent, handle):
    try:
        if type(handle) is not StoreQueryModelHandle or handle.dependency is None:
            raise ValueError('store_query model not admitted')
        cached = _registry(agent, '_store_query_admissions').get((handle.group_id, handle.candidate_id))
        if not _same(handle, cached):
            raise ValueError('store_query model admission changed')
        group, comparison, versions = _model_state(agent, handle.group_id)
        if group.selected_id != handle.candidate_id or validate_dependencies(agent.interpretations, (handle.dependency,)) is not True:
            raise ValueError('store_query model dependency withdrawn')
        model = deepcopy(next(v[3] for v in versions if v[0] == handle.candidate_id))
        _model_state(agent, group.id)
        if agent.interpretations.comparison_basis(group.id) != comparison:
            raise ValueError('store_query model changed during retrieval')
        return model
    except Exception as error:
        return Unknown('store_query_model_unavailable', f'{type(error).__name__}: {error}')




@dataclass(frozen=True)
class StoreQuerySelection:
    group_id: str
    candidate_id: str
    plan: object
    dependencies: tuple


def propose_store_queries(agent, admitted_handle, question, *, parent_dependency):
    """Retain all learned plans; never select a plan or infer a parameter mapping."""
    try:
        model = get_store_query_model(agent, admitted_handle)
        if isinstance(model, Unknown):
            raise ValueError(model.detail)
        if parent_dependency is None:
            raise ValueError('store_query requires an explicitly selected question interpretation')
        workspace = agent.interpretations
        language = _capture(workspace, parent_dependency.group_id, parent_dependency.candidate_id, SentenceAlternative)
        if language.comparison != _expected_basis(parent_dependency):
            raise ValueError('selected question comparison changed')
        candidates = [index for index, act in enumerate(_candidate(language).payload.acts)
                      if act.kind == 'question' and _same(act.meaning, question) and _same(act.frame, question.frame)]
        if len(candidates) != 1:
            raise ValueError('full question must match exactly one selected retained act')
        inherited = grounding_dependencies(agent, language.group_id, language.candidate_id)
        if isinstance(inherited, Unknown):
            raise ValueError(inherited.detail)
        dependencies = tuple(dict.fromkeys((parent_dependency, *inherited, admitted_handle.dependency)))
        batch = model.propose(deepcopy(question))
        source = workspace.add_source('Learned full-question store_query alternatives',
            modality='store_query-proposals', provider='learned-store_query-correspondence', payload=deepcopy(batch),
            metadata={'question': deepcopy(question), 'language': language, 'model': admitted_handle,
                      'dependencies': dependencies})
        expected_source = deepcopy(source)
        expected_children = []
        group = workspace.create_group(source.id, provenance=('store_query alternatives; explicit choice required',))
        for proposal in batch.proposals:
            intended = deepcopy(proposal)
            child = workspace.propose(group.id, proposal, provenance=('learned-store_query-plan',))
            expected_children.append((child.id, intended, ('learned-store_query-plan',)))
        for unresolved in batch.unresolved:
            intended = Unknown('store_query_projection_unresolved', str(unresolved))
            child = workspace.propose(group.id, intended, provenance=('unresolved-store_query-plan',))
            expected_children.append((child.id, intended, ('unresolved-store_query-plan',)))
        retained_group = workspace.get(group.id)
        if not _same(tuple((c.id, c.payload, c.provenance) for c in retained_group.candidates), tuple(expected_children)):
            raise ValueError('published store_query alternative differs from learned proposal')
        cached = deepcopy((expected_source, retained_group))
        _validate_snapshot(workspace, language, SentenceAlternative)
        if isinstance(get_store_query_model(agent, admitted_handle), Unknown) or validate_dependencies(workspace, dependencies) is not True:
            raise ValueError('store_query support changed during publication')
        if not _same(workspace.get_source(source.id), cached[0]) or not _same(workspace.get(group.id), cached[1]):
            raise ValueError('store_query alternatives changed during publication')
        _registry(agent, '_store_query_groups')[group.id] = cached
        return StoreQueryReport(group.id, tuple(c.id for c in retained_group.candidates[:len(batch.proposals)]),
                               source.id, batch.complete, batch.unresolved)
    except Exception as error:
        return Unknown('store_query_projection_unavailable', f'{type(error).__name__}: {error}')


def _validate_group(agent, group_id):
    retained = _registry(agent, '_store_query_groups').get(group_id)
    if retained is None:
        raise ValueError('unrecognized store_query alternative group')
    source, original = retained
    workspace = agent.interpretations
    group = workspace.get(group_id)
    comparison = workspace.comparison_basis(group_id)
    if (group.source_id != source.id or group.provenance != original.provenance
            or not _same(workspace.get_source(source.id), source)
            or not _same(tuple((c.id, c.payload, c.provenance) for c in group.candidates),
               tuple((c.id, c.payload, c.provenance) for c in original.candidates))):
        raise ValueError('store_query alternatives or evidence changed')
    language = source.metadata['language']
    _validate_snapshot(workspace, language, SentenceAlternative)
    support = grounding_dependencies(agent, language.group_id, language.candidate_id)
    dependencies = source.metadata['dependencies']
    if isinstance(support, Unknown) or any(d not in dependencies for d in support):
        raise ValueError('store_query reading support changed')
    if isinstance(get_store_query_model(agent, source.metadata['model']), Unknown):
        raise ValueError('store_query model withdrawn or changed')
    if validate_dependencies(workspace, dependencies) is not True:
        raise ValueError('store_query interpretation dependency changed')
    if not _same(workspace.get_source(source.id), source) or not _same(workspace.get(group_id), group):
        raise ValueError('store_query evidence changed during validation')
    _final_comparisons(workspace, (language,))
    for dependency in dependencies:
        if workspace.comparison_basis(dependency.group_id) != _expected_basis(dependency):
            raise ValueError('store_query support comparison changed during validation')
    if workspace.comparison_basis(group_id) != comparison:
        raise ValueError('store_query group comparison changed during validation')
    return source, group


def select_store_query(agent, group_id, *, decision=None):
    from .core import InterpretationDecision
    try:
        source, group = _validate_group(agent, group_id)
        if not source.payload.complete or source.payload.unresolved:
            raise ValueError('store_query alternatives retain incomplete or unresolved search')
        comparison = agent.interpretations.comparison_basis(group_id)
        if decision is None:
            selector = getattr(agent, 'store_query_selector', None)
            if selector is None:
                return Unknown('store_query_selection_required')
            decision = selector(deepcopy(group))
        if type(decision) is not InterpretationDecision or not decision.reason.strip():
            raise ValueError('an explicit store_query InterpretationDecision is required')
        _validate_group(agent, group_id)
        if agent.interpretations.comparison_basis(group_id) != comparison:
            raise ValueError('store_query comparison changed during decision')
        if decision.compared_revision is not None and decision.compared_revision != group.revision:
            raise ValueError('store_query decision revision is stale')
        if decision.compared_candidate_ids is not None and decision.compared_candidate_ids != tuple(c.id for c in group.candidates):
            raise ValueError('store_query decision comparison is stale')
        if decision.candidate_id is None:
            return Unknown('store_query_selection_deferred', decision.reason)
        candidate = next(c for c in group.candidates if c.id == decision.candidate_id and not c.rejected)
        if not hasattr(candidate.payload, 'plan'):
            raise ValueError('unresolved store_query entry cannot authorize a call')
        agent.interpretations.select(group_id, candidate.id, reason=decision.reason, evidence_ids=decision.evidence_ids)
        dependency = capture_dependency(agent.interpretations, group_id,
            basis=('explicit learned store_query plan choice', decision.reason), evidence_ids=(source.id,))
        selected = StoreQuerySelection(group_id, candidate.id, deepcopy(candidate.payload.plan),
                                       (*source.metadata['dependencies'], dependency))
        _validate_group(agent, group_id)
        if validate_dependencies(agent.interpretations, selected.dependencies) is not True:
            raise ValueError('store_query choice changed')
        return selected
    except Exception as error:
        return Unknown('store_query_selection_unavailable', f'{type(error).__name__}: {error}')



@dataclass(frozen=True)
class StoreAnswer:
    id: str
    evidence_source_id: str
    answers: tuple
    record_ids: tuple[str, ...]
    selection: StoreQuerySelection


def _validate_selection(agent, selection):
    source, group = _validate_group(agent, selection.group_id)
    if group.selected_id != selection.candidate_id or not _same(group.selected.payload.plan, selection.plan):
        raise ValueError('selected store query changed')
    if validate_dependencies(agent.interpretations, selection.dependencies) is not True:
        raise ValueError('store query dependencies changed')
    expected = (*source.metadata['dependencies'],)
    if selection.dependencies[:-1] != expected:
        raise ValueError('store query dependencies are not authentic')
    final = selection.dependencies[-1]
    if final.group_id != group.id or final.candidate_id != group.selected_id:
        raise ValueError('store query choice dependency mismatch')
    for dependency in selection.dependencies:
        if agent.interpretations.comparison_basis(dependency.group_id) != _expected_basis(dependency):
            raise ValueError('store query comparison changed')


def _validate_support(agent, record_ids):
    from ..derivations import validate_record_supports
    supported = validate_record_supports(agent.store, tuple(record_ids))
    if supported is not True:
        raise ValueError(f'store answer support is unavailable: {supported}')


def evaluate_store_query(agent, selection):
    """Evaluate only the explicitly selected full query; retain all supporting records."""
    from .informing_learning import _match_answer
    try:
        if type(selection) is not StoreQuerySelection:
            raise ValueError('an authentic selected store query is required')
        _validate_selection(agent, selection)
        plan = deepcopy(selection.plan)
        if not any(_same(plan.query.scope, scope) for scope in plan.allowed_scopes):
            raise ValueError('query scope is not explicitly allowed')
        records = deepcopy(agent.store.propositions(plan.query.predicate))
        answers, support, unsupported = [], [], []
        for record in records:
            binding = _match_answer(plan.query, record.proposition)
            if binding is None or plan.answer_variable not in binding:
                continue
            try:
                _validate_support(agent, (record.id,))
            except ValueError as error:
                unsupported.append((record.id, str(error)))
                continue
            answers.append(deepcopy(binding[plan.answer_variable]))
            support.append(record)
        if not answers:
            raise ValueError('no supported answer; absence is not an observed empty result')
        identifier = 'store-answer:' + uuid4().hex
        payload = {'selection': deepcopy(selection), 'records': deepcopy(records),
                   'support': deepcopy(support), 'unsupported': tuple(unsupported), 'answers': tuple(answers)}
        source = agent.interpretations.add_source('Supported learned store answer', modality='store-answer',
            provider='learned-store-query', payload=deepcopy(payload),
            metadata={'record_ids': tuple(r.id for r in support), 'dependencies': selection.dependencies})
        if not _same(source.payload, payload):
            raise ValueError('store answer evidence changed during publication')
        result = StoreAnswer(identifier, source.id, tuple(answers), tuple(r.id for r in support), deepcopy(selection))
        cached, returned = deepcopy((result, source)), deepcopy(result)
        _validate_selection(agent, selection)
        if not _same(agent.store.propositions(plan.query.predicate), records):
            raise ValueError('store changed during answering')
        if not _same(agent.interpretations.get_source(source.id), cached[1]):
            raise ValueError('store answer evidence changed')
        if not _same(agent.store.propositions(plan.query.predicate), records):
            raise ValueError('store changed during evidence validation')
        for dependency in selection.dependencies:
            if agent.interpretations.comparison_basis(dependency.group_id) != _expected_basis(dependency):
                raise ValueError('store query authority changed during answering')
        _validate_support(agent, returned.record_ids)
        _validate_selection(agent, selection)
        if not _same(agent.interpretations.get_source(source.id), cached[1]):
            raise ValueError('answer evidence changed during derivation replay')
        _registry(agent, '_store_query_answers')[identifier] = cached
        return returned
    except Exception as error:
        return Unknown('store_answer_unavailable', f'{type(error).__name__}: {error}')


def validate_store_answer(agent, answer):
    """Cached answers have no authority after support, evidence, or model changes."""
    try:
        cached = _registry(agent, '_store_query_answers').get(answer.id)
        if cached is None or not _same(answer, cached[0]):
            raise ValueError('unrecognized or changed store answer')
        expected, source = cached
        _validate_support(agent, expected.record_ids)
        _validate_selection(agent, expected.selection)
        if not _same(agent.interpretations.get_source(source.id), source):
            raise ValueError('retained store answer evidence changed')
        if not _same(agent.store.propositions(expected.selection.plan.query.predicate), source.payload['records']):
            raise ValueError('supporting store records changed or were retracted')
        _validate_selection(agent, expected.selection)
        if not _same(agent.interpretations.get_source(source.id), source):
            raise ValueError('answer source changed during support validation')
        if not _same(agent.store.propositions(expected.selection.plan.query.predicate), source.payload['records']):
            raise ValueError('store changed during support validation')
        for dependency in expected.selection.dependencies:
            if agent.interpretations.comparison_basis(dependency.group_id) != _expected_basis(dependency):
                raise ValueError('store query authority changed during final validation')
        _validate_support(agent, expected.record_ids)
        _validate_selection(agent, expected.selection)
        if not _same(agent.interpretations.get_source(source.id), source):
            raise ValueError('answer evidence changed during derivation replay')
        return True
    except Exception as error:
        return Unknown('store_answer_stale', f'{type(error).__name__}: {error}')


def answer_store_question(agent, question, act, events, *, parent_dependency=None):
    from .core import Outcome
    handle = getattr(agent, 'store_query_model', None)
    if handle is None:
        return Outcome(act, 'unknown', reason='no admitted store query correspondence')
    report = propose_store_queries(agent, handle, question, parent_dependency=parent_dependency)
    if isinstance(report, Unknown):
        return Outcome(act, 'unknown', reason=report.reason, verified=report)
    events.append({'type': 'store_query_interpretation', 'group_id': report.group_id,
                   'source_id': report.source_id, 'candidate_ids': report.candidate_ids})
    selection = select_store_query(agent, report.group_id)
    if isinstance(selection, Unknown):
        return Outcome(act, 'unknown', reason=selection.reason, verified=selection)
    answer = evaluate_store_query(agent, selection)
    if isinstance(answer, Unknown):
        return Outcome(act, 'unknown', plan=selection, reason=answer.reason, verified=answer)
    checked = validate_store_answer(agent, answer)
    if isinstance(checked, Unknown):
        return Outcome(act, 'unknown', plan=selection, reason=checked.reason, verified=checked)
    events.append({'type': 'store_answer', 'source_id': answer.evidence_source_id, 'record_ids': answer.record_ids})
    return Outcome(act, 'answered', plan=selection, answer=list(answer.answers), verified=answer)
