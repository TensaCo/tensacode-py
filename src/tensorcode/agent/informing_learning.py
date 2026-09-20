"""Explicit full-question informing correspondences and guarded read-only answers.

No grammatical role ordering chooses a tool argument. Learned alternatives
preserve the whole question; caller selection and declared read contracts are
separate requirements before observation can support an answer.
"""
from copy import deepcopy
from dataclasses import dataclass, replace, fields, is_dataclass
from datetime import datetime, timezone
from uuid import uuid4

from ..learning.experience import _same
from ..language import Question
from ..outcomes import Unknown
from ..records import Proposition, Claim, Ref, Var, Evidence
from ..runtime import use
from .understand import SentenceAlternative
from .scene_grounding import _capture, _candidate, _validate_snapshot, _final_comparisons, grounding_dependencies
from .task_dependencies import capture_dependency, validate_dependencies, _expected_basis


@dataclass(frozen=True)
class RetainedInformingExample:
    example: object
    language: object
    act_index: int
    evidence_source_id: str


@dataclass(frozen=True)
class InformingModelHandle:
    group_id: str
    candidate_id: str
    model_id: str
    evidence_source_id: str
    dependency: object = None


@dataclass(frozen=True)
class InformingReport:
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


def retain_informing_example(agent, group_id, candidate_id, act_index, plan, *, basis):
    from ..learning.informing import InformingExample, InformingPlan
    try:
        if type(plan) is not InformingPlan:
            raise ValueError('an explicit InformingPlan is required')
        if type(basis) is not tuple or not basis or any(type(x) is not str or not x.strip() for x in basis):
            raise ValueError('informing teaching requires an explicit basis')
        language = _capture(agent.interpretations, group_id, candidate_id, SentenceAlternative)
        question = _question(language, act_index)
        example = InformingExample('informing-example:' + uuid4().hex, language.source.id,
            language.source.text, question, deepcopy(plan), basis)
        support = grounding_dependencies(agent, group_id, candidate_id)
        if isinstance(support, Unknown):
            raise ValueError(support.detail)
        evidence = agent.interpretations.add_source('Explicit full-question informing teaching',
            modality='teaching', provider='explicit-informing-teaching', payload=deepcopy(example),
            metadata={'language': language, 'act_index': act_index, 'dependencies': support})
        result = RetainedInformingExample(example, language, act_index, evidence.id)
        cached, returned = deepcopy((result, evidence)), deepcopy(result)
        _validate_snapshot(agent.interpretations, language, SentenceAlternative)
        if validate_dependencies(agent.interpretations, support) is not True:
            raise ValueError('informing teaching support changed')
        _final_comparisons(agent.interpretations, (language,))
        _registry(agent, '_informing_examples')[evidence.id] = cached
        return returned
    except Exception as error:
        return Unknown('informing_teaching_unavailable', f'{type(error).__name__}: {error}')


def _validate_examples(agent, records):
    for record in records:
        cached = _registry(agent, '_informing_examples').get(record.evidence_source_id)
        if cached is None or not _same(cached[0], record) or not _same(agent.interpretations.get_source(record.evidence_source_id), cached[1]):
            raise ValueError('informing teaching content changed')
        _validate_snapshot(agent.interpretations, record.language, SentenceAlternative)
        if not _same(_question(record.language, record.act_index), record.example.question):
            raise ValueError('informing teaching question changed')
        support = grounding_dependencies(agent, record.language.group_id, record.language.candidate_id)
        if isinstance(support, Unknown) or support != cached[1].metadata['dependencies']:
            raise ValueError('informing teaching support changed')
        if validate_dependencies(agent.interpretations, support) is not True:
            raise ValueError('informing teaching dependency changed')
    _final_comparisons(agent.interpretations, tuple(record.language for record in records))


def _model_state(agent, group_id):
    saved = _registry(agent, '_informing_models').get(group_id)
    if saved is None:
        raise ValueError('unrecognized informing model group')
    source, provenance, versions = saved
    workspace = agent.interpretations
    comparison = workspace.comparison_basis(group_id)
    group = workspace.get(group_id)
    if group.source_id != source.id or group.provenance != provenance or not _same(workspace.get_source(source.id), source):
        raise ValueError('informing model source changed')
    if tuple(c.id for c in group.candidates) != tuple(v[0] for v in versions):
        raise ValueError('informing model versions changed')
    for candidate, (_, version, evidence, model) in zip(group.candidates, versions):
        if (not _same(candidate.payload, version) or not _same(workspace.get_source(evidence.id), evidence)
                or not _same(vars(model), version.snapshot)):
            raise ValueError('informing model content changed')
    if workspace.comparison_basis(group_id) != comparison:
        raise ValueError('informing model comparison changed during validation')
    return group, comparison, versions


def fit_informing_model(agent, training_records, validation_records, *, group_id=None, max_pairs=256):
    from ..learning.informing import fit_informing
    published = None
    try:
        training, validation = tuple(training_records), tuple(validation_records)
        records = (*training, *validation)
        _validate_examples(agent, records)
        if {r.example.source_id for r in training} & {r.example.source_id for r in validation}:
            raise ValueError('informing training and heldout sources overlap')
        previous = _model_state(agent, group_id) if group_id is not None else None
        model = fit_informing(tuple(r.example for r in training), tuple(r.example for r in validation), max_pairs=max_pairs)
        evidence = agent.interpretations.add_source('Historical fitted informing correspondences',
            modality='informing-model', provider='learned-informing-correspondence', payload=deepcopy(vars(model)),
            metadata={'training': training, 'validation': validation, 'max_pairs': max_pairs})
        version = _Version(deepcopy(vars(model)), evidence.id)
        cached = deepcopy((version, evidence, model))
        _validate_examples(agent, records)
        workspace = agent.interpretations
        if previous is None:
            group = workspace.create_group(evidence.id, provenance=('informing model versions',))
            saved = (deepcopy(evidence), group.provenance, ())
        else:
            group, comparison, _ = _model_state(agent, group_id)
            if comparison != previous[1]:
                raise ValueError('informing model admission changed during refit')
            saved = _registry(agent, '_informing_models')[group.id]
        candidate = workspace.propose(group.id, version, provenance=('retained-informing-model',))
        _registry(agent, '_informing_models')[group.id] = (*saved[:2], (*saved[2], (candidate.id, *cached)))
        published = (group.id, candidate.id)
        workspace.unset(group.id, reason='new informing fit requires explicit admission')
        _validate_examples(agent, records)
        _model_state(agent, group.id)
        return InformingModelHandle(group.id, candidate.id, model.id, evidence.id)
    except Exception as error:
        if published:
            agent.interpretations.reject(*published, reason='informing fit evidence changed')
        return Unknown('informing_fit_unavailable', f'{type(error).__name__}: {error}')


def admit_informing_model(agent, handle, *, reason):
    try:
        if type(handle) is not InformingModelHandle or type(reason) is not str or not reason.strip():
            raise ValueError('informing admission requires authentic handle and reason')
        group, comparison, versions = _model_state(agent, handle.group_id)
        rows = [row for row in versions if row[0] == handle.candidate_id]
        if len(rows) != 1 or rows[0][3].id != handle.model_id or rows[0][2].id != handle.evidence_source_id or not rows[0][3].complete:
            raise ValueError('informing model incomplete or handle mismatched')
        candidate = next(c for c in group.candidates if c.id == handle.candidate_id)
        if candidate.rejected:
            raise ValueError('informing model version rejected')
        expected = (comparison[0], group.revision + 1, candidate.id, comparison[3], False, comparison[5], comparison[6])
        agent.interpretations.select(group.id, candidate.id, reason=reason, evidence_ids=(handle.evidence_source_id,))
        dependency = capture_dependency(agent.interpretations, group.id,
            basis=('explicit informing model admission', reason), evidence_ids=(handle.evidence_source_id,))
        result = replace(handle, dependency=dependency)
        _model_state(agent, group.id)
        if agent.interpretations.comparison_basis(group.id) != expected or validate_dependencies(agent.interpretations, (dependency,)) is not True:
            raise ValueError('informing admission changed')
        _registry(agent, '_informing_admissions')[(group.id, candidate.id)] = deepcopy(result)
        return result
    except Exception as error:
        return Unknown('informing_admission_unavailable', f'{type(error).__name__}: {error}')


def get_informing_model(agent, handle):
    try:
        if type(handle) is not InformingModelHandle or handle.dependency is None:
            raise ValueError('informing model not admitted')
        cached = _registry(agent, '_informing_admissions').get((handle.group_id, handle.candidate_id))
        if not _same(handle, cached):
            raise ValueError('informing model admission changed')
        group, comparison, versions = _model_state(agent, handle.group_id)
        if group.selected_id != handle.candidate_id or validate_dependencies(agent.interpretations, (handle.dependency,)) is not True:
            raise ValueError('informing model dependency withdrawn')
        model = deepcopy(next(v[3] for v in versions if v[0] == handle.candidate_id))
        _model_state(agent, group.id)
        if agent.interpretations.comparison_basis(group.id) != comparison:
            raise ValueError('informing model changed during retrieval')
        return model
    except Exception as error:
        return Unknown('informing_model_unavailable', f'{type(error).__name__}: {error}')




@dataclass(frozen=True)
class InformingSelection:
    group_id: str
    candidate_id: str
    plan: object
    dependencies: tuple


def propose_informing(agent, admitted_handle, question, *, parent_dependency):
    """Retain all learned plans; never select a plan or infer a parameter mapping."""
    try:
        model = get_informing_model(agent, admitted_handle)
        if isinstance(model, Unknown):
            raise ValueError(model.detail)
        if parent_dependency is None:
            raise ValueError('informing requires an explicitly selected question interpretation')
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
        source = workspace.add_source('Learned full-question informing alternatives',
            modality='informing-proposals', provider='learned-informing-correspondence', payload=deepcopy(batch),
            metadata={'question': deepcopy(question), 'language': language, 'model': admitted_handle,
                      'dependencies': dependencies})
        expected_source = deepcopy(source)
        expected_children = []
        group = workspace.create_group(source.id, provenance=('informing alternatives; explicit choice required',))
        for proposal in batch.proposals:
            intended = deepcopy(proposal)
            child = workspace.propose(group.id, proposal, provenance=('learned-informing-plan',))
            expected_children.append((child.id, intended, ('learned-informing-plan',)))
        for unresolved in batch.unresolved:
            intended = Unknown('informing_projection_unresolved', str(unresolved))
            child = workspace.propose(group.id, intended, provenance=('unresolved-informing-plan',))
            expected_children.append((child.id, intended, ('unresolved-informing-plan',)))
        retained_group = workspace.get(group.id)
        if not _same(tuple((c.id, c.payload, c.provenance) for c in retained_group.candidates), tuple(expected_children)):
            raise ValueError('published informing alternative differs from learned proposal')
        cached = deepcopy((expected_source, retained_group))
        _validate_snapshot(workspace, language, SentenceAlternative)
        if isinstance(get_informing_model(agent, admitted_handle), Unknown) or validate_dependencies(workspace, dependencies) is not True:
            raise ValueError('informing support changed during publication')
        if not _same(workspace.get_source(source.id), cached[0]) or not _same(workspace.get(group.id), cached[1]):
            raise ValueError('informing alternatives changed during publication')
        _registry(agent, '_informing_groups')[group.id] = cached
        return InformingReport(group.id, tuple(c.id for c in retained_group.candidates[:len(batch.proposals)]),
                               source.id, batch.complete, batch.unresolved)
    except Exception as error:
        return Unknown('informing_projection_unavailable', f'{type(error).__name__}: {error}')


def _validate_group(agent, group_id):
    retained = _registry(agent, '_informing_groups').get(group_id)
    if retained is None:
        raise ValueError('unrecognized informing alternative group')
    source, original = retained
    workspace = agent.interpretations
    group = workspace.get(group_id)
    comparison = workspace.comparison_basis(group_id)
    if (group.source_id != source.id or group.provenance != original.provenance
            or not _same(workspace.get_source(source.id), source)
            or not _same(tuple((c.id, c.payload, c.provenance) for c in group.candidates),
               tuple((c.id, c.payload, c.provenance) for c in original.candidates))):
        raise ValueError('informing alternatives or evidence changed')
    language = source.metadata['language']
    _validate_snapshot(workspace, language, SentenceAlternative)
    support = grounding_dependencies(agent, language.group_id, language.candidate_id)
    dependencies = source.metadata['dependencies']
    if isinstance(support, Unknown) or any(d not in dependencies for d in support):
        raise ValueError('informing reading support changed')
    if isinstance(get_informing_model(agent, source.metadata['model']), Unknown):
        raise ValueError('informing model withdrawn or changed')
    if validate_dependencies(workspace, dependencies) is not True:
        raise ValueError('informing interpretation dependency changed')
    if not _same(workspace.get_source(source.id), source) or not _same(workspace.get(group_id), group):
        raise ValueError('informing evidence changed during validation')
    _final_comparisons(workspace, (language,))
    for dependency in dependencies:
        if workspace.comparison_basis(dependency.group_id) != _expected_basis(dependency):
            raise ValueError('informing support comparison changed during validation')
    if workspace.comparison_basis(group_id) != comparison:
        raise ValueError('informing group comparison changed during validation')
    return source, group


def select_informing(agent, group_id, *, decision=None):
    from .core import InterpretationDecision
    try:
        source, group = _validate_group(agent, group_id)
        if not source.payload.complete or source.payload.unresolved:
            raise ValueError('informing alternatives retain incomplete or unresolved search')
        comparison = agent.interpretations.comparison_basis(group_id)
        if decision is None:
            selector = getattr(agent, 'informing_selector', None)
            if selector is None:
                return Unknown('informing_selection_required')
            decision = selector(deepcopy(group))
        if type(decision) is not InterpretationDecision or not decision.reason.strip():
            raise ValueError('an explicit informing InterpretationDecision is required')
        _validate_group(agent, group_id)
        if agent.interpretations.comparison_basis(group_id) != comparison:
            raise ValueError('informing comparison changed during decision')
        if decision.compared_revision is not None and decision.compared_revision != group.revision:
            raise ValueError('informing decision revision is stale')
        if decision.compared_candidate_ids is not None and decision.compared_candidate_ids != tuple(c.id for c in group.candidates):
            raise ValueError('informing decision comparison is stale')
        if decision.candidate_id is None:
            return Unknown('informing_selection_deferred', decision.reason)
        candidate = next(c for c in group.candidates if c.id == decision.candidate_id and not c.rejected)
        if not hasattr(candidate.payload, 'plan'):
            raise ValueError('unresolved informing entry cannot authorize a call')
        agent.interpretations.select(group_id, candidate.id, reason=decision.reason, evidence_ids=decision.evidence_ids)
        dependency = capture_dependency(agent.interpretations, group_id,
            basis=('explicit learned informing plan choice', decision.reason), evidence_ids=(source.id,))
        selected = InformingSelection(group_id, candidate.id, deepcopy(candidate.payload.plan),
                                       (*source.metadata['dependencies'], dependency))
        _validate_group(agent, group_id)
        if validate_dependencies(agent.interpretations, selected.dependencies) is not True:
            raise ValueError('informing choice changed')
        return selected
    except Exception as error:
        return Unknown('informing_selection_unavailable', f'{type(error).__name__}: {error}')


def _substitute(value, args):
    """Instantiate only explicit named query variables, preserving full metadata."""
    if type(value) is Var:
        return deepcopy(args.get(value.name, value))
    if type(value) is dict:
        return {key: _substitute(item, args) for key, item in value.items()}
    if type(value) is tuple:
        return tuple(_substitute(item, args) for item in value)
    if type(value) is list:
        return [_substitute(item, args) for item in value]
    if is_dataclass(value):
        return replace(value, **{field.name: _substitute(getattr(value, field.name), args)
                                 for field in fields(value) if field.init})
    return deepcopy(value)


def _match_answer(pattern, observed):
    """Bind explicit holes under exact typed structure, including all metadata.

    None is a literal scope/validity value, never an implicit wildcard. Declared
    roles must match exactly; broader observation projections require teaching
    and a provider contract that explicitly describe that projection.
    """
    bindings = {}
    def match(wanted, actual):
        if type(wanted) is Var:
            if type(actual) is Var:
                return False
            if wanted.name in bindings:
                return _same(bindings[wanted.name], actual)
            bindings[wanted.name] = deepcopy(actual)
            return True
        if type(wanted) is not type(actual):
            return False
        if is_dataclass(wanted):
            return all(match(getattr(wanted, field.name), getattr(actual, field.name))
                       for field in fields(wanted))
        if type(wanted) is dict:
            if len(wanted) != len(actual):
                return False
            for key, value in wanted.items():
                keys = [other for other in actual if _same(key, other)]
                if len(keys) != 1 or not match(value, actual[keys[0]]):
                    return False
            return True
        if type(wanted) in (list, tuple):
            return len(wanted) == len(actual) and all(match(a, b) for a, b in zip(wanted, actual))
        return _same(wanted, actual)
    return bindings if match(pattern, observed) else None


def _provider_contract(agent, plan):
    providers = [plugin for plugin in agent.plugins if plugin.name == plan.plugin]
    if len(providers) != 1:
        raise ValueError('informing plan must identify one available provider')
    provider = providers[0]
    capabilities = [cap for cap in provider.capabilities() if cap.name == plan.capability]
    if len(capabilities) != 1:
        raise ValueError('informing plan must identify one declared capability')
    capability = capabilities[0]
    args = dict(plan.args)
    if len(args) != len(plan.args) or set(args) != {param.name for param in capability.params} or capability.effect_kind != 'read':
        raise ValueError('informing requires exact parameters of a read-only capability')
    contracts = [inf for inf in capability.informs if inf.query is not None
                 and inf.answer == plan.answer_variable
                 and _same(_substitute(inf.query, args), plan.answer_query)]
    if not contracts:
        raise ValueError('learned answer query does not match a declared capability contract')
    return provider, capability, args


def answer_informing_question(agent, question, act, events, *, parent_dependency):
    """Answer via one explicitly chosen learned plan and validated fresh evidence."""
    from .core import Outcome
    handle = getattr(agent, 'informing_model', None)
    if handle is None:
        return None  # Passive lookup remains separate; no informing action is inferred.
    report = propose_informing(agent, handle, question, parent_dependency=parent_dependency)
    if isinstance(report, Unknown):
        return Outcome(act, 'unknown', verified=report, reason=report.reason)
    events.append({'type': 'informing_interpretation', 'group_id': report.group_id,
                   'source_id': report.source_id, 'candidate_ids': report.candidate_ids})
    selected = select_informing(agent, report.group_id)
    if isinstance(selected, Unknown):
        return Outcome(act, 'unknown', verified=selected, reason=selected.reason)
    receipt = None
    observations = []
    try:
        provider, capability, args = _provider_contract(agent, selected.plan)
        expected_capability = deepcopy(capability)
        expected_plan = deepcopy(selected.plan)
        def validate():
            current_provider, current_capability, current_args = _provider_contract(agent, expected_plan)
            if current_provider is not provider or not _same(current_capability, expected_capability) or not _same(current_args, args):
                raise ValueError('informing provider capability changed')
            _, group = _validate_group(agent, selected.group_id)
            if group.selected_id != selected.candidate_id or not _same(group.selected.payload.plan, expected_plan):
                raise ValueError('selected informing plan changed')
            if validate_dependencies(agent.interpretations, selected.dependencies) is not True:
                raise ValueError('informing authorization changed')
            for dependency in selected.dependencies:
                if agent.interpretations.comparison_basis(dependency.group_id) != _expected_basis(dependency):
                    raise ValueError('informing comparison changed during validation')
        def guard(_sources):
            try:
                validate()
                return True
            except Exception as error:
                return Unknown('informing_authorization_changed', str(error))
        validate()
        with use(agent.runtime):
            receipt = agent._invoke(provider, capability, args, events, before_dispatch=guard)
        validate()
        if receipt.status != 'applied':
            return Outcome(act, 'unknown', plan=selected, receipt=receipt, reason=receipt.error or receipt.status)
        observations = list(provider.reveal(capability, deepcopy(args), receipt))
        validate()
        answers = []
        propositions = []
        for observed in observations:
            proposition = (Proposition(observed.predicate, {'subject': observed.subject, 'object': observed.object},
                valid=observed.valid, scope=observed.scope) if isinstance(observed, Claim) else observed)
            if type(proposition) is not Proposition:
                raise ValueError('informing provider returned an unsupported observation')
            binding = _match_answer(expected_plan.answer_query, proposition)
            if binding is None or expected_plan.answer_variable not in binding:
                raise ValueError('observations did not satisfy the declared answer query')
            answers.append(binding[expected_plan.answer_variable])
            propositions.append(proposition)
        if not answers:
            raise ValueError('no matching answer observation; absence is not an observed empty result')
        evidence_source = agent.interpretations.add_source('Verified informing response', modality='informing-answer',
            provider=provider.name, payload={'receipt': receipt, 'observations': observations, 'answers': answers},
            metadata={'informing_group_id': selected.group_id, 'dependencies': selected.dependencies})
        expected_evidence = deepcopy(evidence_source)
        validate()
        if not _same(agent.interpretations.get_source(evidence_source.id), expected_evidence):
            raise ValueError('informing response evidence changed')
        for dependency in selected.dependencies:
            if agent.interpretations.comparison_basis(dependency.group_id) != _expected_basis(dependency):
                raise ValueError('informing response authority changed during retention')
        for proposition in propositions:
            agent.store.assert_(proposition, Evidence(source=Ref('plugin:' + provider.name),
                observed_at=datetime.now(timezone.utc), locator=evidence_source.id, method=capability.name))
        return Outcome(act, 'answered', plan=selected, receipt=receipt, answer=answers)
    except Exception as error:
        failure = Unknown('informing_observation_unavailable', f'{type(error).__name__}: {error}')
        agent.interpretations.add_source('Informing attempt did not establish an answer', modality='informing-answer-failed',
            provider=selected.plan.plugin, payload={'receipt': receipt, 'observations': observations, 'failure': failure},
            metadata={'informing_group_id': selected.group_id, 'dependencies': selected.dependencies})
        return Outcome(act, 'unknown', plan=selected, receipt=receipt, verified=failure, reason=failure.detail)
