"""Authenticated contextual browser outcomes, with supplied literal CDP decoding.

Only outcome associations are learned. The projection identifies the acted-on
node in each observation; it contains no click/toggle rule or tag affordance.
"""
from copy import deepcopy
from dataclasses import dataclass
from uuid import uuid4

from ..learning.experience import (Projection, TransitionBatch, ExcludedTransition,
    ValidationPolicy, extract_transitions, fit_transitions, _same)
from ..outcomes import Unknown
from .plugin import Call
from .experience_planning import _validate_evidence


@dataclass(frozen=True)
class RetainedDocumentTransitions:
    id: str
    evidence_source_id: str
    batch: TransitionBatch


@dataclass(frozen=True)
class RetainedDocumentPrediction:
    id: str
    evidence_source_id: str
    prediction: object
    action: Call


def _registry(agent, key):
    if not hasattr(agent, key):
        setattr(agent, key, {})
    return getattr(agent, key)


def _require(value):
    if value is not True:
        raise ValueError(str(value))


def _target(before, action):
    if type(action) is not Call or action.capability != 'activate_node' or tuple(k for k, _ in action.args) != ('target',):
        raise ValueError('explicit activate_node target action required')
    token = action.arg('target')
    if type(token) is not str or not token:
        raise ValueError('opaque target token required')
    targets = before.get('document_targets')
    if type(targets) is not tuple:
        raise ValueError('issued target evidence tuple required')
    matches = [target for target in targets if target.token == token]
    if len(matches) != 1 or not _same(matches[0].action, action):
        raise ValueError('target/action correspondence is missing or ambiguous')
    return matches[0]


def _decode(observation, target):
    """Decode sparse inputChecked only after exact frame/backend correspondence."""
    identity = observation['document_identity']
    if (identity['connection_id'], identity['session_id'], identity['frame_loaders']) != (
            target.connection_id, target.session_id, target.frame_loaders):
        raise ValueError('document identity changed')
    snapshot = observation['document_snapshot']
    strings, documents = snapshot['strings'], snapshot['documents']
    if type(strings) is not list or any(type(s) is not str for s in strings) or type(documents) is not list:
        raise ValueError('malformed snapshot tables')
    def string(index):
        if type(index) is not int or not 0 <= index < len(strings):
            raise ValueError('invalid string index')
        return strings[index]
    matches = []
    for document in documents:
        if string(document['frameId']) != target.frame_id:
            continue
        nodes = document['nodes']
        backends = nodes['backendNodeId']
        if type(backends) is not list or any(type(n) is not int or n <= 0 for n in backends):
            raise ValueError('invalid backend node identities')
        for index, backend in enumerate(backends):
            if backend == target.backend_node_id:
                matches.append((nodes, index, len(backends)))
    if len(matches) != 1:
        raise ValueError('target missing, replaced, or ambiguous')
    nodes, index, count = matches[0]
    for field in ('nodeName', 'nodeType', 'attributes'):
        if type(nodes[field]) is not list or len(nodes[field]) != count:
            raise ValueError('node array length mismatch')
    checked = nodes['inputChecked']
    if type(checked) is not dict or set(checked) != {'index'} or type(checked['index']) is not list:
        raise ValueError('inputChecked requires a sparse Boolean index array')
    indices = checked['index']
    if any(type(i) is not int or not 0 <= i < count for i in indices) or len(indices) != len(set(indices)):
        raise ValueError('invalid inputChecked index')
    attributes = nodes['attributes'][index]
    if type(attributes) is not list or len(attributes) % 2:
        raise ValueError('malformed attribute pairs')
    # Preserve CDP no-string sentinel for attribute values, including Boolean
    # attributes. Names and other string fields still require table indices.
    pairs = [(string(attributes[i]), None if type(attributes[i + 1]) is int and attributes[i + 1] == -1
              else string(attributes[i + 1])) for i in range(0, len(attributes), 2)]
    types = [value for name, value in pairs if name == 'type']
    if len(types) > 1 or type(nodes['nodeType'][index]) is not int or nodes['nodeType'][index] < 1:
        raise ValueError('ambiguous type attribute or invalid node type')
    return {'nodeName': string(nodes['nodeName'][index]), 'nodeType': nodes['nodeType'][index],
            'type_attribute': types[0] if types else None, 'inputChecked': index in indices}


def _features(before, action):
    return _decode(before, _target(before, action))


def _outcome(before, action, after):
    target = _target(before, action)
    _decode(before, target)
    if not _same(_target(after, action), target):
        raise ValueError('target correspondence changed across observations')
    return _decode(after, target)['inputChecked']


def browser_transition_projection():
    return Projection('authenticated-browser-target-inputChecked', _features, _outcome,
        ('supplied literal CDP node fields and sparse Boolean decoder',
         'target relative correspondence authenticated by browser evidence bridge',
         'no authored activation outcome or toggle semantics'))


def _authenticate(provider, row):
    if row.before.get('document_observation_id') == row.after.get('document_observation_id'):
        raise ValueError('before and after must be independently issued observations')
    target = _target(row.before, row.action)
    for observation in (row.before, row.after):
        _require(provider.authenticate_document_observation(observation))
        _require(provider.validate_document_target_observation(target, observation))
        _decode(observation, target)
    return target


def retain_document_transition_batch(agent, provider):
    """Retain eligible applied transitions and explicit exclusions after authentication."""
    try:
        sources = agent.interpretations.sources()
        batch = extract_transitions(sources, provider='plugin:' + provider.name)
        accepted, exclusions = [], list(batch.exclusions)
        for row in batch.transitions:
            try:
                _authenticate(provider, row)
                accepted.append(row)
            except Exception as error:
                exclusions.append(ExcludedTransition(row.attempt_id, row.source_ids, str(error)))
        observations = {}
        for row in accepted:
            for observed in (row.before, row.after):
                ident = observed.get('document_observation_id')
                observations.setdefault(ident, []).append(row.attempt_id)
        aliases = {attempt for attempts in observations.values() if len(attempts) > 1 for attempt in attempts}
        for row in accepted:
            if row.attempt_id in aliases:
                exclusions.append(ExcludedTransition(row.attempt_id, row.source_ids,
                    'provider observation reused across attempts; aliases cannot supply independent support'))
        accepted = [row for row in accepted if row.attempt_id not in aliases]
        batch = TransitionBatch(tuple(accepted), tuple(exclusions))
        selected = {sid for row in accepted for sid in row.source_ids}
        originals = tuple(source for source in sources if source.id in selected)
        evidence = agent.interpretations.add_source('Authenticated document transitions',
            modality='document-transition-batch', provider=provider.name, payload=batch,
            metadata={'source_ids': tuple(source.id for source in originals)})
        result = RetainedDocumentTransitions('document-transitions:' + uuid4().hex, evidence.id, batch)
        cached = deepcopy((result, evidence, originals))
        for source in originals:
            if not _same(agent.interpretations.get_source(source.id), source):
                raise ValueError('transition evidence changed during authentication')
        _registry(agent, '_document_transition_batches')[result.id] = (provider, cached)
        return deepcopy(result)
    except Exception as error:
        return Unknown('document_transitions_unavailable', f'{type(error).__name__}: {error}')


def _validate_batch(agent, provider, batch):
    cached = _registry(agent, '_document_transition_batches').get(batch.id)
    if cached is None or cached[0] is not provider or not _same(batch, cached[1][0]):
        raise ValueError('unrecognized document transition batch')
    _, evidence, originals = cached[1]
    for row in batch.batch.transitions:
        _authenticate(provider, row)
    if not _same(agent.interpretations.get_source(evidence.id), evidence):
        raise ValueError('document batch evidence changed')
    for source in originals:
        if not _same(agent.interpretations.get_source(source.id), source):
            raise ValueError('original transition observation changed')


def validate_document_split(transitions, train_attempt_ids, evaluation_attempt_ids):
    """Reject document reuse across splits, even across distinct attempt/target IDs."""
    rows = {row.attempt_id: row for row in transitions}
    def keys(ids):
        result = set()
        for ident in ids:
            row = rows[ident]
            target = _target(row.before, row.action)
            result.add((target.connection_id, target.session_id, target.frame_id, target.frame_loaders))
        return result
    if keys(train_attempt_ids) & keys(evaluation_attempt_ids):
        raise ValueError('training and evaluation reuse a document identity')
    return True


def fit_document_transitions(agent, provider, retained_batch, *, train_attempt_ids,
                             evaluation_attempt_ids, policy=ValidationPolicy()):
    try:
        train, held = tuple(train_attempt_ids), tuple(evaluation_attempt_ids)
        _validate_batch(agent, provider, retained_batch)
        validate_document_split(retained_batch.batch.transitions, train, held)
        model = fit_transitions(retained_batch.batch.transitions, projection=browser_transition_projection(),
            train_attempt_ids=train, evaluation_attempt_ids=held, policy=policy)
        _validate_batch(agent, provider, retained_batch)
        _registry(agent, '_document_transition_models')[model.id] = (
            provider, deepcopy(retained_batch), model, model.projection,
            model.artifact, model.examples, deepcopy(model.policy))
        return model
    except Exception as error:
        return Unknown('document_transition_fit_unavailable', f'{type(error).__name__}: {error}')


def _model_contract(agent, provider, model):
    registered = _registry(agent, '_document_transition_models').get(model.id)
    if registered is None or registered[0] is not provider or registered[2] is not model:
        raise ValueError('model was not fitted from authenticated document transitions')
    if (model.projection is not registered[3] or not _same(model.artifact, registered[4])
            or not _same(model.examples, registered[5]) or not _same(model.policy, registered[6])):
        raise ValueError('fitted model rule, projection, examples, or policy changed')
    return registered


def _current_prediction(agent, provider, model, prediction, snapshot):
    _model_contract(agent, provider, model)
    if not _same(model.snapshot(), snapshot):
        raise ValueError('model changed during prediction or observation callbacks')
    if not isinstance(prediction, Unknown):
        _validate_evidence(agent, model, prediction)
    _model_contract(agent, provider, model)
    if not _same(model.snapshot(), snapshot):
        raise ValueError('model changed during evidence replay')


def predict_document_transition(agent, provider, model, target_token):
    """Authenticate and retain a fresh before-context; never observe a future outcome."""
    try:
        registered = _model_contract(agent, provider, model)
        model_snapshot = model.snapshot()
        _validate_batch(agent, provider, registered[1])
        _require(provider.validate_document_target(target_token))
        observation = provider.observe_evidence()
        action = Call(provider.name, 'activate_node', (('target', target_token),))
        target = _target(observation, action)
        _require(provider.authenticate_document_observation(observation))
        _require(provider.validate_document_target_observation(target, observation))
        _decode(observation, target)
        source = agent.interpretations.add_source('Authenticated document prediction before action',
            modality='document-transition-prediction', provider=provider.name, payload=observation,
            metadata={'action': action, 'model_id': model.id})
        expected = deepcopy(source)
        prediction = model.predict(deepcopy(observation), deepcopy(action))
        _validate_batch(agent, provider, registered[1])
        _require(provider.validate_document_target(target_token))
        _require(provider.authenticate_document_observation(observation))
        if not _same(agent.interpretations.get_source(source.id), expected):
            raise ValueError('prediction evidence changed')
        result = RetainedDocumentPrediction('document-prediction:' + uuid4().hex, source.id, prediction, action)
        cached, returned = deepcopy((result, expected)), deepcopy(result)
        _current_prediction(agent, provider, model, prediction, model_snapshot)
        _registry(agent, '_document_predictions')[result.id] = (provider, model, cached)
        return returned
    except Exception as error:
        return Unknown('document_prediction_unavailable', f'{type(error).__name__}: {error}')


def observe_document_transition(agent, provider, model, retained_prediction, attempt_id):
    """Use an authenticated paired outcome to suspend a contradicted learned rule."""
    reserved = False
    try:
        cached = _registry(agent, '_document_predictions').get(retained_prediction.id)
        if (cached is None or cached[0] is not provider or cached[1] is not model
                or not _same(cached[2][0], retained_prediction)):
            raise ValueError('unrecognized retained document prediction')
        if isinstance(retained_prediction.prediction, Unknown):
            raise ValueError('abstention is not a learned outcome prediction')
        states = _registry(agent, '_document_prediction_feedback')
        if retained_prediction.id in states:
            raise ValueError('prediction feedback already consumed or in progress')
        states[retained_prediction.id] = 'observing'
        reserved = True
        registered = _model_contract(agent, provider, model)
        model_snapshot = model.snapshot()
        _current_prediction(agent, provider, model, retained_prediction.prediction, model_snapshot)
        _validate_batch(agent, provider, registered[1])
        original = cached[2][1]
        if not _same(agent.interpretations.get_source(original.id), original):
            raise ValueError('retained prediction source changed')
        batch = extract_transitions(agent.interpretations.sources(), provider='plugin:' + provider.name)
        rows = [row for row in batch.transitions if row.attempt_id == attempt_id]
        if len(rows) != 1 or not _same(rows[0].action, retained_prediction.action):
            raise ValueError('feedback requires one paired applied matching action')
        row = rows[0]
        snapshots = tuple(agent.interpretations.get_source(sid) for sid in row.source_ids)
        target = _authenticate(provider, row)
        _require(provider.authenticate_document_observation(original.payload))
        _require(provider.validate_document_target_observation(target, original.payload))
        if (not _same(_target(original.payload, row.action), target)
                or not _same(_features(original.payload, row.action), _features(row.before, row.action))):
            raise ValueError('actual before-context differs from retained prediction context')
        outcome = _outcome(row.before, row.action, row.after)
        evidence = agent.interpretations.add_source('Authenticated document transition feedback',
            modality='document-transition-feedback', provider=provider.name,
            payload={'prediction': retained_prediction, 'transition': row, 'outcome': outcome},
            metadata={'prediction_source_id': original.id, 'source_ids': row.source_ids})
        expected = deepcopy(evidence)
        _validate_batch(agent, provider, registered[1])
        if any(not _same(agent.interpretations.get_source(source.id), source)
               for source in (*snapshots, original, expected)):
            raise ValueError('feedback evidence changed during retention')
        _current_prediction(agent, provider, model, retained_prediction.prediction, model_snapshot)
        event = model.observe_outcome(retained_prediction.prediction, outcome,
            source_ids=(*row.source_ids, evidence.id), reason='authenticated observed document target outcome')
        states[retained_prediction.id] = 'consumed'
        return event
    except Exception as error:
        if reserved:
            _registry(agent, '_document_prediction_feedback')[retained_prediction.id] = 'failed'
        return Unknown('document_transition_feedback_unavailable', f'{type(error).__name__}: {error}')
