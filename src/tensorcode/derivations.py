"""Explicitly admitted, replayable proposition operators.

Operators are supplied procedures, not learned reasoning. They receive detached
premise propositions and parameters only. A receipt authenticates its operator,
premise evidence, optional inspected population, and exact replayed output.
"""
from copy import deepcopy
from dataclasses import dataclass, replace, fields, is_dataclass
from datetime import datetime, timezone
from uuid import uuid4
from weakref import WeakValueDictionary

from .records import Proposition, Evidence, Ref, Interval
from .outcomes import Unknown, Score
from .learning.experience import _same


@dataclass(frozen=True)
class OperatorHandle:
    id: str
    name: str
    version: int


@dataclass(frozen=True)
class DerivationReceipt:
    id: str
    record_id: str
    premise_ids: tuple[str, ...]
    operator: OperatorHandle
    proposition: Proposition
    evidence: Evidence
    basis: tuple[str, ...]


@dataclass
class _State:
    operators: dict
    versions: dict
    receipts: dict
    epoch: int = 0
    executing: bool = False
    bridges: dict = None
    store_id: str = ""
    history: tuple = ()


def _state(store):
    if not hasattr(store, '_derivation_authority'):
        store._derivation_authority = _State({}, {}, {}, bridges={}, store_id="store:" + uuid4().hex)
    return store._derivation_authority


def admit_operator(store, name, operator, *, reason):
    """Explicitly admit a supplied operator version; replace earlier same-name authority."""
    try:
        if type(name) is not str or not name.strip() or not callable(operator):
            raise ValueError('operator needs a name and callable implementation')
        if type(reason) is not str or not reason.strip():
            raise ValueError('operator admission requires an explicit reason')
        state = _state(store)
        version = state.versions.get(name, 0) + 1
        handle = OperatorHandle('operator:' + uuid4().hex, name, version)
        for key, (old, function, active) in tuple(state.operators.items()):
            if old.name == name:
                state.operators[key] = old, function, False
        state.operators[handle.id] = handle, operator, True
        state.versions[name] = version
        state.history += (OperatorEvent(handle, 'admitted', reason),)
        state.epoch += 1
        return handle
    except Exception as error:
        return Unknown('operator_admission_unavailable', str(error))


def withdraw_operator(store, handle, *, reason):
    try:
        if type(reason) is not str or not reason.strip():
            raise ValueError('withdrawal requires an explicit reason')
        state = _state(store)
        saved = state.operators.get(handle.id)
        if saved is None or not _same(saved[0], handle):
            raise ValueError('unrecognized operator handle')
        state.operators[handle.id] = saved[0], saved[1], False
        state.history += (OperatorEvent(handle, 'withdrawn', reason),)
        state.epoch += 1
        return True
    except Exception as error:
        return Unknown('operator_withdrawal_unavailable', str(error))


def _operator(state, handle):
    saved = state.operators.get(handle.id)
    if saved is None or not saved[2] or not _same(saved[0], handle):
        raise ValueError('operator version is withdrawn or unrecognized')
    return saved[1]


def _evidence(evidence):
    return (type(evidence) is Evidence and type(evidence.source) is Ref
        and type(evidence.observed_at) is datetime
        and (evidence.locator is None or type(evidence.locator) is str)
        and (evidence.method is None or type(evidence.method) is str)
        and (evidence.confidence is None or type(evidence.confidence) is Score)
        and type(evidence.derived_from) is tuple
        and all(type(value) is str and value for value in evidence.derived_from))


def _active(store):
    # Store.propositions supplies live values; detach before any operator runs.
    return {record.id: deepcopy(record) for record in store.propositions()}


def _conflict(proposition, records):
    if type(proposition.valid) is not Interval:
        raise ValueError('invalid proposition interval')
    for other in records.values():
        opposite = other.proposition
        if (type(opposite.valid) is Interval
                and _same(replace(proposition, polarity=not proposition.polarity, valid=opposite.valid), opposite)
                and proposition.valid.overlap(opposite.valid) is not None):
            return True
    return False


@dataclass(frozen=True)
class OperatorEvent:
    handle: OperatorHandle
    status: str
    reason: str


def operator_history(store):
    return deepcopy(_state(store).history)


def _passive_same(left, right):
    """Compare retained transparent values without invoking arbitrary equality hooks."""
    if type(left) is not type(right):
        return False
    if left is None or type(left) in (bool, int, float, str, bytes, datetime):
        return left == right
    if is_dataclass(left):
        return all(_passive_same(object.__getattribute__(left, f.name), object.__getattribute__(right, f.name))
                   for f in fields(type(left)))
    if type(left) is dict:
        return len(left) == len(right) and all(any(_passive_same(k, other) and _passive_same(v, value)
            for other, value in right.items()) for k, v in left.items())
    if type(left) in (tuple, list):
        return len(left) == len(right) and all(_passive_same(a, b) for a, b in zip(left, right))
    if type(left) in (set, frozenset):
        return len(left) == len(right) and all(any(_passive_same(a, b) for b in right) for a in left)
    return False


def _touch(readset, store, records=None):
    state = _state(store)
    snapshot = _active(store) if records is None else records
    current = readset.get(id(store))
    if current is None:
        readset[id(store)] = (store, state, state.epoch, deepcopy(snapshot))
    elif current[1] is not state or current[2] != state.epoch or not _passive_same(current[3], snapshot):
        raise ValueError('a visited source store changed during traversal')


def _check_readset(readset):
    # Read backing data directly: no Store methods, operators, or deepcopies here.
    for store, state, epoch, snapshot in readset.values():
        if store._derivation_authority is not state or state.epoch != epoch:
            raise ValueError('visited derivation authority changed during traversal')
        active = {identifier: record for identifier, record in store._props.items() if record.retracted is None}
        if not _passive_same(active, snapshot):
            raise ValueError('visited supporting store changed during traversal')


def _validate(store, identifier, records, state, remaining, path, max_depth, readset, *, branch=None, require_derived=False):
    if len(path) >= max_depth or remaining[0] <= 0:
        raise ValueError('derivation validation budget exhausted')
    remaining[0] -= 1
    identity = (state.store_id, identifier)
    if identity in path:
        raise ValueError('cyclic derivation support')
    record = records.get(identifier)
    if record is None or record.retracted is not None:
        raise ValueError('missing or withdrawn premise')
    if _conflict(record.proposition, records):
        raise ValueError('opposite overlapping proposition remains unresolved')
    if type(record.evidence) not in (tuple, list) or not record.evidence:
        raise ValueError('record lacks observation evidence')
    errors = []
    for evidence in ([branch] if branch is not None else record.evidence):
        if remaining[0] <= 0:
            raise ValueError('derivation branch validation budget exhausted')
        remaining[0] -= 1
        if not _evidence(evidence):
            errors.append('malformed evidence branch')
            continue
        if not evidence.derived_from and evidence.locator not in state.receipts and evidence.locator not in state.bridges:
            if evidence.source.id in state.operators or evidence.method == 'authenticated-derivation-import':
                errors.append('derived provenance cannot be relabeled as direct observation')
                continue
            if require_derived:
                errors.append('this premise requires authenticated derived support')
                continue
            return True
        try:
            bridge = state.bridges.get(evidence.locator)
            if bridge is not None:
                reference, expected_evidence = bridge
                if not _same(evidence, expected_evidence) or not _same(record.proposition, reference.proposition):
                    raise ValueError('imported derivation evidence changed')
                _validate_reference(reference, remaining, (*path, identity), max_depth, readset)
                return True
            saved = state.receipts.get(evidence.locator)
            if saved is None:
                raise ValueError('premise IDs alone do not authenticate a derivation')
            receipt, premises, params, population_predicates, population, derived_premise_ids = saved
            if (receipt.record_id != identifier or not _same(receipt.proposition, record.proposition)
                    or not _same(receipt.evidence, evidence) or receipt.premise_ids != evidence.derived_from):
                raise ValueError('derivation receipt or output changed')
            function = _operator(state, receipt.operator)
            if population_predicates:
                current_population = tuple(r for r in records.values() if r.proposition.predicate in population_predicates)
                if not _same(current_population, population):
                    raise ValueError('inspected population changed')
            for premise in premises:
                if not _same(records.get(premise.id), premise):
                    raise ValueError('premise content or evidence changed')
                _validate(store, premise.id, records, state, remaining, (*path, identity), max_depth, readset,
                          require_derived=premise.id in derived_premise_ids)
            epoch = state.epoch
            output = function(tuple(deepcopy(p.proposition) for p in premises), deepcopy(params))
            if state.epoch != epoch or not _same(_active(store), records):
                raise ValueError('authority or store changed during derivation replay')
            if type(output) is not Proposition or not _same(output, receipt.proposition):
                raise ValueError('operator replay differs from retained output')
            return True
        except Exception as error:
            errors.append(str(error))
    raise ValueError('; '.join(errors) or 'no valid evidence branch')


def validate_record_support(store, record_id, *, max_depth=32, max_nodes=256):
    state = _state(store)
    if state.executing:
        return Unknown('derivation_reentrant', 'operator validation is already active')
    try:
        if type(max_depth) is not int or max_depth < 1 or type(max_nodes) is not int or max_nodes < 1:
            raise ValueError('positive finite validation budgets required')
        state.executing = True
        epoch = state.epoch
        records = _active(store)
        readset = {}
        _touch(readset, store, records)
        _validate(store, record_id, records, state, [max_nodes], (), max_depth, readset)
        if state.epoch != epoch or not _same(_active(store), records):
            raise ValueError('support changed during validation')
        _check_readset(readset)
        return True
    except Exception as error:
        return Unknown('derivation_support_unavailable', str(error))
    finally:
        state.executing = False


def derive(store, handle, premise_ids, *, params=None, basis, population_predicates=(), derived_premise_ids=(),
           max_depth=32, max_nodes=256):
    state = _state(store)
    if state.executing:
        return Unknown('derivation_reentrant', 'operator execution is already active')
    try:
        if type(basis) is not tuple or not basis or any(type(x) is not str or not x.strip() for x in basis):
            raise ValueError('derivation requires an explicit basis')
        premise_ids = tuple(premise_ids)
        if any(type(x) is not str for x in premise_ids) or len(set(premise_ids)) != len(premise_ids):
            raise ValueError('explicit distinct premise IDs required')
        if (type(population_predicates) is not tuple
                or any(type(name) is not str or not name for name in population_predicates)
                or len(set(population_predicates)) != len(population_predicates)):
            raise ValueError('population predicates must be an explicit tuple of distinct nonempty names')
        population_predicates = tuple(sorted(population_predicates))
        if (type(derived_premise_ids) is not tuple
                or any(type(identifier) is not str for identifier in derived_premise_ids)
                or len(set(derived_premise_ids)) != len(derived_premise_ids)
                or not set(derived_premise_ids) <= set(premise_ids)):
            raise ValueError('derived premise IDs must be an explicit distinct tuple subset of premise IDs')
        if type(max_depth) is not int or max_depth < 1 or type(max_nodes) is not int or max_nodes < 1:
            raise ValueError('positive finite validation budgets required')
        state.executing = True
        epoch = state.epoch
        function = _operator(state, handle)
        records = _active(store)
        readset = {}
        _touch(readset, store, records)
        remaining = [max_nodes]
        for identifier in premise_ids:
            _validate(store, identifier, records, state, remaining, (), max_depth, readset,
                      require_derived=identifier in derived_premise_ids)
        premises = tuple(deepcopy(records[x]) for x in premise_ids)
        parameters = deepcopy(params)
        population = tuple(r for r in records.values() if r.proposition.predicate in population_predicates)
        arguments = (tuple(deepcopy(p.proposition) for p in premises), deepcopy(parameters))
        if state.epoch != epoch or not _same(_active(store), records):
            raise ValueError('support changed before operator invocation')
        output = function(*arguments)
        if type(output) is not Proposition:
            raise ValueError('operator must return one explicit proposition')
        output = deepcopy(output)
        if output.predicate in population_predicates:
            raise ValueError('population-dependent output must use a distinct predicate')
        if output.id in premise_ids or _conflict(output, records):
            raise ValueError('cyclic or contradicted derivation output')
        if state.epoch != epoch or not _same(_active(store), records):
            raise ValueError('operator changed store or admission authority')
        identifier = 'derivation:' + uuid4().hex
        evidence = Evidence(Ref(handle.id), datetime.now(timezone.utc), locator=identifier,
                            method=handle.name, derived_from=premise_ids)
        receipt = DerivationReceipt(identifier, output.id, premise_ids, handle, output, evidence, basis)
        cached, returned = deepcopy((receipt, premises, parameters, population_predicates, population, derived_premise_ids)), deepcopy(receipt)
        # Repeat replay before publication; mutable or nondeterministic operators cannot mint authority.
        replay = function(tuple(deepcopy(p.proposition) for p in premises), deepcopy(parameters))
        if not _same(replay, output) or state.epoch != epoch or not _same(_active(store), records):
            raise ValueError('operator replay or supporting state changed')
        _check_readset(readset)
        store.assert_(deepcopy(output), deepcopy(evidence))
        current = _active(store)
        expected = deepcopy(records)
        from .records import PropositionRecord
        if output.id in expected:
            expected[output.id].evidence.append(deepcopy(evidence))
        else:
            expected[output.id] = PropositionRecord(deepcopy(output), [deepcopy(evidence)])
        if state.epoch != epoch or not _same(current, expected):
            raise ValueError('store publication changed derivation evidence')
        # Only this publication is allowed to change the target snapshot.
        readset[id(store)] = (store, state, epoch, expected)
        _check_readset(readset)
        state.receipts[identifier] = cached
        state.epoch += 1
        return returned
    except Exception as error:
        return Unknown('derivation_unavailable', f'{type(error).__name__}: {error}')
    finally:
        state.executing = False


@dataclass(frozen=True)
class DerivationReference:
    source_store_id: str
    receipt_id: str
    proposition: Proposition


_STORES = WeakValueDictionary()


def _validate_reference(reference, remaining, path, max_depth, readset):
    if type(reference) is not DerivationReference:
        raise ValueError('an authentic derivation reference is required')
    source = _STORES.get(reference.source_store_id)
    if source is None:
        raise ValueError('source store is no longer available')
    state = _state(source)
    saved = state.receipts.get(reference.receipt_id)
    if saved is None or not _same(saved[0].proposition, reference.proposition):
        raise ValueError('derivation reference was not issued by its source')
    if state.executing:
        raise ValueError('cyclic or reentrant source derivation validation')
    state.executing = True
    try:
        records, epoch = _active(source), state.epoch
        _touch(readset, source, records)
        receipt = saved[0]
        record = records.get(receipt.record_id)
        if record is None or not any(_same(e, receipt.evidence) for e in record.evidence):
            raise ValueError('source derivation receipt evidence is missing')
        # Authenticate this receipt, not an unrelated direct support branch.
        _validate(source, receipt.record_id, records, state, remaining, path, max_depth, readset, branch=receipt.evidence)
        if state.epoch != epoch or not _same(_active(source), records):
            raise ValueError('source derivation changed during validation')
    finally:
        state.executing = False
    return source


def export_derivation(store, receipt, *, max_depth=32, max_nodes=256):
    try:
        if type(max_depth) is not int or max_depth < 1 or type(max_nodes) is not int or max_nodes < 1:
            raise ValueError('positive finite validation budgets required')
        state = _state(store)
        saved = state.receipts.get(receipt.id)
        if saved is None or not _same(saved[0], receipt):
            raise ValueError('unrecognized or changed derivation receipt')
        _STORES[state.store_id] = store
        reference = DerivationReference(state.store_id, receipt.id, deepcopy(receipt.proposition))
        readset = {}
        _validate_reference(reference, [max_nodes], (), max_depth, readset)
        returned = deepcopy(reference)
        _check_readset(readset)
        return returned
    except Exception as error:
        return Unknown('derivation_export_unavailable', str(error))


def import_derivation(store, reference, *, max_depth=32, max_nodes=256):
    state = _state(store)
    if state.executing:
        return Unknown('derivation_reentrant', 'import while derivation is active')
    try:
        if type(max_depth) is not int or max_depth < 1 or type(max_nodes) is not int or max_nodes < 1:
            raise ValueError('positive finite validation budgets required')
        state.executing = True
        reference = deepcopy(reference)
        epoch, records = state.epoch, _active(store)
        readset = {}
        _touch(readset, store, records)
        _validate_reference(reference, [max_nodes], (), max_depth, readset)
        if _conflict(reference.proposition, records):
            raise ValueError('imported proposition conflicts with target evidence')
        identifier = 'derivation-import:' + uuid4().hex
        evidence = Evidence(Ref(reference.source_store_id), datetime.now(timezone.utc), locator=identifier,
            method='authenticated-derivation-import', derived_from=(reference.proposition.id,))
        cached = deepcopy((reference, evidence))
        if state.epoch != epoch or not _same(_active(store), records):
            raise ValueError('target store changed during import')
        _check_readset(readset)
        record = store.assert_(deepcopy(reference.proposition), deepcopy(evidence))
        expected = deepcopy(records)
        from .records import PropositionRecord
        if reference.proposition.id in expected:
            expected[reference.proposition.id].evidence.append(deepcopy(evidence))
        else:
            expected[reference.proposition.id] = PropositionRecord(deepcopy(reference.proposition), [deepcopy(evidence)])
        readset[id(store)] = (store, state, epoch, expected)
        _validate_reference(reference, [max_nodes], (), max_depth, readset)
        if state.epoch != epoch or not _same(_active(store), expected):
            raise ValueError('derived import changed during publication')
        returned = deepcopy(record)
        _check_readset(readset)
        state.bridges[identifier] = cached
        state.epoch += 1
        return returned
    except Exception as error:
        return Unknown('derivation_import_unavailable', str(error))
    finally:
        state.executing = False


def validate_derivation_reference(reference, *, max_depth=32, max_nodes=256):
    try:
        if type(max_depth) is not int or max_depth < 1 or type(max_nodes) is not int or max_nodes < 1:
            raise ValueError('positive finite validation budgets required')
        readset = {}
        _validate_reference(reference, [max_nodes], (), max_depth, readset)
        _check_readset(readset)
        return True
    except Exception as error:
        return Unknown('derivation_reference_unavailable', str(error))


def validate_record_supports(store, record_ids, *, max_depth=32, max_nodes=256):
    """Validate all answer roots under one bounded, traversal-wide read set."""
    state = _state(store)
    if state.executing:
        return Unknown('derivation_reentrant', 'operator validation is already active')
    try:
        if type(max_depth) is not int or max_depth < 1 or type(max_nodes) is not int or max_nodes < 1:
            raise ValueError('positive finite validation budgets required')
        state.executing = True
        records, readset, remaining = _active(store), {}, [max_nodes]
        _touch(readset, store, records)
        for identifier in tuple(record_ids):
            _validate(store, identifier, records, state, remaining, (), max_depth, readset)
        _check_readset(readset)
        return True
    except Exception as error:
        return Unknown('derivation_support_unavailable', str(error))
    finally:
        state.executing = False


def validate_derivation_references(references, *, max_depth=32, max_nodes=256):
    """Validate a response batch without allowing later replay to stale earlier roots."""
    try:
        if type(max_depth) is not int or max_depth < 1 or type(max_nodes) is not int or max_nodes < 1:
            raise ValueError('positive finite validation budgets required')
        readset, remaining = {}, [max_nodes]
        for reference in tuple(references):
            _validate_reference(reference, remaining, (), max_depth, readset)
        _check_readset(readset)
        return True
    except Exception as error:
        return Unknown('derivation_reference_unavailable', str(error))
