"""Explicit measurement evidence and separately selected arithmetic calculations.

No language extraction, rate pairing, owner-wide aggregation, or property
classification occurs here. Every measurement identity, operand, operation,
output context and calculation selection is supplied explicitly.
"""
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from uuid import uuid4

from ..derivations import admit_operator, derive, export_derivation
from ..learning.experience import _same
from ..outcomes import Receipt, Unknown
from ..quantity import Quantity
from ..quantity_calculations import (CalculationContext, OPERATIONS, calculation_operator,
                                     result_predicate)
from ..records import Evidence, Proposition, Ref, Store, Var
from .plugin import Call, Capability, Informs, Param, Plugin

@dataclass(frozen=True)
class CalculationRegistration:
    reference: Ref
    operation: str
    operand_ids: tuple[str, ...]
    context: CalculationContext
    params: dict
    basis: tuple[str, ...]


@dataclass(frozen=True)
class CalculationSelection:
    reference: Ref
    context: CalculationContext
    reason: str
    revision: int


def _registered_operator(registration, premises, parameters):
    expected = {'context': registration.context, 'params': registration.params,
                'ordered_operand_ids': registration.operand_ids}
    if not _same(parameters, expected):
        return Unknown('calculation_registration_changed')
    return calculation_operator(registration.operation, premises, parameters)


class QuantityPlugin(Plugin):
    def __init__(self, name='quantity'):
        super().__init__(name=name)
        self.mind = Store()
        self._operators = {}
        self._operator_names = {}
        self._registrations = {}
        self._selected = {}
        self._selection_history = []
        self._selection_revision = 0
        self._worked_out = {}

    def remember(self, owner, predicate, quantity, *, measurement, evidence, kind=None):
        if (type(owner) is not Ref or type(predicate) is not str or not predicate
                or type(quantity) is not Quantity or type(measurement) is not Ref
                or type(evidence) is not Evidence or kind is not None and type(kind) is not Ref):
            raise TypeError('explicit measurement identity, owner, predicate, Quantity and Evidence required')
        roles = {'subject': owner, 'object': quantity, 'measurement': measurement, 'predicate': predicate}
        if kind is not None: roles['kind'] = kind
        proposition = Proposition('quantity_measurement', roles)
        self.mind.assert_(deepcopy(proposition), deepcopy(evidence))
        return deepcopy(proposition)

    @property
    def registrations(self):
        return deepcopy(tuple(self._registrations.values()))

    @property
    def selection_history(self):
        return deepcopy(tuple(self._selection_history))

    def register_calculation(self, operator, ordered_operand_ids, *, context, params=None, basis):
        operands = tuple(ordered_operand_ids)
        if (operator not in OPERATIONS or type(context) is not CalculationContext
                or (not operands and operator != 'count_selected') or any(type(x) is not str or not x for x in operands)
                or len(set(operands)) != len(operands)
                or type(basis) is not tuple or not basis
                or any(type(x) is not str or not x.strip() for x in basis)):
            raise ValueError('explicit operation, distinct ordered operands, context and basis required')
        context.__post_init__()
        params = {} if params is None else params
        if type(params) is not dict or set(params) != ({'unit'} if operator == 'convert' else {'factor'} if operator == 'scale' else set()):
            raise ValueError('operation parameters must be explicit and exact')
        reference = Ref('calculation:' + uuid4().hex)
        registration = CalculationRegistration(reference, operator, operands, context, params, basis)
        self._registrations[reference] = deepcopy(registration)
        return reference

    def select_calculation(self, reference, *, reason):
        if type(reason) is not str or not reason.strip():
            raise ValueError('calculation selection requires an explicit reason')
        registration = self._registrations.get(reference)
        if registration is None: return Unknown('unknown_calculation')
        name = self._operator_names.setdefault(registration.context, 'quantity:selected:' + uuid4().hex)
        handle = admit_operator(self.mind, name,
            partial(_registered_operator, deepcopy(registration)), reason=reason)
        if isinstance(handle, Unknown): return handle
        self._operators[registration.context] = handle
        self._selection_revision += 1
        selection = CalculationSelection(reference, registration.context, reason, self._selection_revision)
        self._selected[registration.context] = selection
        self._selection_history.append(selection)
        return True

    def calculate(self, reference):
        registration = self._registrations.get(reference)
        if registration is None: return Unknown('unknown_calculation')
        selection = self._selected.get(registration.context)
        if selection is None or selection.reference != reference:
            return Unknown('calculation_not_selected')
        records = {r.id: r for r in self.mind.propositions()}
        if any(identifier not in records for identifier in registration.operand_ids):
            return Unknown('missing_calculation_operand')
        inspected = tuple(identifier for identifier, record in records.items()
                          if record.proposition.predicate == 'quantity_measurement'
                          or identifier in registration.operand_ids)
        return derive(self.mind, self._operators[registration.context], inspected,
            params={'context': registration.context, 'params': registration.params,
                    'ordered_operand_ids': registration.operand_ids}, basis=registration.basis,
            population_predicates=('quantity_measurement',),
            derived_premise_ids=tuple(identifier for identifier in registration.operand_ids
                if records[identifier].proposition.predicate != 'quantity_measurement'))

    @staticmethod
    def _capability(registration):
        context, operation = registration.context, registration.operation
        typed = context.kind is not None
        name = 'calculate_' + operation + ('_kind_' if typed else '_owner_') + context.predicate
        roles = {'subject': Var('owner'), 'object': Var('answer')}
        params = (Param('owner', 'thing'),)
        if typed:
            roles['kind'] = Var('kind')
            params += (Param('kind', 'kind'),)
        predicate = result_predicate(operation, context)
        return Capability(name, params, informs=(Informs(predicate, 'explicit_context', 'owner',
            query=Proposition(predicate, roles)),), effect_kind='read',
            description='Run an explicitly selected calculation over its exact ordered operands')

    def capabilities(self):
        caps = {}
        for selection in self._selected.values():
            registration = self._registrations[selection.reference]
            capability = self._capability(registration)
            caps[capability.name] = capability
        return tuple(caps.values())

    def _chosen(self, action):
        if type(action) is not Call or action.plugin != self.name: return None
        matches = []
        for selection in self._selected.values():
            registration = self._registrations[selection.reference]
            cap = self._capability(registration)
            args = (('owner', registration.context.owner),)
            if registration.context.kind is not None: args += (('kind', registration.context.kind),)
            if cap.name == action.capability and len(action.args) == len(args) and _same(dict(action.args), dict(args)):
                matches.append((selection, registration))
        return matches[0] if len(matches) == 1 else None

    def execute(self, action, *, key=None):
        chosen = self._chosen(action)
        if chosen is None: return Receipt(action, 'rejected', error='No exact explicitly selected calculation')
        selection, registration = chosen
        result = self.calculate(registration.reference)
        if isinstance(result, Unknown): return Receipt(action, 'rejected', error=result.detail or result.reason)
        reference = export_derivation(self.mind, result)
        if isinstance(reference, Unknown): return Receipt(action, 'rejected', error=reference.detail or reference.reason)
        if self._chosen(action) != chosen:
            return Receipt(action, 'rejected', error='Calculation selection changed')
        receipt = Receipt(action, 'applied', idempotency_key=key)
        self._worked_out[action.capability] = deepcopy((action, receipt, selection, result, reference))
        return receipt

    def reveal(self, cap, args, receipt):
        saved = self._worked_out.get(cap.name)
        if saved is None: return
        action, actual, selection, result, reference = saved
        if not _same(receipt, actual) or not _same(dict(action.args), dict(args)): return
        chosen = self._chosen(action)
        if chosen is None or not _same(chosen[0], selection): return
        current = export_derivation(self.mind, result)
        if isinstance(current, Unknown) or not _same(current, reference): return
        if self._chosen(action) is None or not _same(self._chosen(action)[0], selection): return
        yield current

    def display(self, value):
        return str(value) if isinstance(value, Quantity) else None
