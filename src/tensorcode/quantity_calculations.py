"""Explicitly selected arithmetic over authenticated, ordered proposition operands.

Operations and output contexts are supplied policies. Units check arithmetic;
no unit, owner, spelling, or role ordering selects an operation or an operand.
"""
from dataclasses import dataclass
import math

from .learning.experience import _same
from .outcomes import Unknown
from .quantity import Quantity, Unit, add, sub, mul, div, scale, ratio, percent_of, compare, convert
from .records import Proposition, Ref

OPERATIONS = ('sum', 'sub', 'mul', 'div', 'scale', 'ratio', 'percent_of', 'compare', 'convert', 'count_selected')


@dataclass(frozen=True)
class CalculationContext:
    owner: Ref
    predicate: str
    kind: Ref | None = None

    def __post_init__(self):
        if (type(self.owner) is not Ref or type(self.predicate) is not str or not self.predicate
                or self.kind is not None and type(self.kind) is not Ref):
            raise ValueError('calculation context requires explicit owner, predicate, and optional kind')


def result_predicate(operation, context):
    if operation not in OPERATIONS or type(context) is not CalculationContext:
        raise ValueError('explicit supported operation and context required')
    return 'calculated:' + operation + ':' + context.predicate


def _valid_quantity(value):
    return (type(value) is Quantity and type(value.unit) is Unit
            and math.isfinite(value.value) and math.isfinite(value.unit.factor()))


def calculation_operator(operation, premises, parameters):
    """Replay one admitted arithmetic operation, with no access to any Store."""
    if (operation not in OPERATIONS or type(parameters) is not dict
            or set(parameters) != {'context', 'params', 'ordered_operand_ids'}
            or type(parameters['context']) is not CalculationContext
            or type(parameters['params']) is not dict):
        return Unknown('invalid_calculation_parameters')
    context, params = parameters['context'], parameters['params']
    context.__post_init__()
    expected = {'unit'} if operation == 'convert' else {'factor'} if operation == 'scale' else set()
    if set(params) != expected:
        return Unknown('invalid_calculation_parameters')
    if any(type(p) is not Proposition for p in premises):
        return Unknown('explicit_proposition_operands_required')
    ids = parameters['ordered_operand_ids']
    if type(ids) is not tuple or len(set(ids)) != len(ids) or any(type(i) is not str for i in ids):
        return Unknown('invalid_ordered_operands')
    by_id = {p.id: p for p in premises}
    if any(i not in by_id for i in ids): return Unknown('missing_calculation_operand')
    selected = tuple(by_id[i] for i in ids)
    for selected_p in selected:
        measurement = selected_p.role('measurement')
        if type(measurement) is Ref:
            if any(p.role('measurement') == measurement and not _same(p, selected_p) for p in premises):
                return Unknown('conflicting_measurement_identity')
    if not selected and operation != 'count_selected': return Unknown('bad_arity')
    if operation == 'count_selected':
        if any(p.predicate != 'quantity_measurement' or type(p.role('measurement')) is not Ref for p in selected):
            return Unknown('explicit_measurement_operands_required')
        result = Quantity(len(selected), Unit.of('record'))
    else:
        values, measurements = [], []
        for p in selected:
            measurement, value = p.role('measurement'), p.role('object')
            # The caller requires authenticated derived branches for all other
            # operands. Predicate spelling never establishes that authority.
            is_measurement = p.predicate == 'quantity_measurement'
            if is_measurement and type(measurement) is not Ref:
                return Unknown('explicit_measurement_identity_required')
            expected_roles = {'subject', 'object'} | ({'measurement', 'predicate'} if is_measurement else set())
            if 'kind' in p.roles: expected_roles.add('kind')
            if ((is_measurement and (p.predicate != 'quantity_measurement' or type(p.role('predicate')) is not str or not p.role('predicate')))
                    or not _valid_quantity(value) or type(p.role('subject')) is not Ref
                    or ('kind' in p.roles and type(p.role('kind')) is not Ref)
                    or set(p.roles) != expected_roles
                    or not _same(p, Proposition(p.predicate, p.roles))):
                return Unknown('qualified_or_unidentified_measurement')
            if is_measurement:
                if measurement in measurements:
                    return Unknown('repeated_measurement_operand')
                measurements.append(measurement)
            values.append(value)
        if operation == 'sum':
            result = values[0]
            for value in values[1:]:
                result = add(result, value)
                if isinstance(result, Unknown): return result
        elif operation in ('sub', 'mul', 'div', 'ratio', 'percent_of', 'compare'):
            if len(values) != 2: return Unknown('bad_arity')
            result = {'sub': sub, 'mul': mul, 'div': div, 'ratio': ratio, 'percent_of': percent_of, 'compare': compare}[operation](*values)
        elif operation == 'scale':
            if len(values) != 1 or type(params['factor']) not in (int, float) or not math.isfinite(params['factor']):
                return Unknown('bad_scale_parameter')
            result = scale(values[0], params['factor'])
        else:
            if len(values) != 1 or type(params['unit']) is not Unit: return Unknown('bad_arity')
            result = convert(values[0], params['unit'])
        if isinstance(result, Unknown): return result
        if type(result) is Quantity and not _valid_quantity(result):
            return Unknown('nonfinite_calculation')
    roles = {'subject': context.owner, 'object': result}
    if context.kind is not None: roles['kind'] = context.kind
    return Proposition(result_predicate(operation, context), roles)

