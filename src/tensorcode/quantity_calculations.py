"""Explicitly selected arithmetic over authenticated, ordered proposition operands.

Operations and output contexts are supplied policies. Units check arithmetic;
no unit, owner, spelling, or role ordering selects an operation or an operand.
"""
from dataclasses import dataclass
import math

from .learning.experience import _same
from .outcomes import Unknown
from .quantity import Quantity, Unit, add, sub, mul, div, scale, ratio, percent_of, compare
from .records import Proposition, Ref, Interval

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
            and math.isfinite(value.value))


def _convert_from_definition(value, definition, population, params):
    if (type(params['valid']) is not Interval
            or params['scope'] is not None and type(params['scope']) is not Ref):
        return Unknown('explicit_conversion_context_required')
    roles = {'definition', 'source_unit', 'target_unit', 'factor'}
    if (definition.predicate != 'quantity_conversion' or set(definition.roles) != roles
            or type(definition.role('definition')) is not Ref
            or type(definition.role('source_unit')) is not Unit
            or type(definition.role('target_unit')) is not Unit
            or type(definition.role('factor')) not in (int, float)
            or not math.isfinite(definition.role('factor')) or definition.role('factor') <= 0
            or definition.polarity is not True or definition.modality != 'asserted'
            or type(definition.valid) is not Interval
            or not _same(definition.scope, params['scope'])):
        return Unknown('unsupported_conversion_definition')
    if value.unit != definition.role('source_unit'):
        return Unknown('conversion_source_unit_mismatch')
    requested = params['valid']
    if not _same(definition.valid.overlap(requested), requested):
        return Unknown('conversion_validity_not_covered')
    for other in population:
        if other.predicate != 'quantity_conversion' or not _same(other.scope, definition.scope):
            continue
        same_identity = other.role('definition') == definition.role('definition')
        same_edge = (_same(other.role('source_unit'), definition.role('source_unit'))
                     and _same(other.role('target_unit'), definition.role('target_unit')))
        if not same_identity and not same_edge: continue
        if type(other.valid) is not Interval:
            return Unknown('invalid_conversion_rival')
        if other.valid.overlap(requested) is None: continue
        if (set(other.roles) != roles or not same_edge
                or not _same(other.role('factor'), definition.role('factor'))
                or other.polarity is not True or other.modality != 'asserted'):
            return Unknown('conflicting_conversion_definition')
    result = Quantity(value.value * definition.role('factor'), definition.role('target_unit'))
    return result if _valid_quantity(result) else Unknown('nonfinite_calculation')


def calculation_operator(operation, premises, parameters):
    """Replay one admitted arithmetic operation, with no access to any Store."""
    if (operation not in OPERATIONS or type(parameters) is not dict
            or set(parameters) != {'context', 'params', 'ordered_operand_ids'}
            or type(parameters['context']) is not CalculationContext
            or type(parameters['params']) is not dict):
        return Unknown('invalid_calculation_parameters')
    context, params = parameters['context'], parameters['params']
    context.__post_init__()
    expected = {'scope', 'valid'} if operation == 'convert' else {'factor'} if operation == 'scale' else set()
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
    if operation == 'convert' and len(selected) != 2: return Unknown('bad_arity')
    arithmetic_operands = selected[:1] if operation == 'convert' else selected
    if operation == 'count_selected':
        if any(p.predicate != 'quantity_measurement' or type(p.role('measurement')) is not Ref for p in selected):
            return Unknown('explicit_measurement_operands_required')
        result = Quantity(len(selected), Unit.of('record'))
    else:
        values, measurements = [], []
        for p in arithmetic_operands:
            measurement, value = p.role('measurement'), p.role('object')
            # The caller requires authenticated derived branches for all other
            # operands. Predicate spelling never establishes that authority.
            is_measurement = p.predicate == 'quantity_measurement'
            if is_measurement and type(measurement) is not Ref:
                return Unknown('explicit_measurement_identity_required')
            expected_roles = {'subject', 'object'} | ({'measurement', 'predicate'} if is_measurement else set())
            if 'kind' in p.roles: expected_roles.add('kind')
            qualified = operation == 'convert'
            if qualified and (type(params['valid']) is not Interval or type(p.valid) is not Interval
                    or not _same(p.scope, params['scope'])
                    or not _same(p.valid.overlap(params['valid']), params['valid'])):
                return Unknown('conversion_operand_context_mismatch')
            expected_proposition = Proposition(p.predicate, p.roles,
                scope=p.scope if qualified else None, valid=p.valid if qualified else Interval())
            if ((is_measurement and (p.predicate != 'quantity_measurement' or type(p.role('predicate')) is not str or not p.role('predicate')))
                    or not _valid_quantity(value) or type(p.role('subject')) is not Ref
                    or ('kind' in p.roles and type(p.role('kind')) is not Ref)
                    or set(p.roles) != expected_roles
                    or not _same(p, expected_proposition)):
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
            result = _convert_from_definition(values[0], selected[1], premises, params)
        if isinstance(result, Unknown): return result
        if type(result) is Quantity and not _valid_quantity(result):
            return Unknown('nonfinite_calculation')
    roles = {'subject': context.owner, 'object': result}
    if context.kind is not None: roles['kind'] = context.kind
    return Proposition(result_predicate(operation, context), roles,
        scope=params['scope'] if operation == 'convert' else None,
        valid=params['valid'] if operation == 'convert' else Interval())

