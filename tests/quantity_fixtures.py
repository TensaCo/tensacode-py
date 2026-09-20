"""Authored measurement evidence and explicit arithmetic choices for tests only."""
from datetime import datetime, timezone

from tensorcode.agent.quantity_plugin import QuantityPlugin
from tensorcode.quantity_calculations import CalculationContext
from tensorcode.records import Evidence, Ref


def measurement(plugin, identity, owner, predicate, quantity, *, kind=None):
    """The caller names the measurement occurrence and supplies its quantity."""
    assert type(identity) is Ref
    evidence = Evidence(Ref('fixture:quantity-observer'), datetime.now(timezone.utc),
                        method='explicit authored measurement', locator=identity.id)
    return plugin.remember(owner, predicate, quantity, measurement=identity, evidence=evidence, kind=kind)


def calculation(plugin, operator, operands, context, *, params=None, select=True):
    """No operand discovery: all ordered records and the operation are supplied."""
    reference = plugin.register_calculation(operator, tuple(record.id for record in operands),
        context=context, params=params, basis=('Explicit test operation, ordered operands and output context',))
    if select:
        assert plugin.select_calculation(reference, reason='Explicit fixture calculation selection') is True
    return reference


def conversion_definition(plugin, identity, source_unit, target_unit, factor, *, scope, valid):
    """The test explicitly supplies a conversion definition and its evidence."""
    assert type(identity) is Ref
    return plugin.remember_conversion(identity, source_unit, target_unit, factor,
        scope=scope, valid=valid, evidence=Evidence(Ref('fixture:conversion-author'),
            datetime.now(timezone.utc), method='explicit authored conversion definition', locator=identity.id))
