"""Owned objective identities must survive private backend file moves."""
import pytest
from tensorcode.ops.base import Operation
from tensorcode._internal.training.persistence import configuration, bindings, validate_bindings


class Objective(Operation):
    replayable = True

    def forward(self, value, *, context=None):
        return value

    def configuration(self):
        return {'role': 'objective', 'model': {'width': 4}}

    def _operation_identity(self):
        return 'tensorcode.ops.vec.decode.TextDecoder.objective'


def test_explicit_objective_identity_survives_private_class_move():
    class RelocatedObjective(Objective):
        pass
    source = Objective()
    restored = RelocatedObjective()
    assert configuration(source)['type'] == 'tensorcode.ops.vec.decode.TextDecoder.objective'
    validate_bindings(bindings({'objective': source}), {'objective': restored})
    restored.configuration = lambda: {'role': 'objective', 'model': {'width': 8}}
    with pytest.raises(ValueError, match='Incompatible'):
        validate_bindings(bindings({'objective': source}), {'objective': restored})


@pytest.mark.parametrize('identity', ['', None, 12])
def test_invalid_explicit_identity_fails_closed(identity):
    source = Objective()
    source._operation_identity = lambda: identity
    with pytest.raises(ValueError, match='identity'):
        configuration(source)


def test_operations_without_override_keep_concrete_type_identity():
    from tensorcode.ops.text.encode import TextEncoder
    assert configuration(TextEncoder())['type'] == 'tensorcode.ops.text.encode.TextEncoder'
