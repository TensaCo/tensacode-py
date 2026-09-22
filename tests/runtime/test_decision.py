"""Ordinary application composition of public operations and authored policies."""
from tensorcode.ops import text as text_ops


class Model:
    def __init__(self, structured):
        self.structured = structured
        self.requests = []

    def complete(self, request):
        self.requests.append(request)
        return text_ops.ModelOutput(structured=self.structured)


def test_composition_keeps_distribution_and_explicit_instructions():
    model = Model({'label': 'billing',
                   'distribution': {'billing': .75, 'technical': .25},
                   'abstained': False})
    encode = text_ops.TextEncoder()
    decide = text_ops.Classify.from_model(model, labels=('billing', 'technical'),
                                         instructions='Route this support request')
    result = decide(encode('I was charged twice'))
    assert result.value == 'billing'
    assert result.distribution == {'billing': .75, 'technical': .25}
    assert model.requests[0].instructions == 'Route this support request'


def test_composition_applies_explicit_authored_selection_policy():
    model = Model({'label': 'billing',
                   'distribution': {'billing': .51, 'technical': .49},
                   'abstained': False})
    seen = []
    def require_margin(result):
        seen.append(result)
        return text_ops.ClassificationResult(label=None,
            distribution=result.distribution, abstained=True)
    encode = text_ops.TextEncoder()
    decide = text_ops.Classify.from_model(model, labels=('billing', 'technical'))
    result = require_margin(decide(encode('ambiguous')))
    assert result.abstained and result.value is None
    assert result.distribution == {'billing': .51, 'technical': .49}
    assert seen[0].value == 'billing'


def test_composition_passes_explicit_context_to_decision_operation():
    calls = []
    def encode(value):
        calls.append(('encode', value))
        return value.upper()
    def decide(value, *, context=None):
        calls.append(('decide', value, context))
        return 'chosen'
    result = decide(encode('input'), context={'x': 1})
    assert result == 'chosen'
    assert calls == [('encode', 'input'), ('decide', 'INPUT', {'x': 1})]
