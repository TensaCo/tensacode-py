from dataclasses import dataclass
import pytest
import tensorcode as tc
from tensorcode.ops import Operation


@dataclass(frozen=True)
class Number:
    value: float


class Add(Operation):
    replayable = True

    def forward(self, value, *, context=None):
        return Number(value.value + (context or {}).get('offset', Number(1)).value)


def test_replay_recomputes_intermediates_from_external_roots():
    with tc.trace() as episode:
        first = Add()(Number(2))
        last = Add()(first)
    example = episode.example(episode.ref(last))
    assert len(example.inputs) == 1
    assert len(example.calls) == 2
    root = next(iter(example.inputs))
    assert episode.replay(episode.ref(last), inputs={root: Number(10)}) == Number(12)


def test_context_output_is_a_dependency_not_an_independent_input():
    with tc.trace() as episode:
        offset = Add()(Number(4))
        last = Add()(Number(2), context={'offset': offset})
    example = episode.example(episode.ref(last))
    assert len(example.calls) == 2
    assert len(example.inputs) == 2
    assert episode.replay(episode.ref(last)) == Number(7)


def test_equal_objects_have_distinct_producers_and_scalar_refs_are_explicit():
    class Scalar(Operation):
        replayable = True
        def forward(self, value, *, context=None):
            return 1
    with tc.trace() as episode:
        a, b = Add()(Number(0)), Add()(Number(0))
        Scalar()(0)
        Scalar()(0)
    assert episode.ref(a) != episode.ref(b)
    with pytest.raises(ValueError, match='explicit'):
        episode.ref(1)
    assert episode.replay(episode.calls[-1].output) == 1


def test_explicit_scalar_reference_connects_consumer():
    class Twice(Operation):
        replayable = True
        def forward(self, value, *, context=None):
            return value * 2
    with tc.trace() as episode:
        Twice()(3)
        result = Twice()(episode.calls[-1].output)
    assert result == 12
    example = episode.example(episode.calls[-1].output)
    assert len(example.inputs) == 1
    assert len(example.calls) == 2


def test_failed_operation_and_nested_scopes_restore_outer_trace():
    class Fail(Operation):
        def forward(self, value, *, context=None):
            raise RuntimeError('provider down')
    with tc.trace() as outer:
        Add()(Number(1))
        with tc.trace() as inner:
            with pytest.raises(RuntimeError, match='provider down'):
                Fail()(Number(2))
        Add()(Number(3))
    assert len(outer.calls) == 2
    assert inner.calls[0].error == 'RuntimeError: provider down'
    Add()(Number(5))
    assert len(outer.calls) == 2


def test_effectful_operation_cannot_be_replayed():
    class Effect(Operation):
        def forward(self, value, *, context=None):
            return Number(9)
    with tc.trace() as episode:
        result = Effect()(Number(0))
    with pytest.raises(ValueError, match='replay'):
        episode.replay(episode.ref(result))


def test_external_mutable_input_is_snapshotted():
    class Sum(Operation):
        replayable = True
        def forward(self, value, *, context=None):
            return Number(sum(value))
    values = [1, 2]
    with tc.trace() as episode:
        result = Sum()(values)
    values.append(100)
    assert episode.replay(episode.ref(result)) == Number(3)


def test_foreign_reference_is_rejected():
    with tc.trace() as first:
        result = Add()(Number(1))
    with tc.trace():
        with pytest.raises(ValueError, match='session'):
            Add()(first.ref(result))
