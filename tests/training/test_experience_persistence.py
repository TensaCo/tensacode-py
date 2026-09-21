from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest
import torch
from tensorcode import trace
from tensorcode.ops import Operation
from tensorcode.ops.vec import Classify, Transform
from tensorcode import training


@dataclass(frozen=True)
class Number:
    value: float


class Add(Operation):
    replayable = True
    def forward(self, value, *, context=None):
        return Number(value.value + context.get('offset', Number(1)).value)


def test_roundtrip_dataclass_context_release_and_missing_codecs(tmp_path):
    op = Add()
    with trace() as session:
        a = op(Number(2))
        out = op(a, context={'offset': Number(4)})
    ref = session.ref(out)
    session.supervise(ref, Number(8), loss='custom', source='reviewer:42')
    path = tmp_path / 'experience.json'
    with pytest.raises(TypeError, match='codec'):
        session.save(path, operations={'add': op})
    session.save(path, operations={'add': op}, codecs={'number': Number}, release=True)
    assert all(call.result is None for call in session.calls)
    assert not session._objects
    assert session.replay(ref) == Number(7)
    with pytest.raises(ValueError, match='codec'):
        training.load(path, operations={'add': op})
    loaded = training.load(path, operations={'add': Add()}, codecs={'number': Number})
    assert loaded.replay(loaded.supervisions[0].output) == Number(7)
    assert loaded.supervisions[0].source == 'reviewer:42'


def test_missing_changed_bindings_and_malformed_artifacts(tmp_path):
    op = Classify(torch.nn.Linear(2, 2), labels=('a', 'b'))
    with trace() as session:
        result = op(torch.tensor([1., 2.]))
    session.supervise(result, 'a')
    path = tmp_path / 'experience.json'
    session.save(path, operations={'head': op})
    with pytest.raises(ValueError, match='binding'):
        training.load(path, operations={})
    changed = Classify(torch.nn.Linear(2, 2), labels=('b', 'a'))
    with pytest.raises(ValueError, match='configuration'):
        training.load(path, operations={'head': changed})
    # Updated weights are intentionally not configuration changes.
    training.load(path, operations={'head': Classify(torch.nn.Linear(2, 2), labels=('a', 'b'))})
    data = json.loads(path.read_text())
    data['calls'][0]['value'] = {'kind': 'output', 'call': 999, 'path': []}
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        training.load(path, operations={'head': op})


def test_fresh_subprocess_load_replays_gradients(tmp_path):
    op = Classify(torch.nn.Linear(2, 2), labels=('a', 'b'))
    with trace() as session:
        result = op(torch.tensor([1., -1.]))
    session.supervise(result, 'a')
    path = tmp_path / 'experience.json'
    session.save(path, operations={'head': op}, release=True)
    code = '''
import torch
from tensorcode import training
from tensorcode.ops.vec import Classify
op = Classify(torch.nn.Linear(2, 2), labels=('a', 'b'))
session = training.load(PATH, operations={'head': op})
before = op.module.weight.detach().clone()
trainer = training.Trainer({'head': op}, lr=0.1)
losses = trainer.fit([session], epochs=8)
assert losses[-1] < losses[0]
assert op.module.weight.grad is not None
assert not torch.equal(before, op.module.weight)
'''.replace('PATH', repr(str(path)))
    env = {**os.environ, 'PYTHONPATH': str(Path(__file__).parents[2] / 'src')}
    subprocess.run([sys.executable, '-c', code], check=True, env=env)


def test_effect_boundary_is_recorded_and_never_reinvoked(tmp_path):
    class Effect(Operation):
        calls = 0
        def forward(self, value, *, context=None):
            self.calls += 1
            return torch.tensor([float(value)])
    effect = Effect()
    head = Transform(torch.nn.Linear(1, 1))
    with trace() as session:
        out = head(effect(2))
    session.supervise(out, torch.tensor([4.]), loss='mse')
    path = tmp_path / 'experience.json'
    session.save(path, operations={'external': effect, 'head': head})
    loaded = training.load(path, operations={'external': effect, 'head': head})
    with pytest.raises(ValueError, match='replay'):
        loaded.replay(loaded.supervisions[0].output)
    training.Trainer({'external': effect, 'head': head}).step(loaded)
    assert effect.calls == 1


def test_safe_codec_rejects_executable_type_tags_and_duplicate_json(tmp_path):
    op = Add()
    with trace() as session:
        out = op(Number(1))
    session.supervise(out, Number(2), loss='custom')
    path = tmp_path / 'safe.json'
    session.save(path, operations={'add': op}, codecs={'number': Number})
    data = json.loads(path.read_text())
    data['inputs']['0'] = {'type': 'pickle', 'data': 'not executable'}
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='codec'):
        training.load(path, operations={'add': op}, codecs={'number': Number})
    path.write_text('{"format":"tensorcode.experience","format":"other","version":1}')
    with pytest.raises(ValueError, match='Duplicate'):
        training.load(path, operations={'add': op})


def test_training_import_does_not_import_tensor_backend():
    env = {**os.environ, 'PYTHONPATH': str(Path(__file__).parents[2] / 'src')}
    subprocess.run([sys.executable, '-S', '-c',
                    'import sys; import tensorcode; import tensorcode.training; assert "torch" not in sys.modules'],
                   check=True, env=env)


def test_async_capture_records_awaited_result_and_failure():
    import asyncio
    from tensorcode.tracing import invoke_async
    class AsyncAdd(Add):
        async def aforward(self, value, *, context=None):
            await asyncio.sleep(0)
            return self.forward(value, context=context)
    async def run():
        op = AsyncAdd()
        with trace() as session:
            first = await invoke_async(op, Number(1), {}, op.aforward)
            last = await invoke_async(op, first, {}, op.aforward)
        assert len(session.calls) == 2
        assert session.replay(session.ref(last)) == Number(3)
        async def fail(value, *, context):
            raise RuntimeError('async provider failed')
        with trace() as failed:
            with pytest.raises(RuntimeError, match='provider'):
                await invoke_async(op, Number(0), {}, fail)
        assert failed.calls[0].error == 'RuntimeError: async provider failed'
    asyncio.run(run())


def test_save_rejects_mutated_intermediate_and_is_atomic(tmp_path):
    op = Transform(torch.nn.Linear(1, 1))
    with trace() as session:
        out = op(torch.tensor([1.]))
    path = tmp_path / 'preserved.json'
    path.write_text('original content')
    with torch.no_grad():
        out.add_(1)
    with pytest.raises(ValueError, match='mutated'):
        session.save(path, operations={'head': op})
    assert path.read_text() == 'original content'


def test_release_rejects_mutated_boundaries_without_partially_releasing():
    class External(Operation):
        def forward(self, value, *, context=None):
            return torch.tensor([float(value)])
    external = External()
    head = Transform(torch.nn.Linear(1, 1))
    with trace() as session:
        boundary = external(1)
        out = head(boundary)
    boundary.add_(100)
    with pytest.raises(ValueError, match='mutated'):
        session.release()
    assert not session._released
    assert session.calls[0].result is boundary
    assert session.calls[1].result is out
    assert not session._boundaries


def test_pending_async_calls_cannot_be_persisted_or_released(tmp_path):
    import asyncio
    from tensorcode.tracing import invoke_async
    class External(Operation):
        def forward(self, value, *, context=None):
            return value
    async def run():
        ready = asyncio.Event()
        op = External()
        async def waiting(value, *, context=None):
            await ready.wait()
            return [value]
        with trace() as session:
            task = asyncio.create_task(invoke_async(op, 2, {}, waiting))
            await asyncio.sleep(0)
        for action in (lambda: session.save(tmp_path / 'pending.json', operations={'external': op}), session.release,
                       lambda: session.ref(session.calls[0].output)):
            with pytest.raises(RuntimeError, match='pending'):
                action()
        ready.set()
        assert await task == [2]
        session.save(tmp_path / 'done.json', operations={'external': op})
        loaded = training.load(tmp_path / 'done.json', operations={'external': op})
        assert loaded.replay(loaded.calls[0].output, boundary='recorded') == [2]
        # A child task retaining the old ContextVar cannot begin a new call.
        async def late():
            await asyncio.sleep(0)
            return await invoke_async(op, 3, {}, waiting)
        with trace() as closed:
            delayed = asyncio.create_task(late())
        with pytest.raises(RuntimeError, match='closed'):
            await delayed
        assert not closed.calls
    asyncio.run(run())


def test_immutable_message_and_graph_payloads_roundtrip(tmp_path):
    from tensorcode.ops.llm.classify import ClassificationResult
    from tensorcode.ops.graph import Graph, SourceAnchor
    class Echo(Operation):
        replayable = True
        def forward(self, value, *, context=None):
            # Return new containers to retain unambiguous object identity.
            return {'value': value, 'context': context}
    op = Echo()
    graph = Graph(nodes=('a',), sources=('source:1',), attributes={'nested': {'flag': True}},
                  source_anchors=(SourceAnchor('source:1', target='a', location={'line': 2}),))
    result = ClassificationResult('yes', {'yes': .75, 'no': .25})
    with trace() as session:
        output = op(graph, context={'decision': result})
    codecs = {'graph': Graph, 'anchor': SourceAnchor, 'classification': ClassificationResult}
    session.supervise(output, result, loss='custom')
    path = tmp_path / 'immutable.json'
    session.save(path, operations={'echo': op}, codecs=codecs, release=True)
    loaded = training.load(path, operations={'echo': op}, codecs=codecs)
    restored = loaded.replay(loaded.supervisions[0].output)
    assert restored['value'] == graph
    assert restored['context']['decision'] == result
    with pytest.raises(TypeError):
        restored['context']['decision'].distribution['yes'] = 0
