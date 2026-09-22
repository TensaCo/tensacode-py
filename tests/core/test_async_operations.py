import asyncio
import tensorcode as tc
from tensorcode.ops.base import Operation


class Add(Operation):
    replayable = True
    def forward(self, value, *, context=None):
        return [value[0] + (context or {}).get('amount', 1)]


def test_async_operation_preserves_capture_and_conditioning():
    async def run():
        add = Add()
        with tc.trace() as session:
            first = await add.acall([2], context={'amount': 3})
            second = await add.acall(first)
        assert second == [6]
        assert len(session.calls) == 2
        assert session.replay(session.ref(second)) == [6]
    asyncio.run(run())


from dataclasses import dataclass

import pytest
import torch


@dataclass
class Payload:
    tensor: torch.Tensor


class AwaitedScale(Operation):
    replayable = True

    def __init__(self, entered, resume, *, failure=False):
        self.entered, self.resume = entered, resume
        self.failure = failure
        self.weight = torch.nn.Parameter(torch.tensor(2.))
        self.last_result = None

    def forward(self, value, *, context=None):
        return value['payload'].tensor * context['scale']['amount'] * self.weight

    async def aforward(self, value, *, context=None):
        self.entered.set()
        await self.resume.wait()
        if self.failure:
            raise RuntimeError('provider failed after await')
        self.last_result = self.forward(value, context=context)
        return self.last_result


@pytest.mark.parametrize('mutation', ['tensor', 'view_base', 'dataclass', 'context_tensor', 'context_container'])
def test_async_mutated_live_inputs_fail_before_output_registration(mutation):
    async def run():
        entered, resume = asyncio.Event(), asyncio.Event()
        operation = AwaitedScale(entered, resume)
        base = torch.tensor([1., 2.], requires_grad=True)
        value = {'payload': Payload(base[:1])}
        context = {'scale': {'amount': torch.tensor(3., requires_grad=True)}}
        with tc.trace() as session:
            pending = asyncio.create_task(operation.acall(value, context=context))
            await entered.wait()
            with torch.no_grad():
                if mutation == 'tensor':
                    value['payload'].tensor.add_(5)
                elif mutation == 'view_base':
                    base.add_(5)
                elif mutation == 'dataclass':
                    value['payload'].tensor = torch.tensor([7.])
                elif mutation == 'context_tensor':
                    context['scale']['amount'].add_(5)
                else:
                    context['scale']['amount'] = torch.tensor(7.)
            resume.set()
            with pytest.raises(ValueError, match='mutated.*async'):
                await pending
            failed = session.calls[0]
            assert not failed.pending and failed.result is None
            assert 'ValueError' in failed.error
            with pytest.raises(ValueError, match='Failed call'):
                session.example(failed.output)
            with pytest.raises(ValueError, match='Unknown'):
                session.ref(operation.last_result)
            # A failed call does not poison independently captured later inputs.
            recovered = await operation.acall(value, context=context)
            torch.testing.assert_close(session.replay(recovered), recovered)
            assert session.example(recovered).calls == (1,)
    asyncio.run(run())


def test_async_unchanged_live_tensors_preserve_native_and_replay_gradients():
    async def run():
        entered, resume = asyncio.Event(), asyncio.Event()
        operation = AwaitedScale(entered, resume)
        value_tensor = torch.tensor([2.], requires_grad=True)
        amount = torch.tensor(3., requires_grad=True)
        with tc.trace() as session:
            pending = asyncio.create_task(operation.acall({'payload': Payload(value_tensor)},
                                                         context={'scale': {'amount': amount}}))
            await entered.wait()
            resume.set()
            result = await pending
        result.sum().backward()
        assert value_tensor.grad.item() == 6.
        assert amount.grad.item() == 4.
        assert operation.weight.grad.item() == 6.
        operation.weight.grad = None
        replayed = session.replay(result)
        torch.testing.assert_close(replayed, result)
        replayed.sum().backward()
        assert operation.weight.grad.item() == 6.
    asyncio.run(run())


def test_async_provider_error_is_preserved_after_input_mutation():
    async def run():
        entered, resume = asyncio.Event(), asyncio.Event()
        operation = AwaitedScale(entered, resume, failure=True)
        tensor = torch.tensor([1.])
        with tc.trace() as session:
            task = asyncio.create_task(operation.acall({'payload': Payload(tensor)},
                                                      context={'scale': {'amount': torch.tensor(3.)}}))
            await entered.wait()
            tensor.add_(1)
            resume.set()
            with pytest.raises(RuntimeError, match='provider failed after await'):
                await task
        assert session.calls[0].error == 'RuntimeError: provider failed after await'
        assert not session.calls[0].pending and session.calls[0].result is None
    asyncio.run(run())
