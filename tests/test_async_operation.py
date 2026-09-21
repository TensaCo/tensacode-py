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
