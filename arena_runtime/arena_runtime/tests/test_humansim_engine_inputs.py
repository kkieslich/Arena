"""Engine inputs decided while the engine ticks under the input barrier go out in the step after their frame."""

from __future__ import annotations

import asyncio

from arena_runtime.humans.arena_humansim.engine_inputs import EngineInputs, SequencedClient


class _Engine:
    def __init__(self) -> None:
        self.calls: list[object] = []

    async def call_timeout(self, request: object) -> object:
        self.calls.append(request)
        await asyncio.sleep(0)
        return f"ok {request}"

    def ensure(self) -> str:
        return "ensured"


def test_inactive_calls_go_out_at_once() -> None:
    engine = _Engine()
    client = SequencedClient(engine, EngineInputs(lambda: 0))
    assert asyncio.run(client.call_timeout("a")) == "ok a"
    assert engine.calls == ["a"]
    assert client.ensure() == "ensured"


def test_held_calls_wait_for_the_step_that_covers_their_clock_and_keep_their_order() -> None:
    async def scenario() -> None:
        clock = [100]
        inputs = EngineInputs(lambda: clock[0])
        inputs.active = True
        engine = _Engine()
        client = SequencedClient(engine, inputs)
        first = asyncio.create_task(client.call_timeout("first"))
        second = asyncio.create_task(client.call_timeout("second"))
        await asyncio.sleep(0)
        clock[0] = 150
        later = asyncio.create_task(client.call_timeout("later"))
        await asyncio.sleep(0)
        assert engine.calls == [] and inputs.held == 3
        await inputs.flush(before_ns=150)  # the step of the frame stamped 100 with dt 50
        assert engine.calls == ["first", "second"]
        assert await first == "ok first" and await second == "ok second"
        assert not later.done()
        await inputs.flush(before_ns=200)
        assert await later == "ok later"

    asyncio.run(scenario())


def test_calls_inside_a_step_go_out_at_once() -> None:
    async def scenario() -> None:
        inputs = EngineInputs(lambda: 0)
        inputs.active = True
        engine = _Engine()
        client = SequencedClient(engine, inputs)
        with inputs.step():
            assert await client.call_timeout("in step") == "ok in step"

    asyncio.run(scenario())


def test_a_stopped_clock_lets_held_calls_out() -> None:
    async def scenario() -> None:
        stopped = [False]
        inputs = EngineInputs(lambda: 0, lambda: stopped[0])
        inputs.POLL_S = 0.01
        inputs.active = True
        engine = _Engine()
        call = asyncio.create_task(SequencedClient(engine, inputs).call_timeout("reset"))
        await asyncio.sleep(0.05)
        assert not call.done()
        stopped[0] = True
        assert await asyncio.wait_for(call, 1.0) == "ok reset"
        assert inputs.held == 0

    asyncio.run(scenario())


def test_a_failing_held_call_raises_in_its_caller_and_the_rest_still_go_out() -> None:
    class _Failing(_Engine):
        async def call_timeout(self, request: object) -> object:
            if request == "bad":
                raise RuntimeError("engine down")
            return await super().call_timeout(request)

    async def scenario() -> None:
        inputs = EngineInputs(lambda: 0)
        inputs.active = True
        engine = _Failing()
        client = SequencedClient(engine, inputs)
        bad = asyncio.create_task(client.call_timeout("bad"))
        good = asyncio.create_task(client.call_timeout("good"))
        await asyncio.sleep(0)
        await inputs.flush()
        assert await good == "ok good"
        try:
            await bad
        except RuntimeError as e:
            assert "engine down" in str(e)
        else:
            raise AssertionError("the failure did not reach its caller")

    asyncio.run(scenario())


def test_a_step_cancelled_mid_call_still_answers_the_caller() -> None:
    class _Slow(_Engine):
        async def call_timeout(self, request: object) -> object:
            await asyncio.sleep(0.05)
            return await super().call_timeout(request)

    async def scenario() -> None:
        inputs = EngineInputs(lambda: 0)
        inputs.active = True
        client = SequencedClient(_Slow(), inputs)
        call = asyncio.create_task(client.call_timeout("spawn"))
        await asyncio.sleep(0)
        step = asyncio.create_task(inputs.flush())
        await asyncio.sleep(0.01)
        step.cancel()
        assert await asyncio.wait_for(call, 1.0) == "ok spawn"

    asyncio.run(scenario())
