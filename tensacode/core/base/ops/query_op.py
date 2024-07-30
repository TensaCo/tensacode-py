from typing import Any, ClassVar, Literal, Optional
from typing_extensions import Self

from tensacode.core.base.base_engine import BaseEngine
from tensacode.internal.latent import LatentType
from tensacode.core.base.ops.base_op import Op
from tensacode.internal.utils.tc import loop_until_done


@BaseEngine.register_op()
def query(
    engine: BaseEngine,
    target: Any | None = None,
    query: Optional[Any] = None,
    search_strategy: Literal["beam", "greedy", "breadth", "depth"] = "greedy",
    top_p=1.0,
    max_rounds=1,
    **kwargs: Any,
) -> Any:
    if target is None:
        target = engine.context

    if search_strategy != "greedy":
        raise ValueError(
            f"Search strategy {search_strategy} not supported for query op. sorry! :("
        )

    query_latent = engine.encode(query, **kwargs)

    for step in loop_until_done(
        max_rounds,
        engine=engine,
        continue_prompt="Continue querying?",
        stop_prompt="Done querying?",
    ):
        with engine.scope(step=step, max_steps=max_rounds):
            locator = engine.locate(
                target,
                query_latent=query_latent,
                top_p=top_p,
                **kwargs,
            )
            engine.info("locator", locator=locator)
            value = locator.get(target, current=target, create_missing=False)
            engine.info("value", value=value)
            encoded_value = engine.encode(
                value, target, query_latent=query_latent, **kwargs
            )
            engine.info("encoded_value", encoded_value=encoded_value)
            prepped_input = (locator, value, encoded_value)
            engine.info("prepped_input", prepped_input=prepped_input)
            updated_latent = engine.transform(
                target,
                query_latent=query_latent,
                prepped_input=prepped_input,
                **kwargs,
            )
            engine.info("updated_latent", updated_latent=updated_latent)

            query_latent = updated_latent

    return query_latent
