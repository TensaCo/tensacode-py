from typing import Any, ClassVar
from typing_extensions import Self

from tensacode.core.base.base_engine import BaseEngine
from tensacode.internal.latent import LatentType
from tensacode.core.base.ops.base_op import Op


class BaseQueryOp(Op):
    name: ClassVar[str] = "query"
    latent_type: ClassVar[LatentType] = LatentType
    engine_type: ClassVar[type[BaseEngine]] = BaseEngine


@BaseEngine.register_op_class_for_all_class_instances
@BaseQueryOp.create_subclass(name="query")
def Query(
    engine: BaseEngine,
    target,
    query_latent: Latent = None,
    search_strategy: Literal["beam", "greedy", "breadth", "depth"] = "greedy",
    top_k=1,
    top_p=1.0,
    max_rounds=1,
    **kwargs: Any,
) -> Any | list[Any]:
    if search_strategy != "greedy":
        raise ValueError(
            f"Search strategy {search_strategy} not supported for query op. sorry! :("
        )

    for _ in range(max_rounds):

        locators = engine.locate(
            target,
            query_latent=query_latent,
            top_k=top_k,
            top_p=top_p,
            **kwargs,
        )
        engine.info("locators", locators=locators)
        values = [
            locator.get(target, current=target, create_missing=False)
            for locator in locators
        ]
        engine.info("values", values=values)
        encoded_values = [
            engine.encode(value, target, query_latent=query_latent, **kwargs)
            for value in values
        ]
        engine.info("encoded_values", encoded_values=encoded_values)
        prepped_inputs = [
            (locator, value, encoded_value)
            for locator, value, encoded_value in zip(locators, values, encoded_values)
        ]
        engine.info("prepped_inputs", prepped_inputs=prepped_inputs)
        updated_latent = engine.transform(
            target,
            query_latent=query_latent,
            prepped_inputs=prepped_inputs,
            **kwargs,
        )
        engine.info("updated_latent", updated_latent=updated_latent)

        if engine.decide("done querying?"):
            break
        query_latent = updated_latent

    return query_latent
