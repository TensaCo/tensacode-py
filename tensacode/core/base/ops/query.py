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
    """
    Query an object or context based on the provided parameters.

    This operation uses the engine to search for and retrieve information from the target object or context,
    guided by the provided query and search strategy.

    Args:
        engine (BaseEngine): The engine used for querying.
        target (Any | None, optional): The target object or context to query. If None, uses engine.context. Defaults to None.
        query (Optional[Any], optional): The query to guide the search. Defaults to None.
        search_strategy (Literal["beam", "greedy", "breadth", "depth"], optional): The search strategy to use. Currently, only "greedy" is supported. Defaults to "greedy".
        top_p (float, optional): The cumulative probability threshold for sampling. Defaults to 1.0.
        max_rounds (int, optional): The maximum number of query rounds. Defaults to 1.
        **kwargs: Additional keyword arguments to be passed to the engine.

    Returns:
        Any: The result of the query, typically in the form of a latent representation.

    Raises:
        ValueError: If an unsupported search strategy is specified.

    Examples:
        >>> result = query(engine, target=document, query="Find the author's name")
        >>> print(engine.decode(result))
        "John Doe"

        >>> result = query(engine, query="Summarize the main points", max_rounds=3)
        >>> print(engine.decode(result))
        "1. Introduction to AI
         2. Machine Learning algorithms
         3. Applications in various industries"
    """
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
