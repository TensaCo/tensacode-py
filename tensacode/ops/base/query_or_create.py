from typing import Any, ClassVar, Literal, Optional, Annotated
from typing_extensions import Self

from tensacode.core.base_engine import Engine
from tensacode.internal.latent import LatentType
from tensacode.core.base.ops.base_op import Op
from tensacode.internal.utils.tc import loop_until_done
from tensacode.internal.utils.misc import score_inheritance_distance, Score


@Engine.register_op_on_class
@score_inheritance_distance
def query_or_create(
    engine: Engine,
    target: Annotated[Any | None, Score(coefficient=1)],
    query: Annotated[Optional[Any], Score(coefficient=1)] = None,
    search_strategy: Annotated[Literal["beam", "greedy", "breadth", "depth"], Score(coefficient=1)] = "greedy",
    top_p=1.0,
    max_rounds=1,
    **kwargs: Any,
) -> Any:
    """
    Query an existing object or create a new one based on the provided parameters.

    This operation decides whether to query for an existing object or create a new one,
    using the engine's capabilities to make the decision and perform the chosen action.

    Args:
        engine (Engine): The engine used for querying or creating.
        target (Any | None, optional): The target object to create if creation is chosen. Defaults to None.
        query (Optional[Any], optional): The query to use if querying is chosen. Defaults to None.
        search_strategy (Literal["beam", "greedy", "breadth", "depth"], optional): The search strategy for querying. Defaults to "greedy".
        top_p (float, optional): The cumulative probability threshold for sampling. Defaults to 1.0.
        max_rounds (int, optional): The maximum number of query rounds. Defaults to 1.
        **kwargs: Additional keyword arguments to be passed to the engine.

    Returns:
        Any: The result of the query or the newly created object.

    Raises:
        ValueError: If an invalid choice is made between querying and creating.

    Examples:
        >>> result = query_or_create(engine, target="Create a new user", query="Find user by email")
        >>> print(result)
        User(id=1, name="John Doe", email="john@example.com")

        >>> result = query_or_create(engine, query="Find product by ID", search_strategy="beam", top_p=0.8)
        >>> print(result)
        Product(id=42, name="Smartphone", price=599.99)
    """
    match engine.decode("Should we query or create?", type=Literal["query", "create"]):
        case "query":
            return engine.query(
                query,
                search_strategy=search_strategy,
                top_p=top_p,
                max_rounds=max_rounds,
                **kwargs,
            )
        case "create":
            return engine.decode(target, **kwargs)
        case _:
            raise ValueError("Invalid choice")