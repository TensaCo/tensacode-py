from typing import Any, ClassVar, Literal, Optional
from typing_extensions import Self

from tensacode.core.base.base_engine import BaseEngine
from tensacode.internal.latent import LatentType
from tensacode.core.base.ops.base_op import Op
from tensacode.internal.utils.tc import loop_until_done


@BaseEngine.register_op()
def query_or_create(
    engine: BaseEngine,
    target: Any | None = None,
    query: Optional[Any] = None,
    search_strategy: Literal["beam", "greedy", "breadth", "depth"] = "greedy",
    top_p=1.0,
    max_rounds=1,
    **kwargs: Any,
) -> Any:
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
