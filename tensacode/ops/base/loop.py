from typing import Optional, Generator, Any
from tensacode.core.base_engine import Engine


@Engine.register_op_on_class()
def loop(
    engine: Engine,
    count: int | None = None,
    min: int | None = None,
    max: int | None = None,
    continue_prompt: Optional[str] = None,
    stop_prompt: Optional[str] = None,
) -> Generator[int, None, None]:
    """
    A generator function that loops until specified criteria are met or the engine decides to stop.

    This function yields iteration numbers based on the provided parameters and engine decisions.

    Args:
        engine (Engine): The intelligent LLM-based engine used for making decisions.
        count (int | None, optional): The exact number of iterations to perform. If specified, overrides min and max. Defaults to None.
        min (int | None, optional): The minimum number of iterations. Defaults to None.
        max (int | None, optional): The maximum number of iterations. Defaults to None.
        continue_prompt (Optional[str], optional): The prompt to pass to the engine to decide whether to continue. Defaults to None.
        stop_prompt (Optional[str], optional): The prompt to pass to the engine to decide whether to stop. Defaults to None.

    Yields:
        int: The current iteration number, starting from 0.

    Raises:
        StopIteration: When the iteration criteria are met or the engine decides to stop.

    Examples:
        >>> # Basic usage with max iterations
        >>> list(engine.loop(max=5))
        [0, 1, 2, 3, 4]

        >>> # Using count (overrides min and max)
        >>> list(engine.loop(count=3, min=0, max=5))
        [0, 1, 2]

        >>> # Using min and max with engine decisions
        >>> result = list(engine.loop(min=2, max=7, continue_prompt="Should we continue the loop?"))
        >>> 2 <= len(result) <= 7
        True

        >>> # Using engine with stop prompt
        >>> result = list(engine.loop(max=10, stop_prompt="Should we stop the loop now?"))
        >>> 0 <= len(result) <= 10
        True

        >>> # Unbounded loop (max=-1) with engine decisions
        >>> from itertools import islice
        >>> result = list(islice(engine.loop(max=-1, continue_prompt="Keep looping?"), 20))
        >>> 0 < len(result) <= 20
        True
    """
    # Validate input parameters
    assert (
        max is None or max == -1 or max >= 0
    ), "max must be None, -1, or a non-negative integer"
    assert (
        min is None or min == -1 or min >= 0
    ), "min must be None, -1, or a non-negative integer"
    assert (
        count is None or count == -1 or count >= 0
    ), "count must be None, -1, or a non-negative integer"

    # If count is specified, use it for both min and max
    if count is not None:
        max = min = count

    # Convert -1 to None for min and max
    if min == -1:
        min = None
    if max == -1:
        max = None

    iteration = 0
    while True:
        # Check if minimum iterations have been reached
        if min is None or iteration >= min:
            # Check if maximum iterations have been reached
            if max is not None and iteration >= max:
                raise StopIteration("Maximum iterations reached")

            # Check with engine if we should continue
            if continue_prompt and not engine.decide(continue_prompt):
                raise StopIteration("Engine decided to stop")
            # Check with engine if we should stop
            if stop_prompt and engine.decide(stop_prompt):
                raise StopIteration("Engine decided to stop")

        yield iteration
        iteration += 1
