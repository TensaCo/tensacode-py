from typing import Optional


# TODO


def loop(
    count: int | None = None,
    min: int | None = None,
    max: int | None = None,
    engine=None,
    continue_prompt: Optional[str] = None,
    stop_prompt: Optional[str] = None,
):
    """
    A generator function that loops until a condition is met or specified criteria are reached.

    This function yields iteration numbers until one of the following conditions is met:
    1. The maximum number of iterations (if specified) is reached.
    2. The minimum number of iterations (if specified) is reached and other stop conditions are met.
    3. The exact count of iterations (if specified) is reached.
    4. The engine (if provided) decides to stop based on the given prompt.

    Args:
        max (int | None, optional): The maximum number of iterations. If None, there's no upper limit. Defaults to None.
        min (int | None, optional): The minimum number of iterations. If None, there's no lower limit. Defaults to None.
        count (int | None, optional): The exact number of iterations to perform. If specified, overrides max and min. Defaults to None.
        engine (Any, optional): An engine object with a 'decide' method. If provided, it's used to determine when to stop. Defaults to None.
        continue_prompt (str, optional): The prompt to pass to the engine's 'decide' method to continue. Defaults to None.
        stop_prompt (str, optional): The prompt to pass to the engine's 'decide' method to stop. Defaults to None.

    Yields:
        int: The current iteration number, starting from 0.

    Raises:
        StopIteration: When the iteration criteria are met or the engine decides to stop.

    Examples:
        >>> # Basic usage with max iterations
        >>> list(loop_until_done(max=10))
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        >>> # no engine passed, goes until max

        >>> list(loop_until_done(max=10, engine=engine))
        [0, 1, 2, 3, 4]
        >>> # engine decided to stop after 5 iterations

        >>> # Using min and max
        >>> list(loop_until_done(min=2, max=7, engine=engine))
        [0, 1, 2, 3]
        >>> # engine decided to stop after 4 iterations

        >>> # Using count (overrides min and max)
        >>> list(loop_until_done(count=2, min=0, max=5, engine=engine))
        [0, 1]
        >>> # engine decided to stop after 2 iterations
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
            if engine and continue_prompt and not engine.decide(continue_prompt):
                raise StopIteration("Engine decided to stop")
            # Check with engine if we should stop
            if engine and stop_prompt and engine.decide(stop_prompt):
                raise StopIteration("Engine decided to stop")

        yield iteration
        iteration += 1
