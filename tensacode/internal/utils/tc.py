from typing import Optional


def loop_until_done(
    limit: int | None = None,
    /,
    engine=None,
    continue_prompt: Optional[str] = None,
    stop_prompt: Optional[str] = None,
):
    """
    A generator function that loops until a condition is met or a limit is reached.

    This function yields iteration numbers until one of the following conditions is met:
    1. The iteration limit (if specified) is reached.
    2. The engine (if provided) decides to stop based on the given prompt.

    Args:
        limit (int | None, optional): The maximum number of iterations. If None, there's no limit. Defaults to None.
        engine (Any, optional): An engine object with a 'decide' method. If provided, it's used to determine when to stop. Defaults to None.
        continue_prompt (str, optional): The prompt to pass to the engine's 'decide' method. Defaults to "Continue?".
        stop_prompt (str, optional): The prompt to pass to the engine's 'decide' method. Defaults to "Stop?".

    Yields:
        int: The current iteration number, starting from 0.

    Raises:
        StopIteration: When the iteration limit is reached or the engine decides to stop.

    Example:
        >>> for i in loop_until_done(10, engine=my_engine, continue_prompt="Keep going?", stop_prompt="Stop?"):
        ...     print(f"Iteration {i}")
        Iteration 0
        Iteration 1
        Iteration 2
        Iteration 3
        Iteration 4
    """
    assert (
        limit is None or limit == -1 or limit >= 0
    ), "Limit must be None, -1, or a positive integer"

    iteration = 0
    while True:
        if limit is not None and iteration >= limit:
            raise StopIteration("Iteration limit reached")
        if engine and continue_prompt and not engine.decide(continue_prompt):
            raise StopIteration("Engine decided to stop")
        if engine and stop_prompt and engine.decide(stop_prompt):
            raise StopIteration("Engine decided to stop")
        yield iteration
        iteration += 1
