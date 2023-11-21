from typing import Any


@ledgered
def find(directions, max_height=5, max_nested=2, skip_last_frames=0) -> Any:
    """Performs horizontal (dot-notation access, dict lookup, indexing, etc.) and vertical search (climbing stack frames) to find a variable.

    Args:
        directions (str): What you're looking for.
        max_height (int, optional): Maximum number of frames to climb. Defaults to 5.
        max_nested (int, optional): Maximum levels of nested access to explore. Defaults to 2.
        skip_last_frames (int, optional): Skip the first `skip_last_frames` when climbing the call stack. These skipped frames do not count towards the `max_height`. Defaults to 0.

    Returns:
        Any: The object you're looking for.

    Raises:
        ValueError: If the object cannot be found.
    """
    frames = TODO
    for frame in frames:
        dict = {**frame.f_locals, **frame.f_globals}
