from typing import Any, ClassVar
from typing_extensions import Self

from tensacode.core.base_engine import Engine
from tensacode.internal.latent import LatentType
from tensacode.core.base.ops.base_op import Op
from tensacode.internal.utils.tc import loop_until_done


@Engine.register_op()
def split(
    engine: Engine,
    *inputs: list[Any],
    prompt: Optional[Encoded[str]] = None,
    modify_steps: list[str] = [],
    **kwargs: Any,
) -> Any:
    """
    Split the input(s) into multiple categories.

    This operation divides the input(s) into different categories based on the engine's understanding
    and the optional prompt.

    Args:
        engine (Engine): The engine used for splitting.
        *inputs (list[Any]): The inputs to be split.
        prompt (Optional[Encoded[str]], optional): A prompt to guide the splitting process. Defaults to None.
        modify_steps (list[str], optional): Steps to modify the split results. Defaults to [].
        **kwargs: Additional keyword arguments to be passed to the engine.

    Returns:
        Any: A dictionary of categories and their corresponding elements.

    Examples:
        >>> text = "The quick brown fox jumps over the lazy dog"
        >>> result = split(engine, text, prompt="Split into word types")
        >>> print(result)
        {
            'articles': ['The', 'the'],
            'adjectives': ['quick', 'brown', 'lazy'],
            'nouns': ['fox', 'dog'],
            'verbs': ['jumps'],
            'prepositions': ['over']
        }

        >>> numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        >>> result = split(engine, numbers, prompt="Split into even and odd numbers")
        >>> print(result)
        {
            'even': [2, 4, 6, 8, 10],
            'odd': [1, 3, 5, 7, 9]
        }
    """
    category_names = engine.decode(list[str], prompt="Generate a list of categories")
    category_elem_types = {}
    for bin_name in category_names:
        category_elem_types[bin_name] = engine.decode(
            type, prompt="Determine the element type of the category: {bin_name}"
        )
    categories = {
        category_name: engine.create(category_elem_types[category_name])
        for category_name in category_names
    }
    for _ in loop_until_done(modify_steps, engine=engine, continue_prompt="Continue?"):
        for category_name, category_value in categories.items():
            with engine.scope(category=category_value):
                categories = engine.modify(
                    categories,
                    prompt=f"Modify the {category_name}",
                )
    return categories
