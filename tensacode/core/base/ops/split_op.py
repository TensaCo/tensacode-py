from typing import Any, ClassVar
from typing_extensions import Self

from tensacode.core.base.base_engine import BaseEngine
from tensacode.internal.latent import LatentType
from tensacode.core.base.ops.base_op import Op
from tensacode.internal.utils.tc import loop_until_done


@BaseEngine.register_op()
def split(
    engine: BaseEngine,
    *inputs: list[Any],
    modify_steps: list[str] = [],
    **kwargs: Any,
) -> Any:
    """Split operation"""
    # Existing implementation
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
