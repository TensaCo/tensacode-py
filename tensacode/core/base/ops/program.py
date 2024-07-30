from typing import ClassVar, Any
from tensacode.core.base.base_engine import BaseEngine
from tensacode.core.base.ops.base_op import Op


@BaseEngine.register_op()
def program(
    engine: BaseEngine,
    prompt: Optional[Encoded[str]] = None,
    **kwargs: Any,
) -> Any:
    """
    Generate a program based on the provided prompt and engine's capabilities.

    This operation uses the engine to create a structured program or code based on the given prompt.

    Args:
        engine (BaseEngine): The engine used for program generation.
        prompt (Optional[Encoded[str]], optional): A prompt to guide the program generation process. Defaults to None.
        **kwargs: Additional keyword arguments to be passed to the engine.

    Returns:
        Any: The resulting program, which could be a string of code, a structured object representing code,
             or any other format suitable for representing a program.

    Raises:
        NotImplementedError: This method must be implemented by a subclass.

    Examples:
        >>> prompt = "Create a Python function to calculate the factorial of a number"
        >>> result = program(engine, prompt=prompt)
        >>> print(result)
        def factorial(n):
            if n == 0 or n == 1:
                return 1
            else:
                return n * factorial(n - 1)

        >>> prompt = "Generate a JavaScript class for a simple todo list"
        >>> result = program(engine, prompt=prompt)
        >>> print(result)
        class TodoList {
            constructor() {
                this.items = [];
            }

            addItem(task) {
                this.items.push(task);
            }

            removeItem(index) {
                this.items.splice(index, 1);
            }

            getItems() {
                return this.items;
            }
        }
    """
    raise NotImplementedError("Subclass must implement this method")
