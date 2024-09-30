# TensaCode

## Concepts

- **Single Engine Class**: There is a single `Engine` class that manages all operations.
- **Operations Specify Latent Types**: `Operations` specify the latent type(s) they operate with.
- **Dynamic Operation Dispatching**: The `Engine`'s operation dispatcher function selects the appropriate operation based on the argument types and specified latent types.

```python
from typing import Any, Callable, Dict, List, Type, Union, get_type_hints
from functools import wraps

class LatentType:
    """Base class for latent representations."""
    pass

class TextLatent(LatentType):
    """Represents a text latent representation."""
    pass

class ImageLatent(LatentType):
    """Represents an image latent representation."""
    pass

class Operation:
    """Represents an operation that the Engine can perform."""

    def __init__(self, name: str, latent_types: List[Type[LatentType]], func: Callable, score_fn: Callable = None):
        self.name = name
        self.latent_types = latent_types
        self.func = func
        self.score_fn = score_fn or (lambda *args, **kwargs: 0)

    def matches(self, arg_types: List[Type[Any]]) -> bool:
        """Check if the operation matches the provided argument types."""
        # Simple example: match if any of the arg_types is compatible with the latent_types.
        for arg_type in arg_types:
            for latent_type in self.latent_types:
                if issubclass(arg_type, latent_type):
                    return True
        return False

class Engine:
    """The central Engine class that manages operations and execution."""

    _ops: Dict[str, List[Operation]] = {}

    @classmethod
    def register_op(cls, name: str, latent_types: List[Type[LatentType]], score_fn: Callable = None):
        def decorator(func: Callable):
            op = Operation(name=name, latent_types=latent_types, func=func, score_fn=score_fn)
            if name not in cls._ops:
                cls._ops[name] = []
            cls._ops[name].append(op)
            return func
        return decorator

    def get_op(self, name: str, *args, **kwargs) -> Operation:
        if name not in self._ops:
            raise ValueError(f"No operation named '{name}' is registered.")

        ops = self._ops[name]
        arg_types = [type(arg) for arg in args]
        best_op = None
        best_score = float('-inf')

        for op in ops:
            if op.matches(arg_types):
                score = op.score_fn(*args, arg_types=arg_types, **kwargs)
                if score > best_score:
                    best_score = score
                    best_op = op

        if best_op is None:
            raise ValueError(f"No suitable operation found for '{name}' with argument types {arg_types}.")

        return best_op

    def execute_op(self, name: str, *args, **kwargs):
        op = self.get_op(name, *args, **kwargs)
        return op.func(self, *args, **kwargs)

    # Example methods for common operations
    def encode(self, *args, **kwargs):
        return self.execute_op('encode', *args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.execute_op('decode', *args, **kwargs)

    def transform(self, *args, **kwargs):
        return self.execute_op('transform', *args, **kwargs)

# Define operations with their latent types
engine = Engine()

# Register 'encode' operation for text inputs
@Engine.register_op(name='encode', latent_types=[TextLatent])
def encode_text(engine: Engine, text: str):
    # Encoding logic for text
    print(f"Encoding text: {text}")
    return TextLatent()

# Register 'encode' operation for image inputs
@Engine.register_op(name='encode', latent_types=[ImageLatent])
def encode_image(engine: Engine, image_data: Any):
    # Encoding logic for images
    print(f"Encoding image data: {image_data}")
    return ImageLatent()

# Define a scoring function
def encode_score_fn(*args, arg_types: List[Type], **kwargs):
    # Prioritize operations based on argument types
    if issubclass(arg_types[0], str):
        return 10  # High score for text inputs
    elif issubclass(arg_types[0], bytes):
        return 5   # Lower score for bytes (e.g., image data)
    return 0

# Register 'decode' operation with a scoring function
@Engine.register_op(name='decode', latent_types=[TextLatent], score_fn=encode_score_fn)
def decode_text(engine: Engine, latent: TextLatent):
    # Decoding logic for text latents
    print("Decoding TextLatent")
    return "Decoded text"

# Usage examples
# Encoding text
text_latent = engine.encode("Hello, world!")

# Encoding image data
image_latent = engine.encode(b'\x89PNG\r\n\x1a\n...')  # Example binary data

# Decoding text latent
decoded_text = engine.decode(text_latent)
```

## Code Organization

```plaintext
tensacode
|- core
|  |- engine.py          # Contains the Engine class
|  |- operations.py      # Definitions of operations and their registrations
|  |- latent_types.py    # Definitions of latent types like TextLatent, ImageLatent, etc.
|- utils
|  |- misc.py            # Miscellaneous utility functions
|  |- locator.py         # Locator system for nested data access
|  |- language.py        # Language-related utilities
|- internal
|  |- tcir
|     |- nodes.py        # TCIR node definitions
|     |- parse.py        # Parsing logic for TCIR
|- examples
|  |- encoding_example.py  # Examples showcasing how to use the engine
|- tests
|  |- test_engine.py     # Unit tests for the Engine
|- __init__.py
```

## All Ops

- **encode**: Encode an object into a latent representation.
- **decode**: Decode a latent representation back into an object.
- **modify**: Modify an object.
- **transform**: Transform one or more inputs into a new form.
- **predict**: Predict the next item or value based on input sequence or data.
- **query**: Query an object for specific information.
- **correct**: Correct errors in the input data.
- **convert**: Convert between different types of objects.
- **select**: Select a specific value from a composite object.
- **similarity**: Compute similarity between two objects.
- **split**: Split an object into multiple components.
- **locate**: Locate a specific part of an input object.
- **plan**: Generate a plan based on provided prompts or context.
- **program**: Generate code or functions that can be executed.
- **decide**: Make a boolean decision based on input data.
- **call**: Call a function by obtaining all necessary arguments.

## All Latent Types

- **LatentType**: Base class for latent representations.
- **TextLatent**: Represents textual data in latent form.
- **ImageLatent**: Represents image data in latent form.
- **AudioLatent**: Represents audio data in latent form.
- **VideoLatent**: Represents video data in latent form.
- **VectorLatent**: Represents data as vector embeddings.
- **GraphLatent**: Represents graph structures in latent form.

## Your Task

- **Implement the Single `Engine` Class**: Centralize all engine functionalities into the `Engine` class as shown.
- **Define Operations with Latent Types**: Write operations specifying the latent types they operate with, and register them using the `@Engine.register_op` decorator.
- **Ensure Dynamic Operation Dispatching**: The engine's dispatcher should select the appropriate operation based on argument types and specified latent types.
- **Expand Latent Types and Operations**: Implement additional latent types and corresponding operations as needed.
- **Organize Code Accordingly**: Follow the code organization structure provided to maintain clarity and modularity.

---

By updating the architecture in this way, we align with the current understanding:

- **Simplification**: The architecture now revolves around a single `Engine` class.
- **Operations Specify Latent Types**: Operations declare the latent types they are compatible with, allowing for greater flexibility in handling different data types.
- **Dynamic Dispatching**: The engine selects the most appropriate operation based on the types of the arguments provided when executing an operation.

This design enhances flexibility, extensibility, and maintainability, allowing developers to add new operations and latent types without modifying the core engine logic.