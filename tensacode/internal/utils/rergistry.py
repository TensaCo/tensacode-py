from tensacode.core.base.ops.base_op import BaseOp
from tensacode.internal.utils.misc import advanced_equality_check
from tensacode.core.base.base_engine import BaseEngine
from tensacode.core.base.base_session import BaseSession


from pydantic import BaseModel, Field
from typing import Generic, TypeVar, Any, Optional, ClassVar

T = TypeVar("T")


class Registry(Generic[T], BaseModel):
    """
    A registry class that retrieves items based on the advanced equality measure for keys.

    This class allows storing and retrieving objects using the advanced_equality_check
    function for comparison of keys, while allowing arbitrary objects as values.

    Attributes:
        _registry (dict): A dictionary to store the registered key-value pairs.

    Methods:
        register(key, value): Register a key-value pair in the registry.
        get(key): Retrieve a value from the registry based on advanced equality of keys.
        __contains__(key): Check if a key exists in the registry.
        __len__(): Get the number of items in the registry.

    Examples:
        >>> class TestModel(BaseModel):
        ...     a: int
        ...     b: str
        >>> registry = AdvancedEqualityRegistry[TestModel]()
        >>> registry.register(TestModel(a=1, b='test'), 'a')
        >>> registry.register(TestModel(a=1, b='test'), 'b')
        >>> registry.register(TestModel(a=2, b='test'), 'c')
        >>> registry.get(TestModel(a=1, b='test'))
        'b'
        >>> registry.get(TestModel(a=2, b='test'))
        'c'
        >>> registry.get(TestModel(a=4, b=None))
        None
        >>> registry.get(TestModel(a=None, b='test'))
        'c'
    """

    _registry: dict[T, Any] = Field(default_factory=dict)

    def register(self, key: T, value: Any):
        """
        Register a key-value pair in the registry.

        Args:
            key: The key to be registered.
            value: The value associated with the key.
        """
        self._registry[key] = value

    def get(self, key: T) -> Optional[Any]:
        """
        Retrieve a value from the registry based on advanced equality of keys.

        Args:
            key: The key to search for.

        Returns:
            The matching value from the registry, or None if not found.
        """
        for registered_key, value in self._registry.items():
            if advanced_equality_check(key, registered_key):
                return value
        return None

    def __contains__(self, key: T) -> bool:
        """
        Check if a key exists in the registry.

        Args:
            key: The key to check for.

        Returns:
            bool: True if the key exists in the registry, False otherwise.
        """
        return self.get(key) is not None

    def __len__(self) -> int:
        """
        Get the number of items in the registry.

        Returns:
            int: The number of items in the registry.
        """
        return len(self._registry)


class HasRegistry(Generic[T]):
    _registry: ClassVar[Registry[T]] = Registry[T]()

    @classmethod
    def register(cls, key):
        def decorator(cls):
            cls._registry.register(key, cls)
            return cls

        return decorator

    @classmethod
    def get(cls, key: T) -> Optional[T]:
        return cls._registry.get(key)
