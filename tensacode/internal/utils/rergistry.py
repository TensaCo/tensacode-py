from tensacode.core.base.ops.base_op import BaseOp
from tensacode.internal.utils.misc import advanced_equality_check
from tensacode.core.base.base_engine import BaseEngine
from tensacode.core.base.base_session import BaseSession

from typing import Optional, Annotated
from pydantic import UUID4, BaseModel
from typing import Protocol, Any
from tensacode.core.base.base_engine import BaseEngine
from tensacode.core.base.ops.base_op import BaseOp
from tensacode.internal.utils.pydantic import SubclassField

from pydantic import BaseModel, Field
from typing import Generic, TypeVar, Any, Optional, ClassVar, Callable

T = TypeVar("T")


class Registry(Generic[T], BaseModel):
    """
    A registry class that stores and retrieves items based on a customizable equality measure.

    Attributes:
        _registry (list): A list to store the registered values.
        _comparison_fn (Callable): The function used for comparing items.

    Methods:
        register(value): Register a value in the registry.
        get(key): Retrieve a value from the registry based on the comparison function.
        __contains__(key): Check if a matching value exists in the registry.
        __len__(): Get the number of items in the registry.
    """

    _registry: list[T] = Field(default_factory=list, alias="registry")
    _comparison_fn: Callable[[T, T], bool] = Field(
        default=advanced_equality_check, alias="comparison_fn"
    )

    def __init__(self, comparison_fn: Optional[Callable[[T, T], bool]] = None, **data):
        super().__init__(**data)
        if comparison_fn is not None:
            self._comparison_fn = comparison_fn

    def register(self, value: T):
        """
        Register a value in the registry.

        Args:
            value: The value to be registered.
        """
        self._registry.append(value)

    def get(self, key: T) -> Optional[T]:
        """
        Retrieve a value from the registry based on the comparison function.

        Args:
            key: The key to search for.

        Returns:
            The matching value from the registry, or None if not found.
        """
        for item in self._registry:
            if self._comparison_fn(key, item):
                return item
        return None

    def __contains__(self, key: T) -> bool:
        """
        Check if a matching value exists in the registry.

        Args:
            key: The key to check for.

        Returns:
            bool: True if a matching value exists in the registry, False otherwise.
        """
        return self.get(key) is not None

    def __len__(self) -> int:
        """
        Get the number of items in the registry.

        Returns:
            int: The number of items in the registry.
        """
        return len(self._registry)


class IsSelector(BaseModel):
    engine_type: SubclassField(BaseEngine) | None
    engine_id: str | None
    run_id: UUID4 | None
    operator_type: SubclassField(BaseOp) | None
    operator_id: str | None
    object_type: SubclassField(Any) | None
    object: Any | None
    latent_type: SubclassField(LatentType) | None


class HasRegistry(Generic[T], IsSelector, BaseModel):
    _registry: ClassVar[Registry[T]] = Registry[T]()

    @classmethod
    def set_comparison_fn(cls, comparison_fn: Callable[[T, T], bool]):
        cls._registry = Registry[T](comparison_fn=comparison_fn)

    @classmethod
    def register(cls, value: T):
        cls._registry.register(value)
        return cls

    @classmethod
    def get(cls, key: T) -> Optional[T]:
        return cls._registry.get(key)
