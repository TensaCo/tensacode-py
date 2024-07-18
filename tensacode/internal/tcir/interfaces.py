class HasDependants(ABC):
    @property
    def dependants(self) -> list[Node]:
        return []


# class SupportsEncode(ABC):
#     @abstractmethod
#     def encode(self) -> str: ...


# class SupportsQuerying(ABC):
#     @abstractmethod
#     def query(self, *args, **kwargs) -> Any: ...


# class SupportsDecode(ABC):
#     @abstractmethod
#     def decode(self, value: str) -> Any: ...


# class SupportsIteration(ABC):
#     @abstractmethod
#     def __iter__(self) -> Iterable[Any]: ...


class SupportsHash(SupportsEncode, ABC):
    """Backup in case you can't encode anything"""

    @abstractmethod
    def __hash__(self) -> int: ...


class SupportsSerialization(ABC):
    @abstractmethod
    def serialize(self) -> str: ...


class SupportsDeserialization(ABC):
    @classmethod
    @abstractmethod
    def deserialize(cls, value: str) -> Any: ...


# Now, you will need to provide engine-specific implementations of
# all the essential Node types for each representation format
# (some Node classes are just there for convenience or can share classes)
