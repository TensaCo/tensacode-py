from old3._internal.context_managed_singleton_proxy import (
    LazyContextManagedSingletonProxy,
)
from old3._internal.initialize_on_first_getattr import initialize_on_first_getattr
import old3._text as _text
import old3._vector as _vector


text = LazyContextManagedSingletonProxy(_text.TextModel, args=(), kwargs={})
vector = LazyContextManagedSingletonProxy(_vector.VectorModel, args=(), kwargs={})

__all__ = [
    "text",
    "vector",
]
