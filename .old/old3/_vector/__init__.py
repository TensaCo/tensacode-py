from old3._internal.raise_namespace import (
    import_all_from_public_submodules_recursively,
)
from old3._vector.vector_model import VectorModel


import_all_from_public_submodules_recursively()

VectorModel.__dict__.update(__dict__)
