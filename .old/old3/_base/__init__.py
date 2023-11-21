from old3._internal.raise_namespace import (
    import_all_from_public_submodules_recursively,
)
from old3._base.base_model import BaseModel


import_all_from_public_submodules_recursively()

BaseModel.__dict__.update(__dict__)
