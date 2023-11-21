from old3._internal.raise_namespace import (
    import_all_from_public_submodules_recursively,
)
from old3._text.text_model import TextModel


import_all_from_public_submodules_recursively()

TextModel.__dict__.update(__dict__)
