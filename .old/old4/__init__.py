"""TensorCode

Copyright (c) 2022 Jacob F Valdez. Released under the MIT License

TODO: introduction
"""

from tensorcode.models.model import Model
from tensorcode.operations.operation import Operation
from tensorcode.utils.export_helpers import make_export_helpers
from tensorcode.utils.registry import Registry
from tensorcode.utils.stacked import stacked

# directly specify configs in their native packages
import openai
import forefront
import transformers


encode: Encode
decode: Decoder
select: Select

current_model: Model



registry: Registry
'''
models(task, uri):
  completion/edit/image_generation/inpainting/outpainting/translation/summarization/etc.:
    Task-specific subregistries wrap the native models into a common interface, eg, CompletionModel, ImageGenerationModel, etc.
    openai: (actually lookup all of these jit)
      gpt-3: davinci-001, davinci-text-002, codex-001, etc.
      dalle: dalle-001, dalle-002, etc.
    forefront: (actually lookup all of these jit)
      gpt-j: 6b, 100M, etc.
    stability-ai: (actually lookup all of these jit)
      ...
    transformers: (actually lookup all of these jit)
      ...lookup JIT
    https://url/to/oracle:
'''




DEFAULT_MODEL = DefaultModel()
CURRENT_MODEL = DEFAULT_MODEL

register, deregister, _stack = make_export_helpers(__all__, __dict__)


# todo, somehow link to the current top-level model, but don't try to rewrite the signature
# save, load, train, reward, add_loss, encode, decode, select
# basically, whatever Model method have an @model_export annotation
save = _stack['CURRENT_MODEL'][-1].save




__all__ = ['CURRENT_MODEL', ]
