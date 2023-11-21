"""Instantiate an object from a description and possibly a context."""

import traceback
from tensorcode.prompting.oracle import PromptingException

from tensorcode.prompting.prompt import prompt as _prompt

@tensorcode.export("tensorcode.prompting.create")
def create(prompt, context=None, **kwargs):
  """Instantiate an object from a description and possibly a context."""
  def _creates_something(expr):
    try: return eval(expr) is not None
    except: return False
  def _get_traceback(expr):
    try:
      eval(expr)
      raise PromptingException("No exception raised.")
    except Exception as e:
      if isinstance(e, PromptingException):
        raise e
      return traceback.format_exc()
    
  return _prompt(
    prompt=prompt,
    context=context,
    answer_format='any valid Python expression',
    is_valid=_creates_something,
    parser=eval,
    error_msg=_get_traceback,
    **kwargs)