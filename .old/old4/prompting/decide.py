"""Decide a boolean value based on a prompt and possibly a context."""

from tensorcode.prompting.prompt import prompt as _prompt

@tensorcode.export("tensorcode.prompting.decide")
def decide(prompt, context=None, **kwargs):
  """Decide a boolean value based on a prompt and possibly a context."""
  return _prompt(
    prompt=prompt,
    context=context,
    answer_format='yes/no',
    is_valid=lambda x: x in ['yes', 'no'],
    parser=lambda x: x == 'yes',
    error_msg='Please answer "yes" or "no".',
    **kwargs)