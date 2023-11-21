from textwrap import dedent
from typing import Callable, List, Tuple, Union

import jinja2

from tensorcode.prompting import Oracle
from tensorcode.prompting.oracle import PromptingException


PROMPT_TEMPLATE = jinja2.Template(
  r'{{ prompt }}{% if answer_format %}({{ answer_format }}):{% endif %}')

PROMPT_WITH_CONTEXT_TEMPLATE = jinja2.Template(dedent(r'''\
  {% if context %}{{ context }}{% endif %}

  {0}''' % PROMPT_TEMPLATE.source))

PROMPT_WITH_CONTEXT_AND_EXAMPLES_TEMPLATE = jinja2.Template(dedent(
  """\
  {% for example in examples %}
  {{ example[0] }}

  {0} {{ example[1] }}
  {% endfor %}
  {1}""" % (PROMPT_TEMPLATE.source, PROMPT_WITH_CONTEXT_TEMPLATE.source)))

ERROR_TEMPLATE = jinja2.Template(dedent(r'''\
  {{conversation}}{{ answer }}

  {{ error_msg }}

  {0}''' % PROMPT_TEMPLATE.source))

def prompt(
  prompt: str,
  context: str = None,
  oracle: Oracle = input,
  answer_format: str = None,
  examples: List[Tuple[str,str]] = None,
  sanitizer: Callable[[str], str] = lambda x: x.strip().lower(),
  is_valid: Callable[[str], bool] = lambda x: len(x.split()) == 1,
  parser: Callable[[str], any] = str,
  error_msg: Union[str, Callable[[str], str]] = 'Too many words. Please try again.',
  ) -> bool:
  """ Prompt the oracle for an answer.
  
  Args:
    prompt: The prompt to display to the oracle.
    context: The context to display to the oracle.
    oracle: The oracle to ask for an answer.
    answer_format: The format of the answer.
    examples: A list of examples to display to the oracle.
    sanitizer: A function to sanitize the answer.
    is_valid: A function to test if the answer is valid. Optional.
    parser: A function to parse the answer. Should raise error 
      if the answer is invalid. Optional.
    error_msg: The error message to display to the oracle.
  Returns:
    The parsed answer.
  """

  input = PROMPT_WITH_CONTEXT_AND_EXAMPLES_TEMPLATE.render(
    prompt=prompt,
    context=context,
    examples=examples,
    answer_format=answer_format)

  for _ in range(3):
    output = oracle(input)
    output = sanitizer(output)
    try:
      if is_valid(output):
        return parser(output)
      else:
        raise PromptingException('Invalid answer.')
    except PromptingException as e:
      pass
    input += ERROR_TEMPLATE.render(
      error_msg=error_msg 
        if isinstance(error_msg, str) 
        else error_msg(output),
      prompt=prompt,
      answer_format=answer_format)
  raise PromptingException('Too many failed attempts to answer.')