"""Select N items from a list of items, possibly based on a context."""

import jinja2
from tensacode.prompting.oracle import PromptingException

from tensacode.prompting.prompt import prompt as _prompt

SELECTION_CONTEXT_TEMPLATE = jinja2.Template("""
{% if context is not None %}
{{ context }}
{% endif %}
{% for i, item in enumerate(items) %}
  {{ i }}: {{ item }}
{% endfor %}
""")

@tensorcode.export("tensorcode.prompting.select")
def select(items, N=1, context=None, **kwargs):
  """Select N items from a list of items, possibly based on a context.
  
  Args:
    items: A list of items to select from.
    N: The quantifier for items to select. May be an integer or a string.
      Eg, 1, 5, "several", "some" or "a few".
    context: The context to display to the oracle.
    kwargs: Keyword arguments to pass to the prompt function.
  """
  stringified_items = [str(item).lower() for item in items]
  def parse_item(item):
    item = item.strip(' "\'`')
    if item in stringified_items:
      return items[stringified_items.index(item)]
    if item.isdigit():
      if int(item) < len(items):
        return items[int(item)]
    raise PromptingException("Please enter an integer.")
  def parse_items(items):
    if not items:
      raise PromptingException("No items selected.")
    return [parse_item(item) for item in items.split(', ')]

  return _prompt(
    prompt=f'Please select {N} item(s) from the following list:',
    context=SELECTION_CONTEXT_TEMPLATE.render(
      context=context,
      items=stringified_items),
    answer_format=f'comma-separated list of integers 0-{len(items)-1}',
    parser=parse_items,
    error_msg='Please enter a valid selection.',
    **kwargs)