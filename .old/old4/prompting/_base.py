from __future__ import annotations

from typing import List, Dict,  Any, Callable, Union, Optional, Tuple, Protocol,TypeVar

from dataclasses import dataclass
from textwrap import dedent

from tensorcode.prompting.oracles.oracle import Oracle

decide
select
sort
create
modify
call

def decide




def smart_string(
    prompt: str,
    context: str = None,
    oracle: Oracle = input,
    answer_format: str = 'one-word answer',
    test_fn: Callable[[str], bool] = lambda x: len(x.split()) == 1,
    error_msg: str = 'Too many words. Please try again.',
    ) -> bool:

  input = prompt
  if context:
    input = f'{context}\n\n{prompt}'

  for _ in range(3):
    input += f' ({answer_format}): '
    output = oracle(input).strip().lower()
    if test_fn(output):
      return output
    else:
      input += f'\n\n{output}\n\n{error_msg}'
  raise Exception('Too many failed attempts to answer.')

def smart_bool(
    prompt: str,
    context: str = None,
    oracle: Oracle = input
    ) -> bool:

  return smart_string(
    prompt,
    context,
    oracle,
    answer_format='yes/no',
    test_fn=lambda x: x in ['yes', 'no'],
    error_msg='Please answer "yes" or "no".'
  ) == 'yes'

def smart_choice(
    prompt: str,
    options: Union[List[str],Dict[str, object]],
    context: str = None,
    oracle: Oracle = input) -> str:

  if isinstance(options, list):
    options = {str(option): option for option in options}

  index = smart_string(
    prompt + '\n'.join(f'{i}. {key}' for i, key in enumerate(options.keys())),
    context,
    oracle,
    answer_format=f'enter a number 1-{len(options)}',
    test_fn=lambda x: x.isdigit() and int(x) in range(1, len(options)+1),
    error_msg='Please enter a valid number.'
  )
  key = list(options.keys())[int(index)-1]
  return options[key]