from typing import Protocol


class Oracle(Protocol):
  def __call__(self, prompt: str) -> str:
    ...

# TODO I need to use these in the code.
class PromptingException(Exception):
  """Base class for all prompting exceptions."""

class ResponseParsingException(PromptingException):
  """Raised when a response could not be parsed."""

class PromptDisobedianceException(PromptingException):
  """Raised when a prompt is disobeyed."""

class PromptClarificationException(PromptingException):
  """Raised to clarify a prompt."""