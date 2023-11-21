class PromptingException(Exception):
    """Base class for all prompting exceptions."""

    msg = "An error occurred while prompting. Please try again."

    def __init__(self, msg):
        super().__init__(msg)


class NoFinalAnswerException(PromptingException):
    """Raised when the response does not contain an answer."""

    mgs = "Your response did not contain an answer. Please provide an answer below:"


class AnswerParsingException(PromptingException):
    """Raised when an answer could not be parsed from the prompt."""

    mgs = "We had an error while parsing your answer. Please make sure you have formatted it correctly. If you are still having trouble, try making random changes to your answer until it works."


class PromptDisobedianceException(PromptingException):
    """Raised when a prompt is disobeyed."""

    mgs = "You disobeyed the prompt. Please try again."
