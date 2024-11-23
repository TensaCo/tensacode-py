from abc import abstractmethod
from pydantic import BaseModel

class PromptTemplate(str):
    @abstractmethod
    def __str__(self) -> str:
        pass

class LinearPromptTemplate(PromptTemplate):
    sequences: list[str]

class NonlinearPromptTemplate(PromptTemplate):
    sequences: list[str]

def optimize_prompt_template(
    input_text: str,
    text2text_model: str,
    eval_fn: Callable[[str], float],
):
    pass