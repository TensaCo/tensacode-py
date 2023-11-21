# reuse import to conserve api keys
from tensacode import openai


def query_gpt3(prompt: str, context=None) -> str:

    input = prompt
    if context:
        input = f"{context}\n\n{input}"

    output = openai.Completion.create(
        engine="text-davinci-002", prompt=input, temperature=0.2, max_tokens=10
    )

    print(input, output.choices[0].text)

    return output.choices[0].text


registry.register("completion", "gpt-3", query_gpt3)
