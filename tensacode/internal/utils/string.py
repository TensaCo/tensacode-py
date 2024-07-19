import re


def to_capital_camel_case(string):
    """
    Convert a string to CapitalCamelCase.

    Args:
        string (str): The input string to convert.

    Returns:
        str: The string converted to CapitalCamelCase.
    """
    # Split the string into words, considering capital letters and non-alphanumeric characters
    words = re.findall(r"[A-Z]?[a-z0-9]+|[A-Z]+(?=[A-Z][a-z0-9]|\d|\W|$)|\d+", string)

    # Capitalize the first letter of each word and join them
    return "".join(word.capitalize() for word in words)
