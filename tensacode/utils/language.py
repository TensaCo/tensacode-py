from typing import Literal


Language = Literal["python", "typescript", "chinese", "english"]

LANGUAGE_SHORTHANDS = {
    "py": "python",
    "ts": "typescript",
    "zh": "chinese",
    "cn": "chinese",
    "中文": "chinese",
    "eng": "english",
    "en": "english",
}


def parse_language(lang: str) -> Language:
    if lang in LANGUAGE_SHORTHANDS:
        return LANGUAGE_SHORTHANDS[lang]
    assert isinstance(lang, Language), f"Invalid language: {lang}"
    return lang
