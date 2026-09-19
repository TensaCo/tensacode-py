"""Modifiers survive interpretation and realization without selecting one arbitrarily."""

import pytest

from tensorcode.language import ENGLISH, realize, understand, words
from tensorcode.language.deps_semantics import Reader
from tensorcode.language.grammar import Qualify, build_sem
from tensorcode.language.semantics import Entity


GRAMMAR = ENGLISH.extend(entries=[
    *words("large", "red", cat="Adj"),
    *words("stone", "garden", "wall", cat="N"),
])


@pytest.mark.parametrize("text,expected", [
    ("large red wall", (("amod", "large"), ("amod", "red"))),
    ("stone garden wall", (("compound", "stone"), ("compound", "garden"))),
    ("red stone wall", (("amod", "red"), ("compound", "stone"))),
    ("red red wall", (("amod", "red"), ("amod", "red"))),
])
def test_grammar_retains_relation_order_and_repeated_values_and_realizes_them(text, expected):
    meaning = understand(GRAMMAR, text, starts=("NP",)).meanings[0]
    assert meaning.features["modifiers"] == expected
    assert "quality" not in meaning.features
    assert "name" not in meaning.features
    said = realize(GRAMMAR, meaning, cat="NP", depth=12)
    assert said == text
    reread = understand(GRAMMAR, said, starts=("NP",)).meanings[0]
    assert reread.features == meaning.features


@pytest.mark.parametrize("text,relation,value", [
    ("red wall", "amod", "red"), ("stone wall", "compound", "stone"),
])
def test_single_grammar_modifier_retains_compatibility_feature(text, relation, value):
    meaning = understand(GRAMMAR, text, starts=("NP",)).meanings[0]
    assert meaning.features["quality"] == value
    assert meaning.features["modifiers"] == ((relation, value),)
    assert realize(GRAMMAR, meaning, cat="NP") == text


def test_qualify_does_not_replace_an_earlier_name_or_reintroduce_a_scalar():
    spec = Qualify(1, features_from=(("name", 0),), extend_text_from=(0,))
    result = Entity("description", "wall", {"noun": "wall"})
    for name in ("third", "second", "first"):
        result = build_sem(spec, [name, result], [(name,), (result.text,)], [{}, {}])
    assert result.features["modifiers"] == (("name", "first"), ("name", "second"), ("name", "third"))
    assert "name" not in result.features


def entity(words_, tags, heads_, relations):
    reader = Reader({})
    heads = dict(enumerate(heads_, 1))
    labels = dict(enumerate(relations, 1))
    return reader.entity(len(words_), words_, tags, words_, heads, labels, reader.children(heads))


def test_dependency_reader_retains_multiple_adjectives_and_compounds():
    result = entity(["large", "red", "stone", "garden", "wall"],
                    ["ADJ", "ADJ", "NOUN", "NOUN", "NOUN"],
                    [5, 5, 5, 5, 0], ["amod", "amod", "compound", "compound", "root"])
    modifiers = result.features["modifiers"]
    assert modifiers[:2] == (("amod", "large"), ("amod", "red"))
    assert [(relation, value.text) for relation, value in modifiers[2:]] == [
        ("compound", "stone"), ("compound", "garden")]
    assert all(value.kind == "description" for _, value in modifiers[2:])
    assert "quality" not in result.features
    assert "name" not in result.features


def test_dependency_reader_retains_nested_compounds_and_single_alias():
    result = entity(["stone", "garden", "wall"], ["NOUN"] * 3,
                    [2, 3, 0], ["compound", "compound", "root"])
    relation, garden = result.features["modifiers"][0]
    assert relation == "compound"
    assert garden.text == "stone garden"
    assert garden.kind == "description"
    assert result.features["name"] == garden
    nested_relation, stone = garden.features["modifiers"][0]
    assert nested_relation == "compound" and stone.text == "stone"


def test_dependency_single_adjective_retains_compatibility():
    result = entity(["red", "wall"], ["ADJ", "NOUN"], [2, 0], ["amod", "root"])
    assert result.features["quality"] == "red"
    assert result.features["modifiers"] == (("amod", "red"),)
