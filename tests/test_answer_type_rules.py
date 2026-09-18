"""The expected-answer-type representation: what it claims a question asks for, and what it rejects.

Every case here is one the rules got wrong at some point while they were being tuned on the
HotpotQA train split, so each test names a real regression rather than a restatement of the code.
"""

from __future__ import annotations

import pytest

from tensorcode.answer_type import (
    AnswerType,
    Shape,
    asked_for,
    contains_words,
    could_be,
    from_question,
    mismatch,
    shape,
)


class TestAskedFor:
    def test_plain_interrogatives(self):
        assert asked_for("Who directed Jaws?") is AnswerType.person
        assert asked_for("Where is Aston Villa based?") is AnswerType.place
        assert asked_for("When was the club founded?") is AnswerType.date
        assert asked_for("How many teams competed?") is AnswerType.number

    def test_head_noun_carries_the_requirement(self):
        assert asked_for("What year was it founded?") is AnswerType.date
        assert asked_for("Which city hosted the games?") is AnswerType.place
        assert asked_for("What director made it?") is AnswerType.person

    def test_english_fronts_the_interrogative(self):
        """A later wh may be a relative clause describing the hop, not the ask."""
        q = "Who is the chancellor of the school where the 1994 ceremony was held?"
        assert asked_for(q) is AnswerType.person, "the fronted 'who' asks; 'where' describes the hop"

    def test_a_trailing_ask_is_read_when_nothing_is_fronted(self):
        q = "The team, in which Ossie Asmundson played, was founded in what year?"
        assert asked_for(q) is AnswerType.date

    def test_year_old_is_an_age_modifier_not_a_date(self):
        """'What 43-year-old actress' asks for a person; the bare 'year' fooled an earlier version."""
        assert asked_for("What 43-year-old actress co-hosted the awards?") is AnswerType.person
        assert asked_for("What 19-year MLB veteran was on the cover?") is AnswerType.entity

    def test_no_interrogative_means_yes_no_or_unspecified(self):
        assert asked_for("Are both directors British?") is AnswerType.yes_no
        assert asked_for("The film was released in 1994.") is AnswerType.entity


class TestCouldBe:
    def test_dates_outside_the_common_era_range(self):
        assert could_be("around 8000 BC", AnswerType.date)
        assert could_be("161 to 180 AD", AnswerType.date)
        assert could_be("983", AnswerType.date)

    def test_numbers_in_words(self):
        assert could_be("thirteen teams", AnswerType.number)
        assert could_be("twelfth", AnswerType.number)

    def test_yes_no_is_exact(self):
        assert could_be("yes", AnswerType.yes_no)
        assert not could_be("British", AnswerType.yes_no)

    def test_a_bare_number_may_name_a_thing(self):
        """Ferrari's '458', an area code '284', a South Park episode '201', the single '212'."""
        for want in (AnswerType.entity, AnswerType.person, AnswerType.place):
            assert could_be("458", want), "rejecting bare numbers here cost correct answers on train"

    def test_unknown_kinds_are_admitted(self):
        assert could_be("Ernst Messerschmid", AnswerType.person)
        assert could_be("Ernst Messerschmid", AnswerType.place), "no gazetteer, so no rejection"


class TestContainsWords:
    def test_word_runs_not_characters(self):
        """'no' is a character substring of 'northeastern Ontario' and must not match."""
        assert not contains_words("Strathy Township of Temagami, Northeastern Ontario", "no")
        assert contains_words("the answer is no", "no")

    def test_multiword_runs(self):
        assert contains_words("Which came first, Muppet Treasure Island or Million Dollar Arm?",
                              "Muppet Treasure Island")
        assert not contains_words("Muppet Treasure Island", "Treasure Island Hotel")

    def test_normalisation_ignores_case_and_punctuation(self):
        assert from_question("muppet treasure island!", "Which is older, Muppet Treasure Island?")


class TestShape:
    def test_named_alternatives_are_comparisons(self):
        assert shape("Which came first, A or B?") is Shape.comparison
        assert shape("Are both Marray and Black British?") is Shape.comparison
        assert shape("Who is older, X or Y?") is Shape.comparison

    def test_a_travelled_question_is_a_bridge(self):
        assert shape("Who replaced the manager who began at Leeds United?") is Shape.bridge


class TestMismatch:
    def test_a_comparison_may_answer_itself_from_its_own_words(self):
        q = "Which movie came out first, Muppet Treasure Island or Million Dollar Arm?"
        assert mismatch(q, "Muppet Treasure Island") is None

    def test_a_bridge_answer_lifted_from_the_question_is_rejected(self):
        q = "Who replaced the Aston Villa manager who began his career at Leeds United?"
        got = mismatch(q, "Leeds United")
        assert got is not None and got.reason == "taken_from_the_question"

    def test_a_yes_no_comparison_with_no_alternative_is_checked(self):
        """218 of 220 such questions on train take a yes or a no."""
        got = mismatch("Are both Jonathan Marray and Wayne Black British?", "British")
        assert got is not None and got.reason == "wrong_answer_type"

    def test_an_either_or_comparison_is_not_yes_no_checked(self):
        q = "Does East of Chicago Pizza or Your Pie have more restaurants?"
        assert mismatch(q, "East of Chicago Pizza") is None

    def test_a_date_question_rejects_a_name(self):
        got = mismatch("In what year was the club founded?", "Aston Villa")
        assert got is not None and got.reason == "wrong_answer_type"

    def test_an_empty_candidate_is_not_a_mismatch(self):
        assert mismatch("Who directed it?", "") is None

    @pytest.mark.parametrize("question,answer", [
        ("Who directed Jaws?", "Steven Spielberg"),
        ("In what year did it open?", "1994"),
        ("How many teams competed?", "thirteen"),
        ("Where was he born?", "Kansas City Metropolitan Area"),
        ("What is the area code of the British Overseas Territory?", "284"),
    ])
    def test_correct_answers_are_admitted(self, question, answer):
        assert mismatch(question, answer) is None
