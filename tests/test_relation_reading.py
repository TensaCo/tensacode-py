"""Reading a comparison off a question: the relation, the two candidates, the attribute.

Every case here is one the surface rules got wrong while they were being written against the
HotpotQA train split, so each test names a real regression. The grammar in
:mod:`tensorcode.language` parses these questions but does not carry a coordination of two
candidates or a comparative relation in its frames, which is why these rules exist at all.
"""

from __future__ import annotations

from tensorcode.answer_type import AnswerType
from tensorcode.outcomes import Unknown
from tensorcode.relation import Relation, digitise, read


class TestRelation:
    def test_order_before_magnitude(self):
        # "older" is an age comparison answered from dates, not from a quantity
        assert read("Who is older, Bill Clinton or George Bush?").relation is Relation.earlier
        assert read("Which film was released first, Alien or Tron?").relation is Relation.earlier
        assert read("Which band formed most recently, Muse or Blur?").relation is Relation.later

    def test_magnitude(self):
        assert read("Which band has more members, Muse or Blur?").relation is Relation.greater
        assert read("Which city has a smaller population, Reno or Tulsa?").relation is Relation.less

    def test_yes_no_and_shared(self):
        assert read("Were Ed Wood and Tim Burton of the same nationality?").relation is Relation.same
        assert read("Are both Muse and Blur English rock bands?").relation is Relation.both
        assert read("What profession do Tyson and Ali have in common?").relation is Relation.shared

    def test_no_relation_is_a_refusal_not_a_guess(self):
        out = read("Which documentary is about Finnish rock groups, Promise or Telluride?")
        assert isinstance(out, Unknown) and out.reason == "relation_unsupported"


class TestCandidates:
    def test_coordination_supplies_the_pair(self):
        got = read("Who was born first, Pablo Trapero or Aleksander Ford?")
        assert got.candidates == ("Pablo Trapero", "Aleksander Ford")
        assert got.attribute == "born" and got.wants is AnswerType.person

    def test_leading_interrogative_is_never_a_candidate(self):
        # "Which" is a capitalised run of the same length as "Alien" and was winning the tie
        assert read("Which film is longer, Alien or Tron?").candidates == ("Alien", "Tron")

    def test_a_name_does_not_end_in_a_connector(self):
        assert read("Were Scott Derrickson and Ed Wood of the same nationality?").candidates == (
            "Scott Derrickson", "Ed Wood")

    def test_titles_rescue_a_greedy_run(self):
        # "Kings of Leon American" is one capitalised run; the title says where the name ends
        got = read("Are both The New Pornographers and Kings of Leon American rock bands?",
                   ("The New Pornographers", "Kings of Leon"))
        assert got.candidates == ("The New Pornographers", "Kings of Leon")

    def test_unreadable_pair_refuses(self):
        out = read("Are both of them the same?")
        assert isinstance(out, Unknown) and out.reason == "candidates_unclear"


class TestIsAComparisonAtAll:
    def test_a_bridge_question_containing_first_is_not_a_comparison(self):
        out = read("Who was the first president of the United States and what did he found?")
        assert isinstance(out, Unknown) and out.reason == "not_a_comparison"

    def test_in_common_counts_as_a_marker_even_though_shape_misses_it(self):
        from tensorcode.answer_type import Shape, shape
        q = "What profession do Mike Tyson and Muhammad Ali have in common?"
        assert shape(q) is Shape.bridge  # no " or ", no comparative cue
        assert not isinstance(read(q), Unknown)


class TestDigitise:
    def test_prose_spells_small_numbers_out(self):
        # the quantity reader used here is digit-only; without this, half the magnitude
        # comparisons refuse for a missing value that is written in the sentence
        assert digitise("a band with four members") == "a band with 4 members"
        assert digitise("a duo") == "a 2"
