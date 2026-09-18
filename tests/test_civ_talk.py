"""Speech: claims become sentences and sentences become claims again, losing something on the way."""

import numpy as np

from research.civ_sim import talk


def test_a_claim_survives_a_round_trip_in_the_same_dialect():
    for claim in (("village:Aldmere", "has_food", "much"), ("person:Kalo", "died", "starvation"),
                  ("village:Brenholt", "raided", "Coralin"), ("person:Miol", "hoards", "True"),
                  ("weather:local", "weather_coming", "snow"), ("sky:overhead", "omen", "solar")):
        sentence = talk.realize(claim, village=0)
        heard, conf, note = talk.parse(sentence, village=0, gain=1.0)
        assert heard is not None, sentence
        assert heard[1] == claim[1] and str(heard[2]).lower() == str(claim[2]).lower()
        assert conf > 0.5 and note == ""


def test_a_foreign_dialect_is_understood_but_trusted_less():
    claim = ("village:Aldmere", "has_food", "much")
    home = talk.parse(talk.realize(claim, village=0), village=0, gain=1.0)
    away = talk.parse(talk.realize(claim, village=3), village=0, gain=1.0)
    assert away[0] is not None and away[1] < home[1]
    assert "unfamiliar" in away[2]


def test_secondhand_is_marked_as_such():
    sentence = talk.realize(("village:Aldmere", "has_food", "little"), village=0, secondhand="Miol")
    heard, conf, note = talk.parse(sentence, village=0, gain=1.0)
    assert heard[1] == "has_food" and "secondhand via miol" in note


def test_a_lie_reverses_what_is_said():
    honest = talk.realize(("village:Aldmere", "has_food", "much"), village=0)
    lying = talk.realize(("village:Aldmere", "has_food", "much"), village=0, lie=True)
    assert honest != lying
    assert talk.parse(lying, village=0, gain=1.0)[0][2] == "none"


def test_nonsense_is_not_understood_rather_than_guessed():
    for junk in ("", "The wind is a wheel of hands.", "Ada im happy"):
        heard, conf, note = talk.parse(junk, village=0)
        assert heard is None and conf == 0.0 and note == "not understood"


def test_a_claim_degrades_as_it_passes_from_mouth_to_mouth():
    """Rumour drift: the same fact told down a chain of ten people, with noise and dialects."""
    rng = np.random.default_rng(0)
    start = ("village:Aldmere", "has_food", "much")
    intact = 0
    for _ in range(40):
        claim, hops = start, 0
        for hop in range(10):
            sentence = talk.realize(claim, village=int(rng.integers(0, 4)))
            heard, conf, note = talk.parse(sentence, village=int(rng.integers(0, 4)), gain=0.6, noise=0.3, rng=rng)
            if heard is None:
                break
            claim, hops = heard, hop + 1
        if hops == 10 and claim[2] == start[2]:
            intact += 1
    assert 0 < intact < 40  # some chains survive, some mutate: neither perfect nor pure noise
