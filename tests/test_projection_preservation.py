"""Resolving a phrase's identity must not silently erase its qualifications."""

import pytest

from tensorcode.language.semantics import Entity, Frame, to_propositions
from tensorcode.records import Proposition, Ref


@pytest.fixture(autouse=True)
def reporting_domains(monkeypatch):
    # These are projection tests, independent of an installed WordNet corpus.
    from tensorcode.language import wordnet

    monkeypatch.setattr(wordnet, "verb_domains", lambda: {"say": ("verb.communication",)})


def project(frame, **kwargs):
    return to_propositions(frame, source=Ref("agent:user"), **kwargs)


def test_counted_entity_keeps_identity_and_reports_missing_count():
    plants = Entity("name", "plants", {"count": 3}, ref=Ref("collection:plants"))
    [(proposition, evidence)], dropped = project(Frame("have", {"object": plants}))

    assert proposition.roles == {"object": Ref("collection:plants")}
    assert evidence.source == Ref("agent:user")
    assert dropped == [
        "entity feature not explicitly projected: clause[0].roles.object.features.count"
    ]


def test_plain_entity_identity_does_not_report_spurious_loss():
    [(proposition, _)], dropped = project(Frame("arrive", {"subject": Entity("name", "Alice")}))

    assert proposition.roles["subject"] == Ref("entity:Alice")
    assert dropped == []


def test_nested_reported_frame_and_coordinated_entities_have_precise_paths():
    content = Frame("arrive", {"subject": (
        Entity("name", "plants", {"count": 3}),
        Entity("name", "seeds", {"count": 2}),
    )})
    [(proposition, _)], dropped = project(Frame("say", {"content": content}))

    assert isinstance(proposition.roles["content"], Proposition)
    assert proposition.roles["content"].roles["subject"] == (
        Ref("entity:plants"), Ref("entity:seeds"),
    )
    assert dropped == [
        "entity feature not explicitly projected: clause[0].roles.content.roles.subject[0].features.count",
        "entity feature not explicitly projected: clause[0].roles.content.roles.subject[1].features.count",
    ]


def test_entities_nested_in_discarded_features_are_accounted_for():
    owner = Entity("name", "gardeners", {"count": 2})
    plants = Entity("name", "plants", {"possessor": owner})
    [(proposition, _)], dropped = project(Frame("grow", {"subject": plants}))

    assert proposition.roles == {"subject": Ref("entity:plants")}
    assert dropped == [
        "entity feature not explicitly projected: clause[0].roles.subject.features.possessor",
        "entity feature not explicitly projected: clause[0].roles.subject.features.possessor.features.count",
    ]


def test_custom_identity_resolution_does_not_certify_feature_preservation():
    entity = Entity("name", "plants", {"count": 3})
    [(proposition, _)], dropped = project(
        Frame("grow", {"subject": entity}), resolve=lambda _: Ref("known:plants"),
    )

    assert proposition.roles["subject"] == Ref("known:plants")
    assert len(dropped) == 1
    assert dropped[0].endswith(".features.count")
