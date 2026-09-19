"""Visual structure remains a source-bound, queryable interpretation proposal."""
from dataclasses import dataclass

import pytest

from tensorcode.agent.scene import SceneGraph, SceneProposal, VisualAnchor
from tensorcode.outcomes import Score
from tensorcode.records import Proposition, Ref, Var


IMAGE = Ref("image:kitchen")
PERSON = Ref("entity:person")
CUP = Ref("entity:cup")
EVENT = Ref("event:pouring")
CONTEXT = Ref("hypothesis:breakfast")


def test_whole_scene_relations_events_and_unknown_predicates_share_one_graph():
    graph = SceneGraph(
        IMAGE,
        (PERSON, CUP, EVENT),
        (
            Proposition("atmosphere", {"scene": IMAGE, "quality": "hurried"}),
            Proposition("pouring", {"event": EVENT, "agent": PERSON, "recipient": CUP}),
            Proposition("occludes", {"front": PERSON, "back": CUP}),
            Proposition("domain_specific_pattern_42", {"participants": [PERSON, CUP]}),
        ),
        (VisualAnchor(IMAGE), VisualAnchor(PERSON, (.1, .1, .5, .9))),
        ("The contents of the cup are not visible.",),
    )
    assert graph.match(Proposition("pouring", {"agent": Var("who"), "recipient": CUP})) == (
        {"who": PERSON},
    )
    assert graph.match(Proposition("atmosphere", {"scene": IMAGE})) == ({},)
    assert graph.limitations == ("The contents of the cup are not visible.",)


def test_match_preserves_negation_modality_nested_roles_and_explicit_scopes():
    content = Proposition("touching", {"agent": PERSON, "object": CUP}, polarity=False)
    graph = SceneGraph(IMAGE, (PERSON, CUP, CONTEXT), (
        Proposition("suggests", {"scene": IMAGE, "content": content},
                    modality="hypothesised", scope=CONTEXT),
    ))
    pattern = Proposition("suggests", {"content": Proposition(
        "touching", {"object": Var("target")}, polarity=False,
    )}, modality="hypothesised", scope=CONTEXT)
    assert graph.match(pattern) == ({"target": CUP},)
    assert graph.match(Proposition("suggests")) == ()
    assert graph.match(Proposition("suggests", modality="hypothesised",
                                  scope=Ref("hypothesis:other"))) == ()
    # An omitted scope follows records.matches' wildcard semantics.
    assert graph.match(Proposition("suggests", modality="hypothesised")) == ({},)


@pytest.mark.parametrize("nodes", [(PERSON, PERSON), (IMAGE,)])
def test_duplicate_node_identity_is_rejected(nodes):
    with pytest.raises(ValueError, match="unique"):
        SceneGraph(IMAGE, nodes)


@pytest.mark.parametrize("filler", [PERSON, {"nested": [PERSON]},
                                   Proposition("inner", {"entity": PERSON}),
                                   {PERSON: "value"}])
def test_undeclared_references_are_rejected_recursively(filler):
    with pytest.raises(ValueError, match="undeclared"):
        SceneGraph(IMAGE, propositions=(Proposition("relation", {"value": filler}),))


def test_undeclared_scope_and_anchor_are_rejected():
    with pytest.raises(ValueError, match="undeclared"):
        SceneGraph(IMAGE, propositions=(Proposition("scene", scope=CONTEXT),))
    with pytest.raises(ValueError, match="undeclared"):
        SceneGraph(IMAGE, anchors=(VisualAnchor(PERSON),))


@pytest.mark.parametrize("region", [(0, 0, 0, 1), (0, .5, 1, .5), (-.1, 0, 1, 1),
                                    (0, 0, float("nan"), 1), (0, 0, float("inf"), 1),
                                    (False, 0, 1, 1), (0, 0, 1), [0, 0, 1, 1]])
def test_invalid_visual_regions_are_rejected(region):
    with pytest.raises(ValueError):
        VisualAnchor(IMAGE, region)


@pytest.mark.parametrize("filler", [Var("entity"), None, {"nested": Var("x")},
                                   Proposition("inner", {"entity": Var("x")})])
def test_unbound_fillers_are_not_scene_observations(filler):
    with pytest.raises(ValueError, match="bound"):
        SceneGraph(IMAGE, propositions=(Proposition("relation", {"value": filler}),))


def test_mutable_payload_is_rechecked_before_query_and_explicit_validation():
    roles = {"scene": IMAGE}
    graph = SceneGraph(IMAGE, propositions=(Proposition("scene", roles),))
    roles["intruder"] = PERSON
    with pytest.raises(ValueError, match="undeclared"):
        graph.validate()
    with pytest.raises(ValueError, match="undeclared"):
        graph.match(Proposition("scene"))


def test_cycles_are_rejected_but_shared_payloads_are_supported():
    shared = [IMAGE]
    graph = SceneGraph(IMAGE, propositions=(Proposition("scene", {"a": shared, "b": shared}),))
    graph.validate()
    shared.append(shared)
    with pytest.raises(ValueError, match="cycles"):
        graph.validate()


def test_typed_literal_values_can_contain_declared_references():
    @dataclass(frozen=True)
    class Support:
        entity: Ref
        measurement: float
        optional: str | None = None

    SceneGraph(IMAGE, (CUP,), (Proposition("measurement", {"value": Support(CUP, .3)}),))
    with pytest.raises(ValueError, match="undeclared"):
        SceneGraph(IMAGE, propositions=(Proposition("measurement", {"value": Support(CUP, .3)}),))


def test_scene_proposal_preserves_score_kind_and_provenance():
    graph = SceneGraph(IMAGE)
    score = Score(.7, "uncalibrated", "scene-model/v1")
    proposal = SceneProposal(graph, ("scene-model/v1", "full-image"), score)
    assert proposal.score is score
    assert proposal.score.kind == "uncalibrated"
    assert proposal.provenance == ("scene-model/v1", "full-image")
    assert SceneProposal(graph).score is None
    with pytest.raises(TypeError, match="Score"):
        SceneProposal(graph, score=.7)


@pytest.mark.parametrize("proposition", [Proposition(""), Proposition("x", {"": IMAGE}),
                                        Proposition("x", {"scene": IMAGE}, polarity="yes")])
def test_malformed_proposition_is_rejected(proposition):
    with pytest.raises(ValueError):
        SceneGraph(IMAGE, propositions=(proposition,))


@pytest.mark.parametrize("score", [Score(float("nan"), "uncalibrated"),
                                  Score(float("inf"), "utility"),
                                  Score(1.1, "probability", "test basis")])
def test_scene_rejects_invalid_numeric_scores(score):
    with pytest.raises(ValueError):
        SceneProposal(SceneGraph(Ref("image:test")), score=score)
