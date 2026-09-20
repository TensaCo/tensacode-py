"""The active agent admits and queries identity only through explicit bindings."""
from datetime import datetime, timezone

import pytest

from tensorcode.agent.core import Agent, USER
from tensorcode.agent.grounding import MentionBinding, propose_grounding
from tensorcode.agent.plugin import Plugin
from tensorcode.agent.understand import Act, Sentence, SentenceAlternative
from tensorcode.language import Entity, Frame, Question
from tensorcode.language.semantics import explicit_ref, to_propositions
from tensorcode.outcomes import Unknown
from tensorcode.records import Evidence, Proposition, Ref, Var
from store_query_fixtures import taught_lookup


def entity(text="folder", identity=None):
    return Entity("description", text, ref=Ref(identity) if identity else None)


def tell(agent, frame):
    act = Act("tell", frame, frame)
    sentence = Sentence("source statement", (), None, (act,))
    events = []
    return agent.tell(sentence, act, events), events


def test_identical_descriptions_keep_distinct_explicit_identities():
    agent = Agent()
    for identity, place in [("world:a", "first"), ("world:b", "second")]:
        outcome, _ = tell(agent, Frame("located", {"subject": entity(identity=identity), "place": place}))
        assert outcome.status == "noted"
    assert taught_lookup(agent, Question(Frame("located", {"subject": entity(identity="world:a")}), "place"), Proposition("located", {"subject": Ref("world:a"), "place": Var("answer")}, scope=USER)) == ["first"]
    assert taught_lookup(agent, Question(Frame("located", {"subject": entity(identity="world:b")}), "place"), Proposition("located", {"subject": Ref("world:b"), "place": Var("answer")}, scope=USER)) == ["second"]


def test_different_descriptions_share_identity_only_when_explicitly_bound():
    agent = Agent()
    tell(agent, Frame("located", {"subject": entity("blue box", "world:a"), "place": "desk"}))
    assert taught_lookup(agent, Question(Frame("located", {"subject": entity("my storage", "world:a")}), "place"), Proposition("located", {"subject": Ref("world:a"), "place": Var("answer")}, scope=USER)) == ["desk"]
    assert isinstance(agent.lookup(Question(Frame("located", {"subject": entity("blue box")}), "place")), Unknown)


def test_partly_grounded_query_never_drops_unresolved_constraints():
    agent = Agent()
    tell(agent, Frame("located", {"subject": entity(identity="world:a"), "owner": entity(identity="world:owner"), "place": "desk"}))
    question = Question(Frame("located", {"subject": entity(identity="world:a"), "owner": entity("owner")}), "place")
    grounded = Question(Frame("located", {"subject": entity(identity="world:a"), "owner": entity("owner", "world:owner")}), "place")
    assert taught_lookup(agent, grounded, Proposition('located', {'subject': Ref('world:a'),
        'owner': Ref('world:owner'), 'place': Var('answer')}, scope=USER)) == ['desk']
    from informing_fixtures import selected_question_dependency
    assert isinstance(agent.lookup(question,
        interpretation_dependency=selected_question_dependency(agent, question)), Unknown)
    # Even a capable plugin may not resolve a raw description outside the workspace.
    class GuessingPlugin(Plugin):
        def capabilities(self):
            raise AssertionError("unresolved query must stop before capability discovery")
        def denote(self, value):
            raise AssertionError("must not guess identity")
    agent.plugins = (GuessingPlugin("guess"),)
    act = Act("question", question, question.frame)
    outcome = agent.ask(Sentence("where", (), None, (act,)), act, [])
    assert outcome.status == "unknown"
    assert agent._ref_of(entity()) is None


def test_hypothesis_scopes_require_explicit_query_opt_in():
    agent = Agent()
    hypothesis = Ref("hypothesis:unproven")
    evidence = Evidence(Ref("test:fixture"), datetime.now(timezone.utc), "authored")
    agent.store.assert_(Proposition("located", {"subject": Ref("world:a"), "place": "imagined"}, scope=hypothesis), evidence)
    question = Question(Frame("located", {"subject": entity(identity="world:a")}), "place")
    assert isinstance(taught_lookup(agent, question, Proposition('located', {'subject': Ref('world:a'), 'place': Var('answer')}, scope=USER)), Unknown)
    assert taught_lookup(agent, question, Proposition('located', {'subject': Ref('world:a'), 'place': Var('answer')}, scope=hypothesis)) == ['imagined']
    tell(agent, Frame("located", {"subject": entity(identity="world:a"), "place": "reported"}))
    assert taught_lookup(agent, question, Proposition("located", {"subject": Ref("world:a"), "place": Var("answer")}, scope=USER)) == ["reported"]


def test_grounded_workspace_alternative_flows_through_real_tell():
    agent = Agent()
    workspace = agent.interpretations
    source = workspace.add_source("the folder is on the desk")
    scene = workspace.add_source("authored scene correspondence", modality="image")
    group = workspace.create_group(source.id)
    frame = Frame("located", {"subject": entity(), "place": "desk"})
    parent = workspace.propose(group.id, SentenceAlternative(None, (Act("tell", frame, frame),)))
    proposal = propose_grounding(workspace, group.id, parent.id, [MentionBinding(("acts", 0, "frame", "roles", "subject"), Ref("world:a"), (scene.id,), "explicit scene correspondence")])
    act = proposal.payload.acts[0]
    outcome = agent.tell(Sentence(source.text, (), None, (act,)), act, [])
    assert outcome.status == "noted"
    assert taught_lookup(agent, Question(Frame("located", {"subject": entity(identity="world:a")}), "place"), Proposition("located", {"subject": Ref("world:a"), "place": Var("answer")}, scope=USER)) == ["desk"]
    assert workspace.get(group.id).selected_id is None
    assert parent.payload.acts[0].frame.roles["subject"].ref is None


def test_raw_mentions_do_not_create_world_refs_or_assertions():
    agent = Agent()
    outcome, events = tell(agent, Frame("located", {"subject": entity(), "place": "desk"}))
    assert outcome.status == "not_understood"
    assert not list(agent.store.find(Proposition("located", {})))
    assert "clause[0].roles.subject" in events[0]["why"][0]
    agent.last_image = Ref("image:recent")
    assert agent._ref_of(entity("image")) is None


@pytest.mark.parametrize("failure", [None, Unknown("unresolved")])
def test_converter_drops_failed_nested_clause_with_exact_occurrence_path(failure):
    frame = Frame("outer", {"subject": Ref("world:a"), "content": Frame("nested", {"items": ("safe", entity())})})
    propositions, dropped = to_propositions(frame, source=USER, resolve=lambda _: failure)
    assert not propositions
    assert "unresolved entity identity: clause[0].roles.content.roles.items[1]" in dropped


def test_explicit_literals_keep_values_without_world_identity():
    assert explicit_ref(Entity("number", "three", {"value": 3})) == 3
    assert explicit_ref(Entity("literal", "hello")) == "hello"
    assert isinstance(explicit_ref(entity()), Unknown)
    agent = Agent()
    tell(agent, Frame("measured", {"subject": entity(identity="world:a"), "measurement": Entity("number", "three", {"value": 3})}))
    assert taught_lookup(agent, Question(Frame("measured", {"subject": entity(identity="world:a")}), "measurement"), Proposition("measured", {"subject": Ref("world:a"), "measurement": Var("answer")}, scope=USER)) == [3]


def test_grounded_report_query_passes_identity_without_plugin_semantic_resolution():
    from tensorcode.agent.plugin import Capability, Informs, Param
    from tensorcode.outcomes import Receipt
    from tensorcode.records import Var

    identity = Ref("world:report-target")
    class Reports(Plugin):
        def capabilities(self):
            return (Capability("inspect", (Param("target", "entity"),), informs=(Informs("status", "undergoer", "target", query=Proposition("status", {"subject": Var("target"), "result": Var("answer")})),), effect_kind="read"),)
        def refer(self, *args, **kwargs):
            raise AssertionError("explicit identity must not be reinterpreted by a plugin")
        def execute(self, call, *, key):
            assert call.arg("target") == identity
            return Receipt(call, "applied")
        def reveal(self, cap, args, receipt):
            yield Proposition("status", {"subject": identity, "result": "ready"})
    agent = Agent([Reports("report")])
    frame = Frame("status", {"subject": entity("any wording", identity.id)})
    act = Act("question", Question(frame, "result"), frame)
    from informing_fixtures import teach_informing, selected_question_dependency
    from tensorcode.learning.informing import InformingPlan
    teach_informing(agent, act.meaning, InformingPlan('report', 'inspect', (('target', identity),),
        Proposition('status', {'subject': identity, 'result': Var('answer')}), 'answer'))
    outcome = agent.ask(Sentence("explicit query", (), None, (act,)), act, [],
        interpretation_dependency=selected_question_dependency(agent, act.meaning))
    assert outcome.status == "answered"
    assert outcome.answer == ["ready"]


@pytest.mark.parametrize("reverse", [False, True])
def test_competing_informing_actions_never_execute_by_registry_order(reverse):
    from tensorcode.agent.plugin import Capability, Informs, Param
    from tensorcode.records import Var

    class Competing(Plugin):
        def capabilities(self):
            names = ("first-report", "second-report")
            if reverse:
                names = names[::-1]
            return tuple(Capability(name, (Param("target", "entity"),), informs=(
                Informs("status", "undergoer", "target", query=Proposition(
                    "status", {"subject": Var("target"), "result": Var("answer")})),),
                effect_kind="read") for name in names)
        def execute(self, *args, **kwargs):
            raise AssertionError("competing report contracts cannot choose themselves")
    agent = Agent([Competing("reports")])
    frame = Frame("status", {"subject": entity(identity="world:a")})
    act = Act("question", Question(frame, "result"), frame)
    events = []
    outcome = agent.ask(Sentence("explicit query", (), None, (act,)), act, events)
    assert outcome.status == "unknown"
    assert agent.informing_model is None
    assert outcome.receipt is None
    assert not agent.store.propositions()
