from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
from pathlib import Path

import pytest

from tensorcode.agent.refinements import RefinementLibrary
from tensorcode.goals import Condition, GoalSpec
from tensorcode.language.semantics import Entity, Frame
from tensorcode.language.verbnet import Goal
from tensorcode.outcomes import Unknown
from tensorcode.records import Ref


def project(**features):
    """An unresolved lexical proposal; only a successful recipe creates GoalSpec."""
    entity = Entity("description", "arbitrary surface text",
                    {"noun": "project", "quality": "python", **features})
    return Goal("make", "authored-test", (Condition("be", {"Result": entity}),),
                Frame("make", {"object": entity}, {"mood": "imperative"}))


def test_project_refinement_names_locations_and_explicit_basis(tmp_path):
    library = RefinementLibrary.load(Path(__file__).parent / "fixtures/project_refinements.json")
    goal = project(name=Entity("name", "sample"), location=Entity("name", "scratch"))
    result = library.refine(goal, context={"root": tmp_path})
    assert isinstance(result, GoalSpec)
    assert result.conditions == (
        Condition("directory_exists", {"path": tmp_path / "scratch/sample"}),
        Condition("file_exists", {"path": tmp_path / "scratch/sample/main.py"}),
        Condition("content", {"path": tmp_path / "scratch/sample/main.py", "text": 'print("Hello, world!")\n'}),
    )
    assert result.basis[0].startswith("hand-authored:minimal-python-project:")


def test_arbitrary_data_recipe_without_code_changes(tmp_path):
    recipe = {
        "id": "unfamiliar-symbols",
        "match": {"predicate": "ready", "role": "instrument", "features": {"noun": "glorp", "flavor": "violet"}},
        "bindings": {"color": {"from": "entity.flavor"}},
        "outputs": [{"predicate": "calibrated", "roles": {"hue": {"binding": "color"}}}],
    }
    goal = Goal("prepare", "authored-test", (Condition("ready", {
        "Instrument": Entity("description", "ignored", {"noun": "glorp", "flavor": "violet"})}),),
        Frame("prepare", features={"mood": "imperative"}))
    result = RefinementLibrary([recipe]).refine(goal, context={})
    assert isinstance(result, GoalSpec)
    assert result.conditions == (Condition("calibrated", {"hue": "violet"}),)


def test_ambiguity_is_not_decided_by_recipe_order(tmp_path):
    library = RefinementLibrary.load(Path(__file__).parent / "fixtures/project_refinements.json")
    alternative = deepcopy(library.recipes[0])
    alternative["id"] = "other-convention"
    result = RefinementLibrary([*library.recipes, alternative]).refine(project(), context={"root": tmp_path})
    assert isinstance(result, Unknown)
    assert result.reason == "ambiguous_refinement"


def test_unmatched_quality_and_predicate_do_not_refine(tmp_path):
    library = RefinementLibrary.load(Path(__file__).parent / "fixtures/project_refinements.json")
    for goal in (project(quality="rust"), replace(project(), conditions=(Condition("destroyed", project().conditions[0].args),)),
                 replace(project(), conditions=(Condition("be", project().conditions[0].args, negated=True),))):
        result = library.refine(goal, context={"root": tmp_path})
        assert isinstance(result, Unknown)
        assert result.reason == "no_refinement"


def test_unhandled_modifiers_and_nested_modifiers_fail_closed(tmp_path):
    library = RefinementLibrary.load(Path(__file__).parent / "fixtures/project_refinements.json")
    for goal in (project(quantity=3), project(location=Entity("description", "old scratch", {"noun": "scratch", "quality": "old"}))):
        result = library.refine(goal, context={"root": tmp_path})
        assert isinstance(result, Unknown)
        assert result.reason == "incomplete_refinement"


def test_additional_conditions_invariants_and_basis_survive(tmp_path):
    existing = Condition("preserved", {"path": tmp_path / "README"})
    # Authored test draft isolates qualifier preservation without claiming that
    # its unresolved project description already satisfies GoalSpec's boundary.
    goal = SimpleNamespace(conditions=(*project().conditions, existing),
                           invariants=(existing,), basis=("user",), label="draft")
    result = RefinementLibrary.load(Path(__file__).parent / "fixtures/project_refinements.json").refine(goal, context={"root": tmp_path})
    assert isinstance(result, GoalSpec)
    assert result.conditions[-1] == existing
    assert result.invariants == (existing,)
    assert result.basis[0] == "user"


def test_recipe_cannot_evaluate_code(tmp_path):
    library = RefinementLibrary.load(Path(__file__).parent / "fixtures/project_refinements.json")
    library.recipes[0]["derived"]["directory"] = {"eval": "__import__('os').getcwd()"}
    result = library.refine(project(), context={"root": tmp_path})
    assert isinstance(result, Unknown)
    assert "unsupported recipe expression" in result.detail


def test_recipe_loading_version_and_missing_context(tmp_path):
    result = RefinementLibrary.load(Path(__file__).parent / "fixtures/project_refinements.json").refine(project(), context={})
    assert isinstance(result, Unknown)
    assert result.reason == "incomplete_refinement"


def lexical_project(*, material=None, product_ref=None, unmapped=(), features=None):
    entity = replace(next(iter(project().conditions[0].args.values())), ref=product_ref)
    return Goal("make", "build-26.1-1", (
        Condition("be", {"Product": entity}),
        Condition("made_of", {"Product": entity, "Material": material}),
    ), Frame("make", {"object": entity}, features if features is not None else {"mood": "imperative"}), unmapped)


def test_only_declared_implicit_lexical_conditions_are_omitted(tmp_path):
    library = RefinementLibrary.load(Path(__file__).parent / "fixtures/project_refinements.json")
    result = library.refine(lexical_project(), context={"root": tmp_path})
    assert isinstance(result, GoalSpec)
    assert len(result.conditions) == 3
    assert any(item.startswith("implicit-lexical-condition:") for item in result.basis)
    material = Entity("description", "wood", {"noun": "wood"})
    explicit = library.refine(lexical_project(material=material), context={"root": tmp_path})
    assert isinstance(explicit, Unknown)
    assert explicit.reason == "incomplete_refinement"
    assert "unconsumed entity features" in explicit.detail
    # A known material does not establish which product the relation describes.
    material_ref = Ref("material:supplied")
    partially_grounded = library.refine(lexical_project(material=material_ref), context={"root": tmp_path})
    assert isinstance(partially_grounded, Unknown)
    assert partially_grounded.reason == "incomplete_refinement"
    # Identity alone does not consume qualifications on the preserved relation.
    product_ref = Ref("artifact:supplied")
    grounded = library.refine(lexical_project(
        material=replace(material, ref=material_ref), product_ref=product_ref), context={"root": tmp_path})
    assert isinstance(grounded, Unknown)
    assert "unconsumed entity features" in grounded.detail
    # A separately supplied domain relation has explicit values, not a linguistic
    # description whose qualifiers the normalizer would have to erase.
    supplied = lexical_project()
    relation = Condition("made_of", {"Product": product_ref, "Material": material_ref})
    supplied = replace(supplied, conditions=(supplied.conditions[0], relation))
    projected = library.refine(supplied, context={"root": tmp_path})
    assert isinstance(projected, GoalSpec)
    assert projected.conditions[-1] == relation
    assert not any("made_of" in item for item in projected.basis if item.startswith("implicit-lexical-condition:"))


def test_unmapped_roles_and_frame_modifiers_are_not_discarded(tmp_path):
    library = RefinementLibrary.load(Path(__file__).parent / "fixtures/project_refinements.json")
    for goal in (lexical_project(unmapped=("location",)),
                 lexical_project(features={"mood": "imperative", "modality": "possible"}),
                 lexical_project(features={"mood": "imperative", "polarity": "negative"})):
        result = library.refine(goal, context={"root": tmp_path})
        assert isinstance(result, Unknown)
        assert result.reason == "incomplete_refinement"


def test_undeclared_partial_lexical_condition_blocks_instead_of_crashing(tmp_path):
    goal = lexical_project()
    partial = Goal(goal.verb, goal.verb_class, (*goal.conditions, Condition("relation", {"subject": "bound", "object": None})), goal.frame)
    result = RefinementLibrary.load(Path(__file__).parent / "fixtures/project_refinements.json").refine(partial, context={"root": tmp_path})
    assert isinstance(result, Unknown)
    assert result.reason == "incomplete_refinement"


def test_text_bindings_preserve_names_and_locations_not_noun_lemmas(tmp_path):
    goal = project(name=Entity("description", "Notes", {"noun": "note", "number": "plural"}),
                   location=Entity("description", "documents", {"noun": "document", "number": "plural", "definite": False}))
    result = RefinementLibrary.load(Path(__file__).parent / "fixtures/project_refinements.json").refine(goal, context={"root": tmp_path})
    assert isinstance(result, GoalSpec)
    assert result.conditions[0].args["path"] == tmp_path / "documents/Notes"
    qualified = project(name=Entity("description", "old notes", {"noun": "note", "number": "plural", "quality": "old"}))
    assert isinstance(RefinementLibrary.load(Path(__file__).parent / "fixtures/project_refinements.json").refine(qualified, context={"root": tmp_path}), Unknown)


def test_real_parser_preserves_plural_path_components(tmp_path):
    import pytest
    from agent_test_support import selected_agent as Agent, fixture_goal_selector
    from tensorcode.agent.filesystem import FileSystemPlugin
    from tensorcode.language import verbnet, wordnet

    if wordnet.find_wordnet() is None or verbnet.find_verbnet() is None:
        pytest.skip("requires WordNet and VerbNet data")
    result = Agent([FileSystemPlugin(tmp_path, refinements=RefinementLibrary.load(Path(__file__).parent / "fixtures/project_refinements.json"))],
                   goal_selector=fixture_goal_selector('build-26.1-1', frame_index=0)).turn("make a python project called hello in documents")
    assert result.outcomes[0].status == "done", result
    assert (tmp_path / "documents/hello/main.py").read_text() == 'print("Hello, world!")\n'


def test_implicit_lexical_subject_is_not_an_artifact_requirement(tmp_path):
    goal = lexical_project()
    lexical = Goal("create", "engender-27.1", (
        Condition("be", {"Precondition": "addressee"}), goal.conditions[0],
    ), goal.frame)
    result = RefinementLibrary.load(Path(__file__).parent / "fixtures/project_refinements.json").refine(lexical, context={"root": tmp_path})
    assert isinstance(result, GoalSpec)
    assert len(result.conditions) == 3
    assert any("Precondition=addressee" in item for item in result.basis)
    bound = Goal(lexical.verb, lexical.verb_class, (
        Condition("be", {"Precondition": "specified"}), goal.conditions[0],
    ), goal.frame)
    result = RefinementLibrary.load(Path(__file__).parent / "fixtures/project_refinements.json").refine(bound, context={"root": tmp_path})
    assert isinstance(result, GoalSpec)
    assert result.conditions[0] == bound.conditions[0]


def test_modifier_metadata_requires_each_relation_to_be_consumed(tmp_path):
    library = RefinementLibrary.load(Path(__file__).parent / "fixtures/project_refinements.json")
    name = Entity("name", "Sample")
    valid = project(name=name, modifiers=(("compound", "python"), ("name", name)))
    assert isinstance(library.refine(valid, context={"root": tmp_path}), GoalSpec)
    for modifiers in ((("unknown", "python"),),
                      (("compound", "rust"),),
                      (("compound", "python"), ("amod", "python"))):
        result = library.refine(project(modifiers=modifiers), context={"root": tmp_path})
        assert isinstance(result, Unknown)
        assert result.reason == "incomplete_refinement"


def test_frame_location_fallback_is_consumed_only_when_bound_and_consistent(tmp_path):
    library = RefinementLibrary.load(Path(__file__).parent / "fixtures/project_refinements.json")
    location = Entity("description", "documents", {"noun": "document", "number": "plural"})
    entity = next(iter(project(name="hello").conditions[0].args.values()))
    goal = Goal("make", "build", (Condition("be", {"Product": entity}),),
                Frame("make", {"object": entity, "location": location}, {"mood": "imperative"}), ("location",))
    result = library.refine(goal, context={"root": tmp_path})
    assert isinstance(result, GoalSpec)
    assert result.conditions[0].args["path"] == tmp_path / "documents/hello"
    conflicting_entity = Entity(entity.kind, entity.text, {**entity.features, "location": "elsewhere"})
    conflict = Goal(goal.verb, goal.verb_class, (Condition("be", {"Product": conflicting_entity}),), goal.frame, goal.unmapped_roles)
    result = library.refine(conflict, context={"root": tmp_path})
    assert isinstance(result, Unknown)
    assert "conflicting binding sources" in result.detail
    missing = Goal(goal.verb, goal.verb_class, goal.conditions, Frame("make", {"object": entity}, {"mood": "imperative"}), ("location",))
    result = library.refine(missing, context={"root": tmp_path})
    assert isinstance(result, Unknown)
    assert "Unmapped lexical roles" in result.detail


def test_default_filesystem_has_no_refinement_library(tmp_path):
    from tensorcode.agent.filesystem import FileSystemPlugin

    plugin = FileSystemPlugin(tmp_path)
    assert plugin.refinements is None
    assert plugin.refine_goal(project()).reason == "no_refinement"


@pytest.mark.parametrize("value", [True, False])
def test_boolean_refinement_switches_are_rejected(tmp_path, value):
    from tensorcode.agent.filesystem import FileSystemPlugin

    with pytest.raises(TypeError, match="RefinementLibrary or None"):
        FileSystemPlugin(tmp_path, refinements=value)


def test_library_requires_explicit_path():
    with pytest.raises(TypeError):
        RefinementLibrary.load()
