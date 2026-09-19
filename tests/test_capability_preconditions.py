"""Action applicability remains distinct from effects and supports unknown evidence."""

from tensorcode.agent.plugin import (
    Capability, Effect, Informs, Param, Plugin, Precondition, describe_capabilities,
)
from tensorcode.outcomes import Unknown


def test_existing_positional_capability_declarations_remain_compatible():
    params = (Param("path", "file"),)
    effects = (Effect("be", {"undergoer": "path"}),)
    informs = (Informs("be", "undergoer", "path"),)
    capability = Capability("inspect", params, effects, informs, "read", "Inspect a file")

    assert capability.params == params
    assert capability.effects == effects
    assert capability.informs == informs
    assert capability.effect_kind == "read"
    assert capability.description == "Inspect a file"
    assert capability.preconditions == ()


def test_declared_preconditions_default_to_unknown_including_negative_conditions():
    plugin = Plugin("files")
    for negated in (False, True):
        condition = Precondition("exists", {"entity": "path"}, negated=negated)
        result = plugin.precondition_holds(condition, {"path": "/scratch/example"})
        assert isinstance(result, Unknown)
        assert result.reason == "no_precondition_check"


def test_capability_description_exposes_requirements_separately_from_effects():
    class Files(Plugin):
        def capabilities(self):
            return (Capability(
                "create", (Param("path", "file"), Param("parent", "directory")),
                effects=(Effect("exists", {"entity": "path"}),),
                preconditions=(
                    Precondition("exists", {"entity": "parent"}),
                    Precondition("exists", {"entity": "path"}, negated=True),
                ),
            ),)

    description, = describe_capabilities([Files("files")])
    assert description["preconditions"] == ["exists(entity=parent)", "not exists(entity=path)"]
    assert description["effects"] == ["exists(entity=path)"]
