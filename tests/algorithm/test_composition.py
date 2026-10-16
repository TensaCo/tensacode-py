import itertools
import numpy as np
import pytest
from tensorcode.algorithm import Algorithm, compile_schedule, compare


def test_dependency_readiness_exhaustive_and_portable():
    dependencies = {
        "inspect": [],
        "pick": ["inspect"],
        "reject": ["inspect"],
        "pack": ["pick", "reject"],
    }
    artifact = compile_schedule(dependencies)
    cases = []
    for completed in itertools.product([0.0, 1.0], repeat=4):
        for enabled in itertools.product([0.0, 1.0], repeat=4):
            inputs = {"completed": completed, "enabled": enabled}
            names = list(dependencies)
            expected = [
                float(
                    enabled[i]
                    and not completed[i]
                    and all(completed[names.index(p)] for p in dependencies[n])
                )
                for i, n in enumerate(names)
            ]
            np.testing.assert_array_equal(artifact.run(inputs)["ready"], expected)
            cases.append(inputs)
    for target in ("alg1", "alg2"):
        assert compare(artifact, artifact.compile(target), cases)["passed"]
    assert compare(artifact, Algorithm.from_json(artifact.to_json()), cases)["passed"]
    ns = {}
    exec(artifact.to_python("schedule"), ns)
    np.testing.assert_array_equal(ns["schedule"](cases[-1])[0]["ready"], [0, 0, 0, 0])


@pytest.mark.parametrize(
    "dependencies",
    [
        {},
        {"a": ["missing"]},
        {"a": ["a"]},
        {"a": ["b"], "b": ["a"]},
        {"a": [], "b": ["a", "a"]},
        {"bad name": []},
        {"a": "a"},
        {"a": [1]},
        {str(i): [] for i in range(257)},
    ],
)
def test_reject_invalid_dependency_graph(dependencies):
    with pytest.raises(ValueError):
        compile_schedule(dependencies)
