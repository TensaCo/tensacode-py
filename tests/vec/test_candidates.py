import pytest
import torch

from tensorcode.ops.vec import (
    CandidateSet,
    Decide,
    Decode,
    Latent,
    Retrieve,
    Score,
    Space,
)


class DotScore(torch.nn.Module):
    def __init__(self, scale=1.0):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(float(scale)))

    def forward(self, query, candidates):
        return (query.unsqueeze(-2) * candidates).sum(dim=-1) * self.scale


def _single_candidates():
    space = Space("retrieval/shared", 2)
    return CandidateSet(
        query=Latent(torch.tensor([1.0, 0.0], requires_grad=True), space),
        candidates=Latent(
            torch.tensor([[0.2, 0.0], [0.9, 0.0], [-0.5, 0.0]], requires_grad=True),
            space,
            sources=("memory:index",),
            metadata={"snapshot": 4},
        ),
        identities=("low", "high", "negative"),
        metadata=({"row": 1}, {"row": 2}, {"row": 3}),
    )


def test_score_returns_named_tensor_scores_and_preserves_gradients():
    space = Space("retrieval/shared", 2)
    scorer = Score.from_module(
        DotScore(),
        query_space=space,
        candidate_space=space,
        meaning="unnormalized dot-product similarity",
    )

    result = scorer(_single_candidates())

    assert result.meaning == "unnormalized dot-product similarity"
    assert torch.allclose(result.values, torch.tensor([0.2, 0.9, -0.5]))
    result.values.sum().backward()
    assert scorer.module.scale.grad is not None
    assert result.candidates.query.tensor.grad is not None
    assert result.candidates.candidates.tensor.grad is not None


def test_score_rejects_equal_dimensions_from_an_incompatible_space():
    expected = Space("model-a", 2)
    other = Space("model-b", 2)
    values = CandidateSet(
        Latent(torch.ones(2), expected),
        Latent(torch.ones(2, 2), other),
        identities=("a", "b"),
    )
    scorer = Score.from_module(
        DotScore(),
        query_space=expected,
        candidate_space=expected,
        meaning="similarity",
    )

    with pytest.raises(ValueError, match="incompatible.*space"):
        scorer(values)


def test_candidate_set_rejects_empty_or_misaligned_candidates():
    space = Space("candidates", 2)
    with pytest.raises(ValueError, match="at least one"):
        CandidateSet(
            Latent(torch.ones(2), space),
            Latent(torch.empty(0, 2), space),
            identities=(),
        )
    with pytest.raises(ValueError, match="batch shape"):
        CandidateSet(
            Latent(torch.ones(2, 2), space),
            Latent(torch.ones(3, 4, 2), space),
            identities=("a", "b", "c", "d"),
        )
    with pytest.raises(ValueError, match="identities"):
        CandidateSet(
            Latent(torch.ones(2), space),
            Latent(torch.ones(2, 2), space),
            identities=("only-one",),
        )
    for identities in (("", "b"), ("same", "same")):
        with pytest.raises(ValueError, match="unique nonempty"):
            CandidateSet(
                Latent(torch.ones(2), space),
                Latent(torch.ones(2, 2), space),
                identities=identities,
            )


def test_decide_handles_batches_while_python_identity_conversion_is_explicit():
    space = Space("decision/shared", 2)
    candidates = CandidateSet(
        query=Latent(torch.tensor([[1.0, 0.0], [0.0, 1.0]]), space),
        candidates=Latent(
            torch.tensor(
                [
                    [[0.1, 0.0], [0.9, 0.0]],
                    [[0.0, 0.8], [0.0, 0.2]],
                ]
            ),
            space,
        ),
        identities=("first", "second"),
    )
    scored = Score.from_module(
        DotScore(),
        query_space=space,
        candidate_space=space,
        meaning="utility logit",
    )(candidates)

    decision = Decide()(scored)

    assert torch.equal(decision.indices, torch.tensor([1, 0]))
    assert torch.allclose(decision.scores, torch.tensor([0.9, 0.8]))
    assert decision.identities == ("second", "first")
    with pytest.raises(ValueError, match="batched"):
        _ = decision.identity


def test_retrieve_ranks_existing_candidates_and_preserves_candidate_metadata():
    candidates = _single_candidates()
    space = candidates.query.space
    scored = Score.from_module(
        DotScore(),
        query_space=space,
        candidate_space=space,
        meaning="relevance score",
    )(candidates)

    retrieval = Retrieve({'k':2})(scored)

    assert torch.equal(retrieval.indices, torch.tensor([1, 0]))
    assert torch.allclose(retrieval.scores, torch.tensor([0.9, 0.2]))
    assert retrieval.identities == ("high", "low")
    assert retrieval.metadata == ({"row": 2}, {"row": 1})
    assert retrieval.items.sources == ("memory:index",)
    assert retrieval.items.metadata == {"snapshot": 4}
    assert torch.equal(retrieval.items.tensor, torch.tensor([[0.9, 0.0], [0.2, 0.0]]))


def test_retrieve_rejects_k_beyond_the_candidate_bound():
    candidates = _single_candidates()
    space = candidates.query.space
    scored = Score.from_module(
        DotScore(), query_space=space, candidate_space=space, meaning="relevance"
    )(candidates)

    with pytest.raises(ValueError, match="only 3 candidates"):
        Retrieve({'k':4})(scored)
    with pytest.raises(ValueError, match="positive"):
        Retrieve({'k':0})


def test_decide_and_retrieve_exclude_masked_candidates_and_bound_valid_count():
    space = Space("masked/shared", 1)
    candidates = CandidateSet(
        query=Latent(torch.tensor([1.0]), space),
        candidates=Latent(
            torch.tensor([[1.0], [100.0], [2.0]]),
            space,
            mask=torch.tensor([True, False, True]),
        ),
        identities=("one", "masked", "two"),
    )
    scored = Score.from_module(
        DotScore(), query_space=space, candidate_space=space, meaning="similarity"
    )(candidates)

    assert Decide()(scored).identity == "two"
    assert Retrieve({'k':2})(scored).identities == ("two", "one")
    with pytest.raises(ValueError, match="valid candidates"):
        Retrieve({'k':3})(scored)

    descending_candidates = CandidateSet(
        query=Latent(torch.tensor([1.0]), space),
        candidates=Latent(
            torch.tensor([[1.0], [-100.0], [2.0]]),
            space,
            mask=torch.tensor([True, False, True]),
        ),
        identities=("one", "masked", "two"),
    )
    descending = Score.from_module(
        DotScore(), query_space=space, candidate_space=space, meaning="cost"
    )(descending_candidates)
    assert Decide({'largest':False})(descending).identity == "one"
    assert Retrieve({'k':2,'largest':False})(descending).identities == ("one", "two")


def test_candidate_availability_mask_must_be_boolean_and_nonempty_per_batch():
    space = Space("masked/shared", 1)
    with pytest.raises(ValueError, match="boolean"):
        CandidateSet(
            Latent(torch.tensor([1.0]), space),
            Latent(torch.ones(2, 1), space, mask=torch.ones(2)),
            identities=("a", "b"),
        )
    with pytest.raises(ValueError, match="valid candidate"):
        CandidateSet(
            Latent(torch.tensor([1.0]), space),
            Latent(torch.ones(2, 1), space, mask=torch.zeros(2, dtype=torch.bool)),
            identities=("a", "b"),
        )


def test_decode_uses_supplied_module_and_retains_autograd():
    space = Space("decoder/input", 3)
    module = torch.nn.Linear(3, 2, bias=False)
    decode = Decode.from_module(module, input_space=space, output="two regression values")
    source = torch.ones(3, requires_grad=True)

    result = decode(Latent(source, space))

    assert result.shape == (2,)
    result.sum().backward()
    assert source.grad is not None
    assert module.weight.grad is not None
    with pytest.raises(ValueError, match="incompatible.*space"):
        decode(Latent(torch.ones(3), Space("other", 3)))
