"""Learn exact-question to declarative memory query plans from explicit teaching.

No plugin is fabricated. Query scopes are literal, including None; allowed_scopes
is an additional exact admission set, never a wildcard or scope expansion policy.
"""
from dataclasses import dataclass

from ..language import Question
from ..records import Proposition, Ref
from .informing import (InformingModel, _TYPES, _fit_question_plans, _validate_answer_query)


@dataclass(frozen=True)
class StoreQueryPlan:
    query: Proposition
    answer_variable: str
    allowed_scopes: tuple[Ref | None, ...]

    def __post_init__(self):
        _validate_answer_query(self.query, self.answer_variable)
        if (type(self.allowed_scopes) is not tuple or not self.allowed_scopes
                or any(value is not None and type(value) is not Ref for value in self.allowed_scopes)
                or len(set(self.allowed_scopes)) != len(self.allowed_scopes)):
            raise ValueError('allowed scopes must be a nonempty exact set of Ref or None')
        if self.query.scope not in self.allowed_scopes:
            raise ValueError('literal query scope is not explicitly allowed')


@dataclass(frozen=True)
class StoreQueryExample:
    id: str
    source_id: str
    text: str
    question: Question
    plan: StoreQueryPlan
    basis: tuple[str, ...] = ()


@dataclass(frozen=True)
class StoreQueryProposal:
    plan: StoreQueryPlan
    template_ids: tuple[str, ...]
    training_example_ids: tuple[str, ...]
    validation_example_ids: tuple[str, ...]
    conflicting_validation_example_ids: tuple[str, ...]
    conflicting_training_example_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class StoreQueryCandidates:
    proposals: tuple[StoreQueryProposal, ...]
    complete: bool
    unresolved: tuple[str, ...] = ()


class StoreQueryModel(InformingModel):
    _kind = 'store_query'
    _codec = {**_TYPES, 'StoreQueryPlan': StoreQueryPlan}
    _proposal_type = StoreQueryProposal
    _candidates_type = StoreQueryCandidates

    @staticmethod
    def _coverage(plan):
        # Merely listing a reference in allowed_scopes cannot consume a question
        # participant. It must occur in the actual query, including literal scope.
        return plan.query


def fit_store_queries(training, validation, *, max_pairs=256):
    return _fit_question_plans(training, validation, max_pairs=max_pairs,
        example_type=StoreQueryExample, plan_type=StoreQueryPlan, model_type=StoreQueryModel)
