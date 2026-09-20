"""Model-derived scene discrimination, without teacher answers or commitments.

Policy: partition retained query hypotheses by their exact full denotation sets
on each offered scene; minimize the largest partition. Each retained query counts
once, not as probability mass. Equivalent best scenes remain explicit ties.
"""
from copy import deepcopy
from dataclasses import dataclass

from ..agent.scene import SceneGraph
from ..records import Ref
from .experience import _same
from .graph_queries import match_query
from .scene_grounding import SceneGroundingModel, _description

POLICY = 'minimax exact denotation partitions; uniform retained-query counts; all ties retained'


@dataclass(frozen=True)
class QueryDenotation:
    query_id: str
    references: tuple[Ref, ...]
    training_example_ids: tuple[str, ...]
    validation_example_ids: tuple[str, ...]
    conflicting_validation_example_ids: tuple[str, ...]
    validated: bool
    complete: bool
    unresolved: tuple[str, ...]
    explored: int


@dataclass(frozen=True)
class ScenePredictions:
    scene_id: Ref
    predictions: tuple[QueryDenotation, ...]
    partitions: tuple[tuple[str, ...], ...]
    worst_partition_size: int | None
    discriminating: bool


@dataclass(frozen=True)
class GroundingInvestigation:
    model_id: str
    description: object
    scenes: tuple[SceneGraph, ...]
    predictions: tuple[ScenePredictions, ...]
    best_scene_ids: tuple[Ref, ...]
    complete: bool
    unresolved: tuple[str, ...]
    policy: str = POLICY


def investigate_grounding(model, description, scenes):
    """Predict all exact-description rivals on every offered scene, then rank.

    Authentication belongs to the retaining agent boundary. This pure function
    consumes a supplied SceneGroundingModel and makes no workspace or world edits.
    Partial denotations remain inspectable but never establish preferred scenes.
    """
    if type(model) is not SceneGroundingModel:
        raise TypeError('investigation requires a SceneGroundingModel')
    description, offered = deepcopy(description), deepcopy(tuple(scenes))
    unique = {}
    for scene in offered:
        if type(scene) is not SceneGraph: raise ValueError('offered scenes must be SceneGraphs')
        scene.validate()
        if scene.image in unique and not _same(unique[scene.image], scene):
            raise ValueError('same scene identity has inconsistent graph content')
        unique[scene.image] = scene
    offered = tuple(unique.values())
    unresolved = list(model.unresolved)
    complete = model.complete
    try:
        key = _description(description)
    except (ValueError, TypeError, RecursionError) as error:
        return GroundingInvestigation(model.id, description, offered, (), (), False,
                                      ('unsupported_description:' + str(error),))
    queries = tuple(q for q in model.queries if q.description == key)
    if not queries:
        known = any(_description(example.description) == key for example in model.training_examples)
        unresolved.append('no_supported_query' if known else 'unknown_description')
        if not known:
            complete = False
    if not offered:
        unresolved.append('no_offered_scenes')
        complete = False
    if not model.complete: unresolved.append('model_search_incomplete')
    rows = []
    for scene in offered:
        predictions, partitions = [], {}
        scene_complete = True
        for query in queries:
            result = match_query(query.query, scene, max_matches=model.max_matches, max_states=model.max_matches)
            references = tuple(sorted({m.bindings[0] for m in result.matches}, key=lambda ref: ref.id))
            prediction = QueryDenotation(query.id, references, query.training_example_ids,
                query.validation_example_ids, query.conflicting_validation_example_ids,
                bool(query.validation_example_ids) and not query.conflicting_validation_example_ids,
                result.complete, result.unresolved, result.explored)
            predictions.append(prediction)
            if not result.complete:
                complete = scene_complete = False
                unresolved.append('incomplete_denotation:' + scene.image.id + ':' + query.id)
            partitions.setdefault(references, []).append(query.id)
        groups = tuple(tuple(ids) for ids in partitions.values()) if scene_complete else ()
        worst = max((len(group) for group in groups), default=None)
        rows.append(ScenePredictions(scene.image, tuple(predictions), groups, worst,
                                     scene_complete and len(groups) > 1))
    eligible = tuple(row for row in rows if row.discriminating)
    best = ()
    if complete and eligible:
        score = min(row.worst_partition_size for row in eligible)
        best = tuple(row.scene_id for row in eligible if row.worst_partition_size == score)
    elif complete:
        unresolved.append('no_discriminating_scene')
    return GroundingInvestigation(model.id, description, offered, tuple(rows), best, complete,
                                  tuple(dict.fromkeys(unresolved)))
