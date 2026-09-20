"""Propose explicit proposition observations for unresolved scene groundings.

This module does not observe the world, select a question, or invent references.
Predictions append supplied hypothetical evidence to the same partial scene;
unknown referents and existential witnesses remain possible after either answer.
"""
from dataclasses import dataclass, replace
from datetime import date, datetime
from uuid import uuid4

from ..records import Interval, Proposition, Ref
from .graph_evidence import QueryEvidence, _ground_root, _variables, assess_query
from .graph_queries import _Budget, _Exhausted, _encode
from .scene_grounding import LearnedQuery, SceneGroundingModel, _description


@dataclass(frozen=True)
class GroundingProbe:
    id: str
    proposition: Proposition
    query_ids: tuple[str, ...]
    atom_indices: tuple[tuple[str, int], ...]
    positive_evidence: tuple[tuple[str, QueryEvidence], ...]
    negative_evidence: tuple[tuple[str, QueryEvidence], ...]


@dataclass(frozen=True)
class GroundingProbePlan:
    model_id: str
    scene_id: Ref
    root: Ref
    probes: tuple[GroundingProbe, ...]
    complete: bool
    unresolved: tuple[str, ...] = ()
    query_evidence: tuple[tuple[str, QueryEvidence], ...] = ()
    query_info: tuple[LearnedQuery, ...] = ()
    explored: int = 0


class _Incomplete(Exception):
    pass


def _decode(value, budget):
    budget.tick()
    if type(value) is not tuple or not value:
        raise ValueError('invalid encoded probe value')
    tag, *parts = value
    if tag == 'none' and not parts:
        return None
    if tag == 'str' and len(parts) == 1 and type(parts[0]) is str:
        return parts[0]
    if tag == 'bool' and len(parts) == 1 and parts[0] in ('true', 'false'):
        return parts[0] == 'true'
    if tag == 'int' and len(parts) == 1:
        return int(parts[0])
    if tag == 'float' and len(parts) == 1:
        return float.fromhex(parts[0])
    if tag == 'ref' and len(parts) == 1:
        return Ref(parts[0])
    if tag == 'date' and len(parts) == 1:
        return date.fromisoformat(parts[0])
    if tag == 'datetime' and len(parts) == 2:
        return datetime.fromisoformat(parts[0]).replace(fold=parts[1])
    if tag in ('list', 'tuple') and len(parts) == 1 and type(parts[0]) is tuple:
        items = tuple(_decode(item, budget) for item in parts[0])
        return list(items) if tag == 'list' else items
    if tag == 'dict' and len(parts) == 1 and type(parts[0]) is tuple:
        return {_decode(key, budget): _decode(item, budget) for key, item in parts[0]}
    if tag == 'Interval' and len(parts) == 2:
        return Interval(*(_decode(part, budget) for part in parts))
    if tag == 'Proposition' and len(parts) == 6:
        return Proposition(*(_decode(part, budget) for part in parts))
    raise ValueError('unsupported encoded probe value')


def _decode_exact(value, budget):
    decoded = _decode(value, budget)
    if _encode(decoded, None, budget, set()) != value:
        raise ValueError('probe value does not round-trip exactly')
    return decoded


def _opposite(atom):
    polarity = ('bool', 'false' if atom[3] == ('bool', 'true') else 'true')
    return (*atom[:3], polarity, *atom[4:])


def propose_grounding_probes(model, description, scene, root, *, max_probes=64, max_states=65536):
    """Return alternative missing-atom questions and both answer predictions.

    Positive evidence means observing the exact question proposition; negative
    evidence flips its top-level polarity. No question is automatically chosen.
    Computationally incomplete plans expose no usable probes. Semantic unknowns
    can remain in a complete plan, including missing relational witnesses that
    cannot be named without inventing a new entity.
    """
    from ..agent.scene import SceneGraph

    if type(max_probes) is not int or max_probes < 1:
        raise ValueError('max_probes must be a positive integer')
    budget = _Budget(max_states)
    queries, evidence, unresolved = (), [], []
    model_id = model.id if type(model) is SceneGroundingModel else ''
    scene_id = scene.image if type(scene) is SceneGraph else None

    def result(probes=(), complete=False):
        return GroundingProbePlan(model_id, scene_id, root, tuple(probes), complete,
            tuple(dict.fromkeys(unresolved)), tuple(evidence), queries, budget.used)

    def assess(query, graph):
        remaining = budget.limit - budget.used
        if remaining < 1:
            raise _Exhausted()
        assessment = assess_query(query, graph, max_matches=remaining, max_states=remaining)
        budget.used += assessment.explored
        if not assessment.complete:
            unresolved.extend(assessment.unresolved or ('incomplete_query_assessment',))
            raise _Incomplete()
        return assessment

    try:
        if type(model) is not SceneGroundingModel or type(scene) is not SceneGraph:
            raise ValueError('probe planning requires a scene grounding model and scene graph')
        if type(root) is not Ref or root not in (scene.image, *scene.nodes):
            raise ValueError('probe root must be a declared scene reference')
        if not model.complete:
            unresolved.extend(model.unresolved)
            unresolved.append('incomplete_grounding_model')
            return result()
        description_key = _description(description)
        _variables(description_key, budget)
        selected_queries = []
        for query in model.queries:
            budget.tick()
            if query.description == description_key:
                selected_queries.append(query)
        queries = tuple(selected_queries)
        if not queries:
            unresolved.append('unknown_description_or_no_supported_query')
            return result(complete=True)
        facts = set()
        for fact in scene.propositions:
            facts.add(_encode(fact, None, budget, set()))
        questions = {}
        for learned in queries:
            assessment = assess(learned.query, scene)
            evidence.append((learned.id, assessment))
            unresolved.extend(learned.id + ':' + reason for reason in assessment.unresolved)
            root_evidence = None
            for row in assessment.roots:
                budget.tick()
                if row.reference == root:
                    root_evidence = row
                    break
            if root_evidence is None:
                raise ValueError('root assessment missing')
            if root_evidence.status != 'unknown':
                continue
            for index, atom in enumerate(learned.query.atoms):
                if not _variables(atom, budget) <= {0}:
                    unresolved.append('unresolved_relational_witness:' + learned.id + ':' + str(index))
                    continue
                grounded = _ground_root(atom, root, budget)
                budget.tick()
                if grounded in facts or _opposite(grounded) in facts:
                    continue
                if grounded not in questions and len(questions) >= max_probes:
                    unresolved.append('probe_limit')
                    return result()
                questions.setdefault(grounded, []).append((learned.id, index))
        probes = []
        for encoded, origins in questions.items():
            proposition = _decode_exact(encoded, budget)
            if type(proposition) is not Proposition:
                raise ValueError('probe must preserve a full proposition')
            query_ids = tuple(dict.fromkeys(ident for ident, _ in origins))
            positive_scene = replace(scene, propositions=(*scene.propositions, proposition))
            negative_scene = replace(scene, propositions=(*scene.propositions,
                replace(proposition, polarity=not proposition.polarity)))
            positive, negative = [], []
            # A new fact can also affect a competing existential query whose
            # missing root-only conjunct did not originate this question.
            for learned in queries:
                positive.append((learned.id, assess(learned.query, positive_scene)))
                negative.append((learned.id, assess(learned.query, negative_scene)))
            probes.append(GroundingProbe('grounding-probe:' + uuid4().hex, proposition,
                query_ids, tuple(origins), tuple(positive), tuple(negative)))
        return result(probes, True)
    except _Incomplete:
        return result()
    except _Exhausted:
        unresolved.append('state_budget')
        return result()
    except (ValueError, TypeError, RecursionError, OverflowError) as error:
        unresolved.append(str(error))
        return result()
