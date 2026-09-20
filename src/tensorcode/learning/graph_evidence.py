"""Bounded open-world evidence over supplied scene propositions.

Exhausting stored facts never closes the scene's reference universe. A missing
existential witness is unknown. Refutation requires an explicit opposite of a
necessary conjunct already grounded by the root alone (or containing no refs).
"""
from dataclasses import dataclass

from ..agent.evidence_graph import graph_root
from ..records import Ref
from .graph_queries import GraphMatch, _Budget, _Exhausted, _encode, match_query


@dataclass(frozen=True)
class RootEvidence:
    reference: Ref
    status: str
    supporting_matches: tuple[GraphMatch, ...] = ()
    refuting_atoms: tuple[tuple[int, tuple[int, ...]], ...] = ()


@dataclass(frozen=True)
class QueryEvidence:
    matches: tuple[GraphMatch, ...]
    roots: tuple[RootEvidence, ...]
    complete: bool
    unresolved: tuple[str, ...] = ()
    explored: int = 0
    unseen_referents_possible: bool = True


def _variables(value, budget):
    budget.tick()
    if type(value) is not tuple:
        return set()
    if len(value) == 2 and value[0] == 'var':
        return {value[1]}
    found = set()
    for part in value:
        found.update(_variables(part, budget))
    return found


def _ground_root(value, reference, budget):
    budget.tick()
    if type(value) is not tuple:
        return value
    if len(value) == 2 and value[0] == 'var':
        return ('ref', reference.id)
    grounded = tuple(_ground_root(part, reference, budget) for part in value)
    # Variable-key mapping order may differ from literal reference-key order.
    if grounded and grounded[0] == 'dict':
        return ('dict', tuple(sorted(grounded[1])))
    return grounded


def assess_query(query, scene, *, max_matches=128, max_states=2048):
    """Assess each declared root without treating absent facts as false.

    ``complete`` describes computation, not knowledge completeness. Conflicted
    witnesses remain visible. ``refuting_atoms`` also retains opposition when
    both polarities are present; such evidence yields ``conflicted``, not false.
    All work shares one budget, including matching and subsequent root checks.
    """
    budget = _Budget(max_states)
    matched = match_query(query, scene, max_matches=max_matches, max_states=max_states)
    budget.used = matched.explored
    roots = []
    unresolved = list(matched.unresolved)
    try:
        # Matching has already validated the query and scene. Unsupported inputs
        # must not enter a second, more permissive evidence path.
        if any(reason not in ('match_limit', 'state_budget', 'contradictory_match_evidence')
               for reason in matched.unresolved):
            return QueryEvidence(matched.matches, (), False, tuple(unresolved), budget.used)
        necessary = []
        for index, atom in enumerate(query.atoms):
            if _variables(atom, budget) <= {0}:
                necessary.append((index, atom))
        fact_index = {}
        for index, fact in enumerate(scene.propositions):
            encoded = _encode(fact, None, budget, set())
            budget.tick()
            fact_index.setdefault(encoded, []).append(index)
        by_root = {}
        for witness in matched.matches:
            budget.tick()
            by_root.setdefault(witness.bindings[0], []).append(witness)
        for reference in (graph_root(scene), *scene.nodes):
            budget.tick()
            witnesses = tuple(by_root.get(reference, ()))
            conflict = False
            for witness in witnesses:
                budget.tick()
                if witness.conflicts:
                    conflict = True
            refuted = False
            opposition = []
            for index, atom in necessary:
                grounded = _ground_root(atom, reference, budget)
                polarity = ('bool', 'false' if grounded[3] == ('bool', 'true') else 'true')
                opposite = (*grounded[:3], polarity, *grounded[4:])
                budget.tick()
                negative = fact_index.get(opposite, ())
                if negative:
                    retained_indices = []
                    for fact_index_ in negative:
                        budget.tick()
                        retained_indices.append(fact_index_)
                    opposition.append((index, tuple(retained_indices)))
                    if grounded in fact_index:
                        conflict = True
                    else:
                        refuted = True
            if conflict or (witnesses and refuted):
                status = 'conflicted'
                if 'contradictory_match_evidence' not in unresolved:
                    unresolved.append('contradictory_match_evidence')
            elif witnesses:
                status = 'supported'
            elif refuted:
                status = 'refuted'
            else:
                status = 'unknown'
            roots.append(RootEvidence(reference, status, witnesses, tuple(opposition)))
        return QueryEvidence(matched.matches, tuple(roots), matched.complete, tuple(unresolved), budget.used)
    except _Exhausted:
        if 'state_budget' not in unresolved:
            unresolved.append('state_budget')
        return QueryEvidence(matched.matches, tuple(roots), False, tuple(unresolved), budget.used)
