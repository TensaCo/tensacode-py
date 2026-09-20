"""Bounded connected partial witnesses over supplied graph evidence.

Variables retain their original query indices. Bindings grow only through actual
facts joining an already bound variable; no node Cartesian product or invented
reference can supply a missing existential witness. Partiality is not negation.
"""
from dataclasses import dataclass

from ..agent.evidence_graph import graph_root
from ..records import Ref
from .graph_evidence import _variables
from .graph_queries import (
    GraphQuery, _Budget, _Exhausted, _scene, _unify, _validate_encoded,
)


@dataclass(frozen=True)
class PartialGraphMatch:
    bindings: tuple[tuple[int, Ref], ...]
    supporting: tuple[tuple[int, int], ...]
    remaining: tuple[int, ...]
    conflicts: tuple[tuple[int, tuple[int, ...]], ...] = ()


@dataclass(frozen=True)
class PartialQueryMatches:
    matches: tuple[PartialGraphMatch, ...]
    complete: bool
    unresolved: tuple[str, ...] = ()
    explored: int = 0
    pending: int = 0


def match_partial_query(query, scene, root, *, max_matches=128, max_states=2048):
    """Retain all bounded connected partial joins, including the root seed.

    Supporting pairs are ``(query_atom_index, scene_fact_index)``. Conflicts use
    ``(supporting_fact_index, exact_opposite_fact_indices)``. Contradictory joins
    remain diagnostic witnesses. Search completeness says nothing about unseen
    scene entities or absent facts. Both failed joins and preprocessing consume
    the same work budget; stopping exposes explicit incomplete search.
    """
    if type(max_matches) is not int or max_matches < 1:
        raise ValueError('max_matches must be a positive integer')
    budget = _Budget(max_states)
    matches, unresolved = [], []
    try:
        if (type(query) is not GraphQuery or type(query.atoms) is not tuple or not query.atoms
                or type(query.variable_count) is not int or query.variable_count < 1):
            raise ValueError('invalid graph query')
        variables = set()
        atom_variables = []
        for atom in query.atoms:
            _validate_encoded(atom, variables, budget)
            if atom[0] != 'Proposition':
                raise ValueError('query atoms must be encoded propositions')
            atom_variables.append(_variables(atom, budget))
        if variables != set(range(query.variable_count)):
            raise ValueError('invalid graph query variables')
        facts, _ = _scene(scene, budget)
        if type(root) is not Ref or root not in (graph_root(scene), *scene.nodes):
            raise ValueError('partial query root must be a declared scene reference')
        index = {}
        for fact_index, fact in enumerate(facts):
            budget.tick()
            index.setdefault(fact, []).append(fact_index)
        opposites = {}
        for fact, indices in index.items():
            budget.tick()
            polarity = ('bool', 'false' if fact[3] == ('bool', 'true') else 'true')
            opposites[(*fact[:3], polarity, *fact[4:])] = tuple(indices)
        initial = (((0, root),), ())
        frontier, seen = [initial], {initial}
        while frontier:
            budget.tick()
            bindings, supporting = frontier.pop()
            bound = dict(bindings)
            used_atoms, used_facts = set(), set()
            conflicts = []
            for atom_index, fact_index in supporting:
                budget.tick()
                used_atoms.add(atom_index)
                used_facts.add(fact_index)
                opposition = opposites.get(facts[fact_index], ())
                if opposition:
                    conflicts.append((fact_index, opposition))
            remaining = []
            for atom_index in range(len(query.atoms)):
                budget.tick()
                if atom_index not in used_atoms:
                    remaining.append(atom_index)
            matches.append(PartialGraphMatch(bindings, supporting, tuple(remaining), tuple(conflicts)))
            if conflicts and 'contradictory_match_evidence' not in unresolved:
                unresolved.append('contradictory_match_evidence')
            for atom_index in remaining:
                budget.tick()
                if not atom_variables[atom_index].intersection(bound):
                    continue
                for fact_index, fact in enumerate(facts):
                    budget.tick()
                    if fact_index in used_facts:
                        continue
                    for updated in _unify(query.atoms[atom_index], fact, bound, budget):
                        budget.tick()
                        joined = (tuple(sorted(updated.items())),
                                  tuple(sorted((*supporting, (atom_index, fact_index)))))
                        if joined not in seen:
                            seen.add(joined)
                            frontier.append(joined)
            if len(matches) >= max_matches and frontier:
                return PartialQueryMatches(tuple(matches), False, (*unresolved, 'match_limit'),
                                           budget.used, len(frontier))
        return PartialQueryMatches(tuple(matches), True, tuple(unresolved), budget.used)
    except _Exhausted:
        return PartialQueryMatches(tuple(matches), False, (*unresolved, 'state_budget'), budget.used, 1)
    except (ValueError, TypeError, RecursionError) as error:
        return PartialQueryMatches(tuple(matches), False, (*unresolved, str(error)), budget.used, 1)
