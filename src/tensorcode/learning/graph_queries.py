"""Bounded, typed graph-pattern abstraction and conjunctive matching.

These operations inspect supplied propositions, not pixels. Negated facts must
be explicitly present. Predicate/role names have no built-in domain semantics.
Every source reference becomes an injective variable; variable zero is the root.
"""
from dataclasses import dataclass
from datetime import date, datetime
from itertools import permutations
import math

from ..outcomes import Unknown
from ..records import Interval, Proposition, Ref


@dataclass(frozen=True)
class GraphQuery:
    atoms: tuple[tuple, ...]
    variable_count: int


@dataclass(frozen=True)
class GraphMatch:
    bindings: tuple[Ref, ...]
    fact_indices: tuple[int, ...]
    # Supporting fact index -> exact opposite-polarity fact indices.
    conflicts: tuple[tuple[int, tuple[int, ...]], ...] = ()


@dataclass(frozen=True)
class QueryEnumeration:
    patterns: tuple[GraphQuery, ...]
    complete: bool
    unresolved: tuple[str, ...] = ()
    explored: int = 0
    pending: int = 0


@dataclass(frozen=True)
class QueryMatches:
    matches: tuple[GraphMatch, ...]
    complete: bool
    unresolved: tuple[str, ...] = ()
    explored: int = 0
    pending: int = 0


class _Exhausted(Exception):
    pass


class _Budget:
    def __init__(self, limit):
        if type(limit) is not int or limit < 1:
            raise ValueError('max_states must be a positive integer')
        self.limit, self.used = limit, 0

    def tick(self):
        if self.used >= self.limit:
            raise _Exhausted()
        self.used += 1


def encode_value(value, refs=None):
    """Return a hashable typed value; a supplied reference list enables variables.

    References are literal IDs when ``refs`` is None. Lists/tuples, bool/int,
    date/datetime, scope, validity, modality and polarity remain distinct.
    Unsupported opaque values raise ValueError; no repr/string identity fallback.
    """
    return _encode(value, refs, None, set())


def _encode(value, refs, budget, active):
    if budget is not None:
        budget.tick()
    kind = type(value)
    if kind is Ref:
        if refs is None:
            return ('ref', value.id)
        if value not in refs:
            refs.append(value)
        return ('var', refs.index(value))
    if value is None:
        return ('none',)
    if kind in (str, bool, int):
        return (kind.__name__, value if kind is str else ('true' if value else 'false') if kind is bool else str(value))
    if kind is float:
        if not math.isfinite(value):
            raise ValueError('nonfinite graph value')
        return ('float', value.hex())
    if kind is datetime:
        return ('datetime', value.isoformat(), value.fold)
    if kind is date:
        return ('date', value.isoformat())
    if id(value) in active:
        raise ValueError('cyclic graph value')
    active.add(id(value))
    try:
        if kind in (tuple, list):
            return (kind.__name__, tuple(_encode(item, refs, budget, active) for item in value))
        if kind is dict:
            # Literal sorting is independent of insertion order for ordinary
            # scalar role keys. Matching treats nested mappings as unordered.
            ordered = sorted(value.items(), key=lambda item: _encode(item[0], None, budget, active))
            return ('dict', tuple(sorted((_encode(key, refs, budget, active), _encode(item, refs, budget, active))
                                        for key, item in ordered)))
        if kind is Interval:
            return ('Interval', _encode(value.start, refs, budget, active), _encode(value.end, refs, budget, active))
        if kind is Proposition:
            if type(value.roles) is not dict or type(value.predicate) is not str or not value.predicate:
                raise ValueError('unsupported proposition roles or predicate')
            if type(value.polarity) is not bool or type(value.modality) is not str or type(value.valid) is not Interval:
                raise ValueError('invalid proposition metadata')
            if value.scope is not None and type(value.scope) is not Ref:
                raise ValueError('invalid proposition scope')
            return ('Proposition', _encode(value.predicate, refs, budget, active),
                    _encode(value.roles, refs, budget, active), _encode(value.polarity, refs, budget, active),
                    _encode(value.modality, refs, budget, active), _encode(value.valid, refs, budget, active),
                    _encode(value.scope, refs, budget, active))
        raise ValueError('unsupported graph value type: ' + kind.__name__)
    finally:
        active.remove(id(value))


def _query(facts, root, budget):
    if type(root) is not Ref or not facts or any(type(fact) is not Proposition for fact in facts):
        raise ValueError('query requires propositions and an explicit root Ref')
    references = [root]
    encoded = tuple(_encode(fact, references, budget, set()) for fact in facts)
    if not any(_has_variable(atom, 0) for atom in encoded):
        raise ValueError('query root is absent from its facts')
    best = None
    # Canonicalize variable names, including references used as mapping keys.
    # Atom sorting alone cannot remove accidental source reference ordering.
    for ordering in permutations(references[1:]):
        budget.tick()
        refs = [root, *ordering]
        atoms = tuple(sorted(_encode(fact, refs, budget, set()) for fact in facts))
        if best is None or atoms < best:
            best = atoms
    return GraphQuery(best, len(references))


def _has_variable(value, index):
    return type(value) is tuple and (value == ('var', index) or any(_has_variable(item, index) for item in value))


def query_from_facts(facts, root, *, max_states=2048):
    """Abstract supplied facts, canonically numbering variables and ordering atoms within a budget."""
    budget = _Budget(max_states)
    try:
        return _query(tuple(facts), root, budget)
    except _Exhausted:
        return Unknown('graph_query_budget', 'canonical pattern abstraction did not finish')
    except (ValueError, TypeError, RecursionError) as error:
        return Unknown('unsupported_graph_query', str(error))


def _scene(scene, budget):
    from ..agent.evidence_graph import graph_root
    observation_root = graph_root(scene)
    allowed = set()
    for node in (observation_root, *scene.nodes):
        budget.tick()
        if type(node) is not Ref or node in allowed:
            raise ValueError('scene references must be distinct explicit Refs')
        allowed.add(node)
    facts, references = [], []
    for fact in scene.propositions:
        if type(fact) is not Proposition:
            raise ValueError('scene facts must be Propositions')
        refs = []
        _encode(fact, refs, budget, set())
        if not set(refs) <= allowed:
            raise ValueError('scene contains undeclared references')
        facts.append(_encode(fact, None, budget, set()))
        references.append(frozenset(refs))
    return tuple(facts), tuple(references)


def enumerate_rooted_queries(scene, root, *, max_atoms=3, max_patterns=128, max_states=2048):
    """Enumerate nonempty connected fact subsets containing the chosen root.

    Work includes encoding nodes/values, testing extension branches and variable
    canonicalization. ``max_atoms`` defines the requested pattern class; the
    other limits report incomplete search when they stop enumeration. No absent
    facts are manufactured. ``pending`` is a lower bound on unfinished work.
    """
    if type(max_atoms) is not int or max_atoms < 1 or type(max_patterns) is not int or max_patterns < 1:
        raise ValueError('pattern limits must be positive integers')
    budget = _Budget(max_states)
    patterns, seen_patterns = [], set()
    try:
        _, references = _scene(scene, budget)
        from ..agent.evidence_graph import graph_root
        if type(root) is not Ref or root not in {graph_root(scene), *scene.nodes}:
            raise ValueError('root must be a declared scene reference')
        frontier, seen = [], set()
        for index, refs in enumerate(references):
            budget.tick()
            if root in refs:
                subset = (index,)
                frontier.append(subset)
                seen.add(subset)
        while frontier:
            if len(patterns) >= max_patterns:
                return QueryEnumeration(tuple(patterns), False, ('pattern_limit',), budget.used, len(frontier))
            budget.tick()
            subset = frontier.pop(0)
            query = _query(tuple(scene.propositions[i] for i in subset), root, budget)
            if query not in seen_patterns:
                seen_patterns.add(query)
                patterns.append(query)
            if len(subset) == max_atoms:
                continue
            connected = set().union(*(references[i] for i in subset))
            for index, refs in enumerate(references):
                budget.tick()
                if index in subset or not connected.intersection(refs):
                    continue
                extended = tuple(sorted((*subset, index)))
                if extended not in seen:
                    seen.add(extended)
                    frontier.append(extended)
        return QueryEnumeration(tuple(patterns), True, explored=budget.used)
    except _Exhausted:
        return QueryEnumeration(tuple(patterns), False, ('state_budget',), budget.used, 1)
    except (ValueError, TypeError, RecursionError) as error:
        return QueryEnumeration(tuple(patterns), False, (str(error),), budget.used, 1)


def _unify(pattern, fact, bindings, budget):
    budget.tick()
    if not isinstance(pattern, tuple) or not isinstance(fact, tuple) or not pattern or not fact:
        return
    if pattern[0] == 'var':
        if len(pattern) != 2 or type(pattern[1]) is not int or fact[0] != 'ref':
            return
        slot, identity = pattern[1], Ref(fact[1])
        if slot in bindings:
            if bindings[slot] == identity:
                yield bindings
        elif identity not in bindings.values():
            yield {**bindings, slot: identity}
        return
    if pattern[0] != fact[0] or len(pattern) != len(fact):
        return
    if pattern[0] == 'dict':
        left, right = pattern[1], fact[1]
        if len(left) != len(right):
            return
        def entries(index, used, current):
            if index == len(left):
                yield current
                return
            for other, pair in enumerate(right):
                budget.tick()
                if other in used:
                    continue
                for keyed in _unify(left[index][0], pair[0], current, budget):
                    for valued in _unify(left[index][1], pair[1], keyed, budget):
                        yield from entries(index + 1, (*used, other), valued)
        yield from entries(0, (), bindings)
        return
    if pattern[0] in ('tuple', 'list'):
        left, right = pattern[1], fact[1]
    else:
        left, right = pattern[1:], fact[1:]
    if len(left) != len(right):
        return
    def sequence(index, current):
        if index == len(left):
            yield current
            return
        p, f = left[index], right[index]
        if type(p) is tuple and type(f) is tuple:
            for updated in _unify(p, f, current, budget):
                yield from sequence(index + 1, updated)
        elif type(p) is type(f) and p == f:
            yield from sequence(index + 1, current)
    yield from sequence(0, bindings)


def _validate_encoded(value, variables, budget):
    """Reject malformed public query tuples even when no scene fact matches."""
    budget.tick()
    if type(value) is not tuple or not value or type(value[0]) is not str:
        raise ValueError('invalid encoded graph value')
    tag = value[0]
    if tag == 'var':
        if len(value) != 2 or type(value[1]) is not int or value[1] < 0:
            raise ValueError('invalid graph variable')
        variables.add(value[1])
        return
    if tag == 'none' and len(value) == 1:
        return
    if tag in ('str', 'int', 'float', 'bool', 'ref', 'date') and len(value) == 2 and type(value[1]) is str:
        if tag == 'bool' and value[1] not in ('true', 'false'):
            raise ValueError('invalid encoded boolean')
        return
    if tag == 'datetime' and len(value) == 3 and type(value[1]) is str and type(value[2]) is int and value[2] in (0, 1):
        return
    if tag in ('tuple', 'list', 'dict') and len(value) == 2 and type(value[1]) is tuple:
        for item in value[1]:
            if tag == 'dict':
                if type(item) is not tuple or len(item) != 2:
                    raise ValueError('invalid encoded mapping')
                for part in item:
                    _validate_encoded(part, variables, budget)
            else:
                _validate_encoded(item, variables, budget)
        return
    if (tag == 'Interval' and len(value) == 3) or (tag == 'Proposition' and len(value) == 7):
        for part in value[1:]:
            _validate_encoded(part, variables, budget)
        return
    raise ValueError('unsupported encoded graph value')


def match_query(query, scene, *, max_matches=128, max_states=2048):
    """Match a conjunction with injective Ref bindings and exact typed metadata.

    Fact indices follow query atom order. The work budget bounds preprocessing,
    failed candidates and recursive joins as well as successful matches. Hitting
    an output cap never claims that all other matches have been ruled out.
    Exact top-level opposite-polarity facts are retained on each witness, with
    a contradiction diagnostic; they never disappear into a positive match.
    Completeness describes search exhaustion, not consistency or a closed world.
    """
    if type(max_matches) is not int or max_matches < 1:
        raise ValueError('max_matches must be a positive integer')
    budget = _Budget(max_states)
    matches = []
    unresolved = []
    try:
        if (type(query) is not GraphQuery or type(query.atoms) is not tuple or not query.atoms
                or type(query.variable_count) is not int or query.variable_count < 1):
            raise ValueError('invalid graph query')
        variables = set()
        for atom in query.atoms:
            _validate_encoded(atom, variables, budget)
            if atom[0] != 'Proposition':
                raise ValueError('query atoms must be encoded propositions')
        if variables != set(range(query.variable_count)):
            raise ValueError('invalid graph query variables')
        facts, _ = _scene(scene, budget)
        fact_index = {}
        for index, fact in enumerate(facts):
            budget.tick()
            fact_index.setdefault(fact, []).append(index)
        # Tuples are shared across witnesses; no quadratic all-pairs scan.
        opposite_index = {}
        for fact, indices in fact_index.items():
            budget.tick()
            polarity = ('bool', 'false' if fact[3] == ('bool', 'true') else 'true')
            opposite_index[(*fact[:3], polarity, *fact[4:])] = tuple(indices)
        frontier = [(0, {}, ())]
        while frontier:
            budget.tick()
            atom_index, bindings, indices = frontier.pop()
            if atom_index == len(query.atoms):
                if set(bindings) != set(range(query.variable_count)):
                    raise ValueError('invalid graph query variables')
                conflicts = []
                for index in indices:
                    budget.tick()
                    opposing = opposite_index.get(facts[index], ())
                    if opposing:
                        conflicts.append((index, opposing))
                matches.append(GraphMatch(tuple(bindings[i] for i in range(query.variable_count)), indices, tuple(conflicts)))
                if conflicts and 'contradictory_match_evidence' not in unresolved:
                    unresolved.append('contradictory_match_evidence')
                if len(matches) >= max_matches and frontier:
                    return QueryMatches(tuple(matches), False, (*unresolved, 'match_limit'), budget.used, len(frontier))
                continue
            for index, fact in enumerate(facts):
                budget.tick()
                if index in indices:
                    continue
                for updated in _unify(query.atoms[atom_index], fact, bindings, budget):
                    frontier.append((atom_index + 1, updated, (*indices, index)))
        return QueryMatches(tuple(matches), True, tuple(unresolved), explored=budget.used)
    except _Exhausted:
        return QueryMatches(tuple(matches), False, (*unresolved, 'state_budget'), budget.used, 1)
    except (ValueError, TypeError, RecursionError) as error:
        return QueryMatches(tuple(matches), False, (*unresolved, str(error)), budget.used, 1)
