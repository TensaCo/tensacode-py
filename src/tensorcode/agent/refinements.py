"""Inspectable domain knowledge refining desired outcomes into concrete conditions.

Recipes are explicit, hand-authored conventions, not learned knowledge. Matching
uses predicates and Entity features, never utterance strings. The tiny expression
language supports literals, binding lookup, and path joining; it cannot run code.
Unconsumed qualifiers and ambiguous applicable recipes fail closed.
"""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from ..goals import Condition, GoalSpec
from ..language.semantics import Entity
from ..language.verbnet import Goal, role_class
from ..outcomes import Unknown


def _lookup(value: Any, path: str) -> Any:
    for part in path.split("."):
        if isinstance(value, Entity):
            value = value.features[part]
        elif isinstance(value, Mapping):
            value = value[part]
        else:
            raise KeyError(path)
    return value


def _scalar(value: Any) -> Any:
    if isinstance(value, Entity):
        if not value.resolved or value.features.keys() - {"noun", "definite"}:
            raise ValueError("a binding contains unresolved or unconsumed entity features")
        return value.features.get("noun", value.text)
    return value


def _binding_value(value: Any, spec: Mapping[str, Any]) -> Any:
    extraction = spec.get("extraction", "scalar")
    if extraction == "scalar":
        return _scalar(value)
    if extraction != "text":
        raise ValueError(f"unsupported binding extraction: {extraction}")
    if isinstance(value, Entity):
        accepted = set(spec.get("accepted_features", ()))
        if not value.resolved or value.features.keys() - accepted:
            raise ValueError("a text binding contains unresolved or unconsumed entity features")
        return value.text
    if not isinstance(value, (str, Path)):
        raise ValueError("text extraction requires an entity, string, or Path")
    return value


def _expression(expr: Any, bindings: Mapping[str, Any]) -> Any:
    if not isinstance(expr, dict):
        return expr
    if set(expr) == {"binding"}:
        return bindings[expr["binding"]]
    if set(expr) == {"path"}:
        parts = [_expression(part, bindings) for part in expr["path"]]
        if not parts or any(not isinstance(p, (str, Path)) for p in parts):
            raise ValueError("path expressions require string or Path components")
        return Path(*parts)
    raise ValueError(f"unsupported recipe expression: {expr!r}")


class RefinementLibrary:
    """Load finite declarative recipes; preserve all non-refined conditions.

    A recipe selects a predicate/role and constrains Entity features. ``bindings``
    read dotted paths from ``entity`` or ``context``; optional defaults are
    explicit domain assumptions. A binding may request ``extraction: text`` with
    an explicit ``accepted_features`` list to retain spelling rather than a noun
    lemma; other entity qualifiers still block refinement. ``derived`` binds finite expressions in order.
    ``outputs`` declare predicates/roles, independent of execution capabilities.
    ``implicit_lexical_conditions`` may explicitly identify a lexical resource's
    relation to an unsaid participant (e.g. an artifact's unspecified material).
    Such omission is recorded in the basis and applies only to lexical Goals;
    explicit or otherwise unhandled partial relations are never silently lost.
    """

    def __init__(self, recipes: list[Mapping[str, Any]], *, source: str = "in-memory"):
        self.recipes = deepcopy(recipes)
        self.source = source
        ids = [recipe["id"] for recipe in recipes]
        if len(ids) != len(set(ids)):
            raise ValueError("refinement recipe IDs must be unique")

    @classmethod
    def load(cls, path: str | Path) -> RefinementLibrary:
        path = Path(path)
        document = json.loads(path.read_text())
        if document.get("version") != 1:
            raise ValueError("unsupported refinement library version")
        return cls(document["recipes"], source=str(path))

    def _apply(self, recipe: Mapping[str, Any], condition: Condition,
               context: Mapping[str, Any], frame_roles: Mapping[str, Any]) -> tuple[tuple[Condition, ...], set[str]] | None:
        match = recipe["match"]
        if condition.pred != match["predicate"] or condition.negated != match.get("negated", False):
            return None
        roles = [(role_class(k), v) for k, v in condition.args.items()]
        if len(roles) != 1 or roles[0][0] != role_class(match["role"]):
            return None
        entity = roles[0][1]
        if not isinstance(entity, Entity) or not entity.resolved:
            return None
        for feature, required in match.get("features", {}).items():
            if feature not in entity.features or _scalar(entity.features[feature]) != required:
                return None
        semantic_features = set(match.get("features", {}))
        for binding in recipe.get("bindings", {}).values():
            for source in [binding["from"], *binding.get("fallbacks", ())]:
                path = source.split(".")
                if path[0] == "entity" and len(path) > 1:
                    semantic_features.add(path[1])
        covered = semantic_features | set(recipe.get("accepted_features", ()))
        if "modifiers" in entity.features:
            aliases = recipe.get("modifier_aliases", {})
            seen = set()
            for relation, value in entity.features["modifiers"]:
                alias = aliases.get(relation)
                if alias is None or alias not in semantic_features or alias not in entity.features or alias in seen or value != entity.features[alias]:
                    raise ValueError(f"unconsumed or repeated modifier: {relation}")
                seen.add(alias)
            covered.add("modifiers")
        if entity.features.keys() - covered:
            raise ValueError(f"unconsumed entity features: {sorted(entity.features.keys() - covered)}")
        bindings: dict[str, Any] = {}
        consumed_roles = set()
        for name, spec in recipe.get("bindings", {}).items():
            found = []
            for source in [spec["from"], *spec.get("fallbacks", ())]:
                try:
                    value = _lookup({"entity": entity, "context": context, "frame": {"roles": frame_roles}}, source)
                except KeyError:
                    continue
                found.append(_binding_value(value, spec))
                if source.startswith("frame.roles."):
                    consumed_roles.add(source.split(".")[2])
            if not found:
                if "default" not in spec:
                    raise ValueError(f"missing required binding: {name}") from None
                found.append(_binding_value(spec["default"], spec))
            if any(value != found[0] for value in found[1:]):
                raise ValueError(f"conflicting binding sources: {name}")
            bindings[name] = found[0]
        for name, expression in recipe.get("derived", {}).items():
            bindings[name] = _expression(expression, bindings)
        outputs = tuple(Condition(item["predicate"],
                                  {role: _expression(expr, bindings) for role, expr in item["roles"].items()},
                                  item.get("negated", False)) for item in recipe["outputs"])
        if not outputs:
            raise ValueError("a refinement must specify at least one output condition")
        return outputs, consumed_roles

    def refine(self, goal: Any, *, context: Mapping[str, Any]) -> GoalSpec | Unknown:
        conditions: list[Condition] = []
        basis = list(getattr(goal, "basis", ()))
        changed = False
        applied = []
        consumed_roles = set()
        for condition in goal.conditions:
            matches = []
            errors = []
            for recipe in self.recipes:
                try:
                    result = self._apply(recipe, condition, context, goal.frame.roles if isinstance(goal, Goal) else {})
                except (KeyError, TypeError, ValueError) as error:
                    errors.append(f"{recipe['id']}: {error}")
                    continue
                if result is not None:
                    matches.append((recipe, result))
            if len(matches) > 1:
                return Unknown("ambiguous_refinement", ", ".join(recipe["id"] for recipe, _ in matches))
            if not matches:
                if errors:
                    return Unknown("incomplete_refinement", "; ".join(errors))
                conditions.append(condition)
                continue
            recipe, (outputs, bound_roles) = matches[0]
            consumed_roles.update(bound_roles)
            conditions.extend(outputs)
            applied.append((recipe, next(iter(condition.args.values()))))
            basis.append(f"hand-authored:{recipe['id']}:{self.source}")
            changed = True
        if not changed:
            return Unknown("no_refinement", "No domain recipe applies to these conditions")
        if isinstance(goal, Goal):
            unmapped = set(goal.unmapped_roles) - consumed_roles
            if unmapped:
                return Unknown("incomplete_refinement", f"Unmapped lexical roles: {sorted(unmapped)}")
            # An imperative is the grammatical wrapper for the desired state.
            # Other semantic frame features have no refinement semantics yet.
            remaining_features = dict(goal.frame.features)
            if remaining_features.get("mood") == "imperative":
                remaining_features.pop("mood")
            if remaining_features:
                return Unknown("incomplete_refinement", f"Unconsumed frame features: {remaining_features}")
            retained = []
            for condition in conditions:
                omitted = False
                for recipe, target in applied:
                    for pattern in recipe.get("implicit_lexical_conditions", ()):
                        if not {"unbound", "implicit_addressee"}.intersection(pattern["roles"].values()):
                            continue  # A fully specified condition is never implicit.
                        if condition.pred != pattern["predicate"] or condition.negated != pattern.get("negated", False):
                            continue
                        if set(condition.args) != set(pattern["roles"]):
                            continue
                        if all((condition.args[role] is None if kind == "unbound" else
                                condition.args[role] == "addressee" if kind == "implicit_addressee" else
                                condition.args[role] == target if kind == "matched" else False)
                               for role, kind in pattern["roles"].items()):
                            basis.append(f"implicit-lexical-condition:{recipe['id']}:{condition.describe()}")
                            omitted = True
                            break
                    if omitted:
                        break
                if not omitted:
                    retained.append(condition)
            conditions = retained
        try:
            return GoalSpec(tuple(conditions), label=getattr(goal, "label", ""),
                            invariants=getattr(goal, "invariants", ()), basis=tuple(basis))
        except ValueError as error:
            return Unknown("incomplete_refinement", str(error))
