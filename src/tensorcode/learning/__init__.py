"""Inducing readable artifacts from traces, and the controls that decide adoption.

    from tensorcode.learning import candidate_literals, decision_list, verify_decision_list, Library

    literals = candidate_literals(train)                      # the space, read off the data
    rules    = decision_list(train, literals)                 # MDL-stopped, printable
    check    = verify_decision_list(rules, train=train, held_out=test, literals=literals)
    if check.adopted:                                         # four controls, not one fit
        Library(root).publish("router", rules, provenance={...}, fixture=test[:5])

The controls (held-out, same-size random, wrong-question, consistent renaming) and
the "a fit is not a finding" discipline come from the user's own repos; see
``docs/revival/09-language-and-induction.md``.
"""

from .certificate import MISSING, ReadSet, Reader, certified, revalidate, value_digest
from .induce import (
    DecisionList, Precondition, Rule, RoleType, decision_list, effects, preconditions, role_type, shape,
)
from .library import Entry, FixtureMismatch, Library, LibraryError, MissingArtifact, digest_of, from_json, to_json
from .literals import Case, Literal, candidate_literals, facts_of, rename_case, rename_facts, rename_map
from .verify import Concept, ConceptCheck, Verification, check_concept, propose_concepts, verify_decision_list

__all__ = [
    "MISSING", "ReadSet", "Reader", "certified", "revalidate", "value_digest",
    "Case", "Concept", "ConceptCheck", "DecisionList", "Entry", "FixtureMismatch", "Library", "LibraryError",
    "Literal", "MissingArtifact", "Precondition", "RoleType", "Rule", "Verification", "candidate_literals",
    "check_concept", "decision_list", "digest_of", "effects", "facts_of", "from_json", "preconditions",
    "propose_concepts", "rename_case", "rename_facts", "rename_map", "role_type", "shape", "to_json", "verify_decision_list",
]
