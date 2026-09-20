"""Evidence-backed, occurrence-specific grounding proposals.

The caller supplies references and justification. This mechanism neither infers
identity from descriptions nor asserts that a proposed binding is true. Each call
adds an alternative; selection, rejection, and belief admission remain separate.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from copy import deepcopy
import json
from typing import Any

from ..language import Entity, Frame, Question, Request
from ..records import Ref
from .interpretation import InterpretationCandidate, InterpretationWorkspace
from .understand import Act, SentenceAlternative


@dataclass(frozen=True)
class MentionBinding:
    """Bind one occurrence, e.g. ``('acts', 0, 'frame', 'roles', 'object')``.

    Evidence IDs name retained workspace sources, not arbitrary citations. Basis
    records the caller's justification, not a confidence score or proof.
    """

    path: tuple[str | int, ...]
    reference: Ref
    evidence_ids: tuple[str, ...]
    basis: str

    def __post_init__(self) -> None:
        if not isinstance(self.path, tuple) or len(self.path) < 4:
            raise ValueError("a binding requires an occurrence path")
        if any(type(part) not in (str, int) for part in self.path):
            raise TypeError("path components must be strings or integer indices")
        if self.path[0] != "acts" or type(self.path[1]) is not int or self.path[2] != "frame":
            raise ValueError("binding paths must start with ('acts', index, 'frame')")
        if not isinstance(self.reference, Ref):
            raise TypeError("a binding requires an explicit Ref")
        if not isinstance(self.evidence_ids, tuple) or not self.evidence_ids or any(
            not isinstance(item, str) or not item for item in self.evidence_ids
        ):
            raise ValueError("a binding requires nonempty source evidence IDs")
        if not isinstance(self.basis, str) or not self.basis.strip():
            raise ValueError("a binding requires a nonempty basis")


def _bind(value: Any, path: tuple[str | int, ...], reference: Ref) -> Any:
    if not path:
        if not isinstance(value, Entity):
            raise ValueError("a binding path must identify an Entity occurrence")
        if value.ref is not None and value.ref != reference:
            raise ValueError("binding conflicts with an existing explicit reference")
        return replace(value, ref=reference, candidates=())
    head, *tail = path
    rest = tuple(tail)
    if isinstance(value, Mapping):
        if head not in value:
            raise ValueError(f"unknown mapping key in binding path: {head!r}")
        return {**value, head: _bind(value[head], rest, reference)}
    if isinstance(value, tuple):
        if type(head) is not int or not 0 <= head < len(value):
            raise ValueError("binding tuple index is out of range")
        return tuple(_bind(item, rest, reference) if i == head else item for i, item in enumerate(value))
    if isinstance(value, (Frame, Entity)):
        allowed = ("roles", "features") if isinstance(value, Frame) else ("features", "candidates")
        if head not in allowed:
            raise ValueError(f"invalid structural binding path field: {head!r}")
        return replace(value, **{head: _bind(getattr(value, head), rest, reference)})
    raise ValueError("binding path does not traverse a semantic structure")


def _with_frame(act: Act, frame: Frame) -> Act:
    if isinstance(act.meaning, (Request, Question)):
        if act.meaning.frame != act.frame:
            raise ValueError("source act has inconsistent meaning and frame")
        meaning = replace(act.meaning, frame=frame)
    elif isinstance(act.meaning, Frame) and act.meaning == act.frame:
        meaning = frame
    else:
        raise ValueError("grounding requires a frame-backed act meaning")
    interpretation = act.interpretation
    if interpretation is not None:
        if interpretation.request.frame != act.frame:
            raise ValueError("source act has inconsistent request interpretation")
        interpretation = replace(interpretation, request=replace(interpretation.request, frame=frame))
    return replace(act, frame=frame, meaning=meaning, interpretation=interpretation)


def propose_grounding(
    workspace: InterpretationWorkspace,
    group_id: str,
    candidate_id: str,
    bindings: Sequence[MentionBinding],
) -> InterpretationCandidate:
    """Append a grounded alternative without changing source or selection.

    All validation precedes the proposal, so failure leaves workspace state
    unchanged. Provenance includes a JSON audit record per supplied occurrence.
    The original parser reading remains retained as evidence of the reader output;
    the alternative's acts carry the proposed semantic revision.
    """
    group = workspace.get(group_id)
    workspace.get_source(group.source_id)
    parent = next((item for item in group.candidates if item.id == candidate_id), None)
    if parent is None or parent.group_id != group_id:
        raise ValueError("candidate does not belong to the supplied group")
    if not isinstance(parent.payload, SentenceAlternative):
        raise TypeError("grounding requires a SentenceAlternative candidate")
    bindings = tuple(bindings)
    if not bindings or any(not isinstance(binding, MentionBinding) for binding in bindings):
        raise ValueError("grounding requires MentionBinding records")
    paths = [binding.path for binding in bindings]
    if len(set(paths)) != len(paths):
        raise ValueError("a proposal may bind each occurrence only once")
    acts = list(parent.payload.acts)
    audit = []
    for binding in bindings:
        for source_id in binding.evidence_ids:
            workspace.get_source(source_id)
        index = binding.path[1]
        if not 0 <= index < len(acts):
            raise ValueError("binding act index is out of range")
        act = acts[index]
        frame = _bind(act.frame, binding.path[3:], binding.reference)
        acts[index] = _with_frame(act, frame)
        audit.append("grounding:" + json.dumps({
            "path": binding.path, "reference": binding.reference.id,
            "evidence_ids": binding.evidence_ids, "basis": binding.basis,
        }, sort_keys=True))
    child = workspace.propose(
        group_id, replace(parent.payload, acts=tuple(acts)),
        provenance=parent.provenance + (f"grounding-parent:{parent.id}", f"grounding-source:{group.source_id}", *audit),
    )
    # Keep authenticated structural ancestry separately from human-readable
    # provenance so successive bindings retain their learned support dependencies.
    if not hasattr(workspace, "_grounding_derivations"):
        workspace._grounding_derivations = {}
    workspace._grounding_derivations[child.id] = (group_id, parent.id, deepcopy(child))
    return child
