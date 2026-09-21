"""Bounded research over caller-supplied local text documents."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import json
import os
from pathlib import Path
import re
from typing import Any

from tensorcode.integrations import OpenAICompatibleModel
from tensorcode.ops import llm
from tensorcode.runtime import ActionLoop, ActionOutcome


SUPPORTED_SUFFIXES = frozenset({".txt", ".md", ".rst", ".csv", ".json"})


@dataclass(frozen=True)
class Document:
    source_id: str
    relative_path: str


@dataclass(frozen=True)
class Source:
    source_id: str
    relative_path: str
    text: str


@dataclass(frozen=True)
class ResearchState:
    question: str
    matches: tuple[str, ...] = ()
    sources: tuple[Source, ...] = ()
    answer: str | None = None


@dataclass(frozen=True)
class ResearchReport:
    answer: str | None
    stop_reason: str
    sources: tuple[Source, ...]
    receipts: tuple[Any, ...]


def _documents(root: Path, max_documents: int) -> tuple[Document, ...]:
    paths = []
    for path in root.rglob("*"):
        resolved = path.resolve()
        relative = path.relative_to(root)
        if (path.is_file() and not path.is_symlink()
                and not any(part.startswith(".") for part in relative.parts)
                and path.suffix.casefold() in SUPPORTED_SUFFIXES
                and resolved.is_relative_to(root)):
            paths.append(relative.as_posix())
    paths.sort()
    if not paths:
        raise ValueError("document directory contains no supported documents")
    if len(paths) > max_documents:
        raise ValueError(f"document directory contains {len(paths)} supported documents; "
                         f"increase max_documents above {max_documents} explicitly")
    return tuple(Document(f"doc-{index:04d}", path) for index, path in enumerate(paths, 1))


def _read(root: Path, document: Document, max_chars: int) -> str:
    path = (root / document.relative_path).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"document escaped the configured directory: {document.relative_path}")
    with path.open(encoding="utf-8") as stream:
        return stream.read(max_chars)


def research(docs_dir: str | Path, question: str, *, model: Any,
             max_steps: int = 6, max_documents: int = 40,
             max_chars: int = 8_000) -> ResearchReport:
    root = Path(docs_dir).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError("docs_dir must be a directory")
    if not question.strip():
        raise ValueError("question must be nonempty")
    if isinstance(max_documents, bool) or max_documents < 1:
        raise ValueError("max_documents must be a positive integer")
    if isinstance(max_chars, bool) or max_chars < 1:
        raise ValueError("max_chars must be a positive integer")
    documents = _documents(root, max_documents)

    def search_action(state: ResearchState) -> ActionOutcome:
        terms = tuple(set(re.findall(r"[a-z0-9]+", state.question.casefold())))
        ranked = []
        for document in documents:
            text = _read(root, document, max_chars * 4).casefold()
            score = sum(text.count(term) for term in terms)
            ranked.append((-score, document.relative_path, document.source_id))
        matches = tuple(item[2] for item in sorted(ranked)[: min(5, len(ranked))])
        return ActionOutcome(replace(state, matches=matches), {"matches": matches})

    def read_action(document: Document):
        def run(state: ResearchState) -> ActionOutcome:
            if any(source.source_id == document.source_id for source in state.sources):
                return ActionOutcome(state, {"source_id": document.source_id, "cached": True})
            source = Source(document.source_id, document.relative_path,
                            _read(root, document, max_chars))
            effect = {"source_id": source.source_id, "relative_path": source.relative_path}
            return ActionOutcome(replace(state, sources=state.sources + (source,)), effect)

        return run

    def finish_action(state: ResearchState) -> ActionOutcome:
        if not state.sources:
            return ActionOutcome(state, {"answered": False, "sources": ()}, done=True)
        evidence = "\n\n".join(
            f"[SOURCE {source.source_id}: {source.relative_path}]\n{source.text}"
            for source in state.sources
        )
        prompt = (
            "Answer the question using only the supplied source excerpts. "
            "Cite source IDs in square brackets. If the excerpts do not answer it, say so.\n\n"
            f"Question: {state.question}\n\n{evidence}"
        )
        response = llm.Transform(model)((llm.Message("user", prompt),))
        answer = llm.TextDecoder()(response)
        known = {source.source_id for source in state.sources}
        citations = {
            value for value in re.findall(r"\[([^\[\]]+)\]", answer)
            if value.startswith("doc-")
        }
        unknown = citations - known
        if unknown:
            raise ValueError(f"answer cited unknown source IDs: {sorted(unknown)}")
        if not citations:
            raise ValueError("answer must cite at least one read source ID")
        effect = {"answered": True, "sources": tuple(sorted(known))}
        return ActionOutcome(replace(state, answer=answer), effect, done=True)

    actions = {"search": search_action}
    actions.update({f"read:{doc.source_id}": read_action(doc) for doc in documents})
    actions["finish"] = finish_action
    decide = llm.Decide(model, options=tuple(actions),
        instructions=(
            "Choose exactly one supplied action. Search ranks local documents; read actions "
            "open only their fixed file; finish answers only from read excerpts."
        ),
    )

    def choose(request, *, context=None):
        state = request.state
        manifest = {
            "question": state.question,
            "step": request.step,
            "documents": [asdict(document) for document in documents],
            "search_matches": list(state.matches),
            "read_sources": [
                {"source_id": source.source_id, "relative_path": source.relative_path,
                 "excerpt": source.text[:1000]} for source in state.sources
            ],
            "options": list(request.options),
        }
        return decide((llm.Message("user", json.dumps(manifest, sort_keys=True)),))

    initial = ResearchState(question.strip())
    result = ActionLoop(chooser=choose, actions=actions, max_steps=max_steps)(initial)
    return ResearchReport(result.state.answer, result.stop_reason, result.state.sources, result.receipts)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Research a bounded directory of local text files.")
    parser.add_argument("docs_dir", type=Path)
    parser.add_argument("question")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--max-documents", type=int, default=40)
    parser.add_argument("--max-chars", type=int, default=8_000)
    args = parser.parse_args(argv)
    model = OpenAICompatibleModel(base_url=args.base_url, model=args.model,
                                  api_key=os.environ.get(args.api_key_env))
    report = research(args.docs_dir, args.question, model=model, max_steps=args.max_steps,
                      max_documents=args.max_documents, max_chars=args.max_chars)
    print(json.dumps({"answer": report.answer, "stop_reason": report.stop_reason,
                      "sources": [s.relative_path for s in report.sources],
                      "receipts": [asdict(r) for r in report.receipts]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
