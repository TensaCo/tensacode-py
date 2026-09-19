"""Compare learned dependency candidates on a reproducible held-out UD sample.

Runs existing cached weights; no training or downloads. Both decoders receive the
same greedy predicted tags, isolating dependency search from tag search. Oracle
scores use gold annotations to pick a candidate and are upper bounds, not an
implemented interpretation policy. Short-sentence sampling is explicitly biased.

Fixed-token decoder comparison:

    python -m eval.parsing.evaluate_candidates --output eval/results/parsing_candidates.json

For actual reader evaluation over original text and alternative tokenizations:

    python -m eval.parsing.span_evaluation
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import statistics
import sys
import time
from pathlib import Path

from eval.parsing.legacy_baseline import parse as legacy_parse
from tensorcode.language import learned_parser
from tensorcode.language.learned_parser import load_model
from tensorcode.language.treebank import find_treebank, read_conllu


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def counts(sentence, heads, labels):
    tokens = [token for token in sentence if token.upos != "PUNCT"]
    uas = sum(heads.get(t.id) == t.head for t in tokens)
    las = sum(heads.get(t.id) == t.head and labels.get(t.id) == t.deprel for t in tokens)
    return {"tokens": len(tokens), "uas_correct": uas, "las_correct": las,
            "exact_labeled_tree_nonpunct": las == len(tokens),
            "exact_labeled_tree_all_tokens": all(heads.get(t.id) == t.head and labels.get(t.id) == t.deprel for t in sentence)}


def evaluate(args) -> dict:
    implementation_sha256 = digest(Path(learned_parser.__file__))
    implementation_sources = {str(p): digest(p) for p in (
        Path(__file__), Path(__file__).with_name("legacy_baseline.py"),
        Path(learned_parser.__file__).with_name("treebank.py"))}
    root = args.treebank or find_treebank()
    if root is None:
        raise ValueError("No local treebank; this evaluation does not download data")
    files = sorted(root.glob(f"*-ud-{args.split}.conllu"))
    if not files:
        raise ValueError(f"No {args.split} CoNLL-U files under {root}")
    all_rows = [(sid, sentence) for path in files for sid, sentence in read_conllu(path, with_ids=True)]
    eligible = [(sid, s) for sid, s in all_rows if args.min_tokens <= len(s) <= args.max_tokens
                and any(t.upos != "PUNCT" for t in s)]
    excluded_ids = set()
    for report_path in args.exclude_results:
        excluded_ids.update(r["sent_id"] for r in json.loads(report_path.read_text())["sentences"])
    eligible = [(sid, sentence) for sid, sentence in eligible if sid not in excluded_ids]
    selected = random.Random(args.seed).sample(eligible, min(args.sample, len(eligible)))
    if not selected:
        raise ValueError("No eligible sentences")
    loaded = load_model(args.model)
    if loaded is None:
        raise ValueError(f"No local cached model: {args.model}")
    tagger, parser = loaded
    rows = []
    started = time.perf_counter()
    for sid, sentence in selected:
        words = [token.form for token in sentence]
        t0 = time.perf_counter()
        tags = tagger.tag(words)
        tag_ms = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        heads, labels = legacy_parse(parser, words, tags)
        greedy_ms = (time.perf_counter() - t0) * 1000
        greedy = counts(sentence, heads, labels)
        t0 = time.perf_counter()
        search = parser.parse_candidates(words, tags, beam_width=args.beam_width,
                                         max_candidates=args.max_candidates,
                                         max_expansions=args.max_expansions, ranking=args.ranking)
        candidate_ms = (time.perf_counter() - t0) * 1000
        scored = [counts(sentence, c.heads, c.labels) for c in search.candidates]
        valid_greedy = parser.greedy_candidate(words, tags) if hasattr(parser, "greedy_candidate") else None
        valid_scored = counts(sentence, valid_greedy.heads, valid_greedy.labels) if valid_greedy is not None else None
        union_scored = scored + ([valid_scored] if valid_scored is not None else [])
        top = scored[0] if scored else counts(sentence, {}, {})
        oracle_uas = max((c["uas_correct"] for c in scored), default=0)
        oracle_las = max((c["las_correct"] for c in scored), default=0)
        rows.append({"sent_id": sid, "tokens_including_punctuation": len(sentence),
                     "greedy": greedy, "top_candidate": top,
                     "oracle_uas_correct": oracle_uas, "oracle_las_correct": oracle_las,
                     "exact_tree_in_candidates_nonpunct": any(c["exact_labeled_tree_nonpunct"] for c in scored),
                     "exact_tree_in_candidates_all_tokens": any(c["exact_labeled_tree_all_tokens"] for c in scored),
                     "validated_greedy_available": valid_scored is not None,
                     "union_oracle_uas_correct": max((c["uas_correct"] for c in union_scored), default=0),
                     "union_oracle_las_correct": max((c["las_correct"] for c in union_scored), default=0),
                     "union_exact_tree_nonpunct": any(c["exact_labeled_tree_nonpunct"] for c in union_scored),
                     "union_exact_tree_all_tokens": any(c["exact_labeled_tree_all_tokens"] for c in union_scored),
                     "candidate_count": len(scored), "complete": search.complete,
                     "truncated": search.truncated, "expansions": search.expansions, "reason": search.reason,
                     "tag_ms": tag_ms, "greedy_ms": greedy_ms, "candidate_ms": candidate_ms})
        if len(rows) % 25 == 0:
            print(f"Evaluated {len(rows)}/{len(selected)} sentences", file=sys.stderr, flush=True)
    n = len(rows)
    tokens = sum(r["greedy"]["tokens"] for r in rows)
    metrics = {}
    for name in ("greedy", "top_candidate"):
        metrics[name] = {"uas": sum(r[name]["uas_correct"] for r in rows) / tokens,
                         "las": sum(r[name]["las_correct"] for r in rows) / tokens,
                         "exact_tree_recall_nonpunct": sum(r[name]["exact_labeled_tree_nonpunct"] for r in rows) / n,
                         "exact_tree_recall_all_tokens": sum(r[name]["exact_labeled_tree_all_tokens"] for r in rows) / n}
    metrics["oracle_candidates"] = {
        "uas": sum(r["oracle_uas_correct"] for r in rows) / tokens,
        "las": sum(r["oracle_las_correct"] for r in rows) / tokens,
        "exact_tree_recall_nonpunct": sum(r["exact_tree_in_candidates_nonpunct"] for r in rows) / n,
        "exact_tree_recall_all_tokens": sum(r["exact_tree_in_candidates_all_tokens"] for r in rows) / n}
    if hasattr(parser, "greedy_candidate"):
        metrics["oracle_union_with_validated_greedy"] = {
            "uas": sum(r["union_oracle_uas_correct"] for r in rows) / tokens,
            "las": sum(r["union_oracle_las_correct"] for r in rows) / tokens,
            "exact_tree_recall_nonpunct": sum(r["union_exact_tree_nonpunct"] for r in rows) / n,
            "exact_tree_recall_all_tokens": sum(r["union_exact_tree_all_tokens"] for r in rows) / n,
            "validated_greedy_sentences": sum(r["validated_greedy_available"] for r in rows)}
    latency = {}
    for field in ("tag_ms", "greedy_ms", "candidate_ms"):
        values = sorted(r[field] for r in rows)
        latency[field] = {"total": sum(values), "median": statistics.median(values),
                          "p95": values[min(n - 1, int(.95 * n))], "max": max(values)}
    return {"evaluation": "learned_dependency_candidates", "python": platform.python_version(),
            "implementation": {"path": "src/tensorcode/language/learned_parser.py",
                               "sha256": implementation_sha256, "supporting_sources": implementation_sources},
            "model": {"path": str(args.model), "sha256": digest(args.model)},
            "dataset": {"root": str(root), "split": args.split,
                        "files": [{"name": p.name, "sha256": digest(p)} for p in files],
                        "all_sentences": len(all_rows), "eligible_sentences": len(eligible),
                        "excluded_sent_ids": sorted(excluded_ids),
                        "exclude_reports": [str(p) for p in args.exclude_results],
                        "sampled_sentences": n, "scored_nonpunctuation_tokens": tokens,
                        "selection": "random.Random(seed).sample, original CoNLL-U file/sentence order",
                        "seed": args.seed, "min_tokens_including_punctuation": args.min_tokens,
                        "max_tokens_including_punctuation": args.max_tokens,
                        "bias": "Short sentences only; excludes long constructions; not a full-test estimate."},
            "budgets": {"beam_width": args.beam_width, "max_candidates": args.max_candidates,
                        "max_expansions": args.max_expansions, "ranking": args.ranking},
            "method": "Both decoders receive the same greedy predicted tags. Oracle UAS and LAS independently choose the best candidate per sentence using gold annotations. Missing candidate sets score zero.",
            "limitations": ["Oracle access to gold annotations is unavailable to the agent; this does not measure actual meaning selection.",
                            "No semantic understanding, grounding, execution, or visual inference is evaluated.",
                            "The sample is public and now development-exposed; future untuned confirmation needs a fresh predeclared sample.",
                            "The new search requires complete legal single-root trees; the legacy greedy decoder permits fallback root attachment, so comparison changes validity constraints as well as search."],
            "metrics": metrics,
            "search": {"mean_candidates": statistics.mean(r["candidate_count"] for r in rows),
                       "min_candidates": min(r["candidate_count"] for r in rows),
                       "max_candidates": max(r["candidate_count"] for r in rows),
                       "empty_candidate_sets": sum(not r["candidate_count"] for r in rows),
                       "truncated_sentences": sum(r["truncated"] for r in rows),
                       "exhaustive_sentences": sum(r["complete"] for r in rows),
                       "total_expansions": sum(r["expansions"] for r in rows),
                       "budget_exhaustions": sum(r["reason"] == "budget_exhausted" for r in rows)},
            "latency_ms": latency, "evaluation_seconds": time.perf_counter() - started, "sentences": rows}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, default=Path.home() / ".cache/tensorcode/models/ud_ewt_parser.pickle")
    ap.add_argument("--treebank", type=Path)
    ap.add_argument("--split", choices=("dev", "test"), default="test")
    ap.add_argument("--sample", type=int, default=250)
    ap.add_argument("--seed", type=int, default=20260919)
    ap.add_argument("--min-tokens", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=15)
    ap.add_argument("--beam-width", type=int, default=8)
    ap.add_argument("--max-candidates", type=int, default=4)
    ap.add_argument("--max-expansions", type=int, default=10000)
    ap.add_argument("--ranking", choices=("raw", "local_margin"), default="raw")
    ap.add_argument("--exclude-results", type=Path, action="append", default=[])
    ap.add_argument("--output", type=Path, default=Path("eval/results/parsing_candidates.json"))
    args = ap.parse_args()
    if min(args.sample, args.min_tokens, args.beam_width, args.max_candidates, args.max_expansions) < 1 or args.max_tokens < args.min_tokens:
        ap.error("Counts/budgets must be positive and token range ordered")
    report = evaluate(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "sentences"}, indent=2))


if __name__ == "__main__":
    main()
