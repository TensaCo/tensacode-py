"""Does any expectation in these files encode a defect the code has since fixed?

Twice now a stale expectation has outlived the bug it was written for: the stemmer
gave `arriv` for "arrived", the benchmark asserted `arriv`, and when the stemmer was
fixed the *test* was what looked broken. A benchmark written beside the system inherits
its defects as expectations, and nothing about reading the file makes that visible.

So this is a test rather than a note. For every word that appears in an utterance in
the files below, it asks what the current code makes of that word, and flags any short
literal in the same file that is the word with a suffix chopped off but is *not* one of
the stems the code now produces. It reads the real tables, so it stays honest as they
change, and it found one defect on its first run: `inflect` turned "say" into "saies",
so nothing round-tripped to "says" and the stemmer refused the word.
"""

import ast
import pathlib

from tensorcode.language.grammar import GUESS_SUFFIXES, OpenClass, guess_entries

ROOT = pathlib.Path(__file__).resolve().parent.parent

FILES = ["tests/test_language_grammar.py", "tests/test_language_english.py",
         "tests/test_language_desktop.py", "tests/test_language_open_vocabulary.py",
         "tests/test_learning_induction.py", "eval/language_benchmark.py"]

SPECS = [OpenClass(r".*", cat, sem="word", morphology=True) for cat in ("V", "N", "Adj")]


def _stems_of(word: str) -> set[str]:
    return {entry.sem for spec in SPECS for entry in guess_entries(word, spec)
            if isinstance(entry.sem, str)} | {word}


def _strings(path: pathlib.Path):
    """Every string literal except docstrings, which are prose about the code."""
    tree = ast.parse(path.read_text())
    docs = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            first = (node.body or [None])[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) \
                    and isinstance(first.value.value, str):
                docs.add(id(first.value))
    return [(n.lineno, n.value) for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docs]


def _stale(path: pathlib.Path) -> list[str]:
    lits = _strings(path)
    words = {w.strip(".,?!'\"").lower() for _, v in lits
             if " " in v and "\n" not in v and len(v) <= 90
             for w in v.split() if w.strip(".,?!'\"").isalpha() and len(w.strip(".,?!'\"")) > 3}
    short = [(line, v.lower()) for line, v in lits if v.isalpha() and 1 < len(v) < 12]
    out = []
    for word in sorted(words):
        stems = _stems_of(word)
        # what chopping a suffix off this word would leave, before any repair
        chopped = {word[: -len(suffix)] for suffix, _, _, _ in GUESS_SUFFIXES
                   if suffix and word.endswith(suffix) and len(word) > len(suffix) + 1}
        for line, lit in short:
            if lit in chopped and lit not in stems:
                out.append(f"{path.name}:{line}: {lit!r} is {word!r} with a suffix chopped off; "
                           f"the code reads it as {sorted(stems - {word})}")
    return sorted(set(out))


def test_no_expectation_encodes_a_stem_the_code_no_longer_produces():
    stale = [note for name in FILES for note in _stale(ROOT / name)]
    assert not stale, "stale expectations:\n  " + "\n  ".join(stale)


def test_the_check_catches_the_expectation_it_was_written_for(tmp_path):
    """A guard that cannot fail is not a guard: hand it the expectation we really had."""
    stale = tmp_path / "stale_case.py"
    stale.write_text('CASES = [("the grain has arrived", "arriv", {"aspect": "perfect"})]\n')
    found = _stale(stale)
    assert found and "'arriv'" in found[0] and "arrive" in found[0]

    fixed = tmp_path / "fixed_case.py"
    fixed.write_text('CASES = [("the grain has arrived", "arrive", {"aspect": "perfect"})]\n')
    assert _stale(fixed) == []
