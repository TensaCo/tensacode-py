"""Bindings and policy for the support router. Deployment configuration, kept out of the program.

The program in ``agent.py`` is identical under every configuration below.
"""

from __future__ import annotations

import csv
import random
import re
from pathlib import Path

import tensorcode as tc
from tensorcode.backends.builtin import IN_PROCESS, KeywordClassifier, UtilityChooser

from .domain import Intent, InboundEmail, SupportRequest

# -- parse: deterministic email parsing (rules)

_QUOTED = re.compile(r"(^>.*$)|(^On .* wrote:$)", re.M)
_LAST4 = re.compile(r"\b(?:ending(?: in)?|last (?:four|4)(?: digits)?(?: of)?|card)\s*(?:#|no\.?|number)?\s*(\d{4})\b", re.I)


@tc.implementation(
    "parse",
    name="email-rules",
    version="1",
    accepts=lambda r: isinstance(r.subject, InboundEmail) and r.target is SupportRequest,
    profile=IN_PROCESS,
)
def parse_email(request: tc.Request) -> SupportRequest | tc.Unknown:
    email: InboundEmail = request.subject
    body = _QUOTED.sub("", email.body).split("\n-- \n")[0].strip()
    text = " ".join(part for part in (email.subject.strip(), body) if part)
    if not text:
        return tc.Unknown("empty_message")
    last4 = _LAST4.search(text)
    return SupportRequest(email.id, email.sender.lower(), text, last4.group(1) if last4 else None)


# -- classify tier 1: keyword rules, written from the *training* split only.
#    Revised once after reviewing errors on the training holdout (see eval/README); frozen before test.

KEYWORD_RULES = KeywordClassifier(
    Intent,
    {
        Intent.pin_blocked: [r"\bpin\b.*\b(blocked|locked|unblock|too many|tries|attempts)\b", r"\b(blocked|locked)\b.*\bpin\b"],
        Intent.lost_or_stolen_phone: [r"\bphone\b.*\b(lost|stolen|stole|misplaced)\b", r"\b(lost|stolen|misplaced)\b.*\bphone\b"],
        Intent.terminate_account: [r"\b(close|delete|terminate|cancel|discontinue)\b.*\baccount\b"],
        Intent.card_swallowed: [r"\b(atm|machine)\b.*\b(swallow\w*|ate|eaten|kept|retain\w*)\b.*\bcard\b", r"\b(atm|machine)\b.*\b(swallow\w*|ate|eaten)\b"],
        Intent.apple_pay_or_google_pay: [r"\b(apple|google|android) ?pay\b"],
        Intent.visa_or_mastercard: [r"^(?!.*\batms?\b).*\b(visa|master ?card)\b"],
        Intent.passcode_forgotten: [r"\b(forg[eo]t\w*|reset|remember|recover)\b.*\bpass ?code\b", r"\bpass ?code\b.*\b(forg[eo]t\w*|reset)\b"],
        Intent.age_limit: [r"\bhow old\b", r"\bminimum age\b", r"\bage (limit|requirement)"],
    },
    name="intent-keyword-rules",
    version="2",
)


def load_banking77(path: Path) -> list[tuple[str, Intent]]:
    with path.open(newline="") as f:
        return [(row["text"], Intent(row["category"])) for row in csv.DictReader(f)]


def split_train_validation(rows: list[tuple[str, Intent]], *, fraction: float = 0.1, seed: int = 0):
    """Stratified holdout carved from the training file; the test file is never touched here."""
    rng = random.Random(seed)
    by_label: dict[Intent, list[tuple[str, Intent]]] = {}
    for row in rows:
        by_label.setdefault(row[1], []).append(row)
    train, val = [], []
    for label_rows in by_label.values():
        rng.shuffle(label_rows)
        k = max(1, round(len(label_rows) * fraction))
        val += label_rows[:k]
        train += label_rows[k:]
    return train, val


def learned_classifier(train_csv: Path, *, target_accuracy: float = 0.95):
    from tensorcode.backends.linear import LinearTextClassifier

    train, val = split_train_validation(load_banking77(train_csv))
    return LinearTextClassifier.fit(
        [t for t, _ in train],
        [y for _, y in train],
        label_type=Intent,
        validation=([t for t, _ in val], [y for _, y in val]),
        target_accuracy=target_accuracy,
        basis=f"banking77/train-holdout-10%-seed0; target selective accuracy {target_accuracy}",
    )


def local_model_classifier(model_id: str):
    from tensorcode.backends.hf_local import ChatClassifier, LocalChatModel

    # Qwen3 models think by default; classification here wants a bare label.
    kwargs = {"enable_thinking": False} if "Qwen3" in model_id else {}
    return ChatClassifier(LocalChatModel(model_id, template_kwargs=kwargs), Intent)


def bindings(*, learned=None, model=None) -> list:
    """Declared cascade order: rules, then learned, then (optionally) a general model."""
    chain = [parse_email, KEYWORD_RULES]
    chain += [learned] if learned is not None else []
    chain += [model] if model is not None else []
    return chain + [UtilityChooser(margin=0.0)]


LOCAL_ONLY = tc.Policy(localities=frozenset({"in_process"}), available=frozenset({"sklearn"}), max_attempts=3)
WITH_LOCAL_MODEL = tc.Policy(localities=frozenset({"in_process"}), available=frozenset({"sklearn", "cuda"}), max_attempts=3)
