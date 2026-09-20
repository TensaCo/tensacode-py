# TensorCode

**Typed cognitive operations with swappable implementations.** Your program says *what*
it needs (parse this, classify that, choose an action under these constraints, check this
claim). A policy decides *which* implementation answers: rules, a learned model, or a
general model. Every answer is validated against the operation's type, every abstention
is an explicit value, and every attempt is traced.

> **Status: pre-alpha (0.1.0a1).** Nothing is stable before 1.0: any 0.x release may change
> or remove any part of the API, and [CHANGELOG.md](https://github.com/TensaCo/tensacode-py/blob/main/CHANGELOG.md)
> lists what changed. The core has no third-party dependencies. Nothing here calls a
> language model unless you bind one.

```bash
pip install --pre tensorcode
```

Python 3.11+.

## Agent development direction

The agent is being developed toward a [structured cognitive workspace](docs/revival/36-structured-cognitive-workspace.md):
language, images, and observations become revisable interpretations with evidence; reasoning,
planning, hypothesis formation, and learning operate on that structure; communicative and
action intentions produce output and new observations. This is the objective, not a claim
that all of these capabilities are implemented. [The current planning milestone](docs/revival/35-model-based-planning.md)
executes supplied models and some narrow language requests; broad interpretation and integrated
model learning remain unfinished. Language turns now require an explicit interpretation
selection policy to dispatch candidate acts; the default retains and defers them. Image
providers return scene proposals, with no legacy direct-claim fallback. See the
[scene checkpoint](docs/revival/38-scene-interpretations.md) and
[semantic-authority cleanup](docs/revival/39-removing-implicit-semantic-authority.md) for the
breaking API changes. Automatic post-selection resolution and bundled project/request
knowledge are removed; explicit selection alone does not supply missing task semantics.

Vision here means holistic scene understanding: layout, grouping, relational structure,
events, affordances, and competing explanations of the whole situation. Object labels and
regions are supporting evidence. The target uses extensible relational representations and
shared grounding with language; it is not limited to UI elements or classification. Learned
scene formation, temporal dynamics, and active visual investigation remain open work.
[Relational scene grounding](docs/revival/63-learning-relational-scene-grounding.md)
now develops learned graph-query proposals from supplied scenes and labeled
referents, with explicit admission and competing bindings. This does not yet learn
scene structure from pixels or infer the meaning of unfamiliar descriptions.

## Quick start

```python
import enum
import tensorcode as tc
from tensorcode.backends.builtin import KeywordClassifier

class Intent(enum.Enum):
    LOST_CARD = "lost_card"
    REFUND = "refund"

print(tc.classify("I lost my card", Intent))
# Unknown(reason='no_implementation', ...)   <- nothing bound: an explicit Unknown, not a guess

rules = KeywordClassifier(Intent, {
    Intent.LOST_CARD: [r"\blost\b", r"\bstolen\b"],
    Intent.REFUND: [r"\brefund\b"],
})

@tc.implementation("classify", name="fallback", version="1")
def ask_a_human(request):
    return tc.Unknown("needs_human", request.subject)

with tc.use(tc.Runtime([rules, ask_a_human])) as rt:
    print(tc.classify("I lost my card", Intent))       # Intent.LOST_CARD
    print(tc.classify("where is my parcel?", Intent))  # Unknown(reason='needs_human', ...)
    print(rt.trace.render())
```

```text
classify -> Intent.LOST_CARD  [0.04 ms total, 0.01 ms backend]
  - keyword-rules@1 answer
classify -> {'unknown': 'needs_human', ...}  [0.03 ms total, 0.01 ms backend]
  - keyword-rules@1 abstain: no_rule_matched
  - fallback@1 abstain: needs_human, usd=? (unknown)
```

The program never names a backend. Swap the rules for a scikit-learn classifier or a local
model by changing the `Runtime`, not the program.

## Core ideas

**Operations fix the meaning.** Each facade validates its output and fails closed:

| Family | Operations |
| --- | --- |
| infer | `parse`, `classify`, `choose`, `rank` |
| check | `check`, `verify` |
| act | `invoke` (returns a `Receipt`), `Plan`, `run_plan` |
| context | `pack`, `dedupe` |

`classify` estimates what *is true*. `choose` selects what *to do*, under an `Objective`
and hard `Constraint`s that TensorCode checks itself before any backend sees the options.

**Outcome values keep distinctions a caller must not collapse.**
- `Unknown` is not `False` and not a low-confidence guess. It raises if used as a boolean.
- `Verdict` has three states. `fails` and `unknown` are different.
- `Receipt` separates "did not happen" from "may have happened".
- `Score` says what kind of number it is. A similarity is not a probability, and a
  probability must name the data it was calibrated on.

**The runtime binds implementations by policy.** An implementation declares `Traits`
(locality, egress, determinism, requirements) and a measured `Profile`. `Policy` filters
on hard constraints and orders a cascade; `Budget` caps cost and attempts across calls.
Unmeasured means unknown: a missing cost is never counted as zero, and a cost cap excludes
implementations whose cost is unknown.

**Records carry evidence.** `Store` holds entity and claim records with evidence, validity
intervals and scope, and detects conflicts between them.

## Optional extras

| Extra | Installs | For |
| --- | --- | --- |
| `learned` | scikit-learn, numpy | `tensorcode.backends.linear` |
| `local-model` | torch, transformers | `tensorcode.backends.hf_local` |
| `learned-neural` | torch, transformers | `tensorcode.backends.neural` |

## Examples

The examples live in this repository, not in the package. Run them from a checkout:

| Example | Command |
| --- | --- |
| Recovery after failed or ambiguous actions | `python -m examples.recovery.demo` |
| Knowledge store: ingest, conflicts, queries | `python -m examples.knowledge.demo` |
| Packing context into a token budget | `python -m examples.context_select.demo` |
| Support router (needs the Banking77 train CSV) | `python -m examples.support_router.demo --train banking77_train.csv` |
| Decision service with HTTP API and operator UI | `python -m examples.decisions.service` |
| **General agent**: chat with it while it works a simulated desktop (needs the `computerworld` wheel, WordNet, VerbNet) | `python -m examples.general_agent.server` |
| Live browser agents, no model calls (needs `playwright`, `numpy`, `pillow`) | `python -m examples.browser_agents.live` |

Expected output is checked in beside each demo (`OUTPUT*.txt`).

## What has been measured, honestly

The design notes in [`docs/revival/`](https://github.com/TensaCo/tensacode-py/blob/main/docs/revival/README.md) report measurements,
including the unflattering ones:

- A local zero-shot Qwen3-8B escalation tier **lowered** Banking77 selective accuracy from
  94.1% to 91.0%. Tiers should be admitted only by measured quality.
- Recovery logic based on explicit facts made 0 duplicate money movements in 5,000
  simulated episodes, compared with 4.8% for naive retry. The simulator and its fault model
  are the project's own, and with wrong facts about the target system duplicates return (0.9%).
- The [evidence audit](https://github.com/TensaCo/tensacode-py/blob/main/docs/revival/11-evidence-audit.md) found that 13 of the project's 19
  headline results ran in environments it wrote **and** were graded by code it wrote. On
  public open-domain benchmarks, the cascade did worse than plainly prompting the same
  model on 3 of 4.

Treat the agent results as demonstrations, not benchmarks.

## Repository layout

```text
src/tensorcode/  the package
tests/           pytest suite (some tests use examples/ and eval/)
examples/        runnable programs and agents
eval/            evaluation scripts and result files (eval/results/*.json)
research/        experiments, including a civilization simulation
docs/revival/    design notes and measurements
```

## Development

```bash
pip install -e ".[dev,learned]"
python -m pytest -q
```

Evaluation scripts read downloaded datasets and trained artifacts from `$TENSORCODE_SCRATCH`
(default `~/.cache/tensorcode`). Tests that need those artifacts skip when they are absent.

The legacy 2023–2024 package (`tensacode`, with `Engine` and TCIR; never published) is
preserved at the git tag `legacy-2024-11`. It was never functional and is not compatible
with this one. The PyPI name `tensacode` is a placeholder that installs `tensorcode`.

## License

[MIT](https://github.com/TensaCo/tensacode-py/blob/main/LICENSE)
