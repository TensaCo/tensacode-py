# 4. Example cognitive programs and agents

**Labeling.** Every code block marked **runs** is copied from
`examples/` and runs against the prototype package `tensacode`. None of
it runs against the legacy `tensacode` package, which cannot be imported (see
[01](01-assessment.md)). Blocks marked **conceptual** are not implemented.

Outputs below are excerpts of real runs, stored verbatim in `examples/*/OUTPUT*.txt`.

## 4.0 What each example costs to write

This is the honest line accounting: non-blank lines, docstrings included.

| Example | Main program | Domain types + policy | Bindings/config | Simulated environment | Demo driver |
| --- | ---: | ---: | ---: | ---: | ---: |
| Support router ([`support_router/`](../../examples/support_router)) | `agent.py` **23** | `domain.py` 108 (77 labels, 4 state types, 2 actions, options, 2 constraints, objective) | `config.py` 82 (email parser, keyword rules, learned + model tiers) | `bank.py` 49 | 62 (+ `cases.py` 59, persistence) |
| Knowledge ([`knowledge/`](../../examples/knowledge)) | `program.py` **33** | `domain.py` 63 (types + note parser) | inline | none | 83 |
| Context selection ([`context_select/`](../../examples/context_select)) | `program.py` **31** | reuses knowledge types | inline | none | 46 |
| Recovery ([`recovery/`](../../examples/recovery)) | `agent.py` **28** | `domain.py` 113 (action, outcome enum, limits, steps, classification rules, 3 constraints, objective) | inline | `service.py` 44 | 36 |

The main programs stay in the 15–40 line range. The policy modules are larger, and
that is deliberate: allowed actions, constraints, objectives, and abstention rules
are the substance of an agent, and here they are explicit, testable Python instead
of prompt text.

Remaining friction:
- The `Constraint(name, fn)` wrappers.
- The `@tc.implementation(..., accepts=lambda r: ...)` registration for rule implementations.
- `choose` over three recovery steps with a utility function is heavier than an `if` chain. What it buys is uniform exclusion notes in the trace and constraint enforcement that a model backend cannot bypass.

## 4.1 Event-routing agent: card-support router

*Parse an inbound email, classify intent, retrieve account state, choose an allowed
action, invoke it, verify by observation, persist the case.*

**Main program (runs):** [`support_router/agent.py`](../../examples/support_router/agent.py)

```python
def handle(email: InboundEmail, bank: Bank, cases: CaseLog) -> Case:
    request = tc.parse(email, SupportRequest)
    if isinstance(request, tc.Unknown):
        return cases.needs_human(email, "could not parse message", request)

    intent = tc.classify(request.text, Intent)
    if isinstance(intent, tc.Unknown):
        return cases.needs_human(email, "intent unknown", intent)

    account = bank.find_account(request.sender)
    situation = Situation(intent, request, account, cases.recent_actions(request.sender, hours=24))
    action = tc.choose(options(intent, request, account), objective=RESOLVE_SAFELY, given=situation, constraints=SUPPORT_POLICY)
    if isinstance(action, tc.Unknown):
        return cases.needs_human(email, "no single safe action", action, intent=intent)

    key = f"{email.id}:{type(action).__name__}"
    cases.begin(email, intent, action, key)  # durable intent before any effect
    receipt = tc.invoke(action, executor=bank, key=key)
    verdict = tc.verify(receipt, observe=lambda: bank.observe(action), expect=action.achieved)
    return cases.finish(email, receipt, verdict)
```

**Typed state and policy** (supporting, [`domain.py`](../../examples/support_router/domain.py)):
- **Observations:** `InboundEmail` and the parsed `SupportRequest`.
- **State:** `Account` and `Card` (system of record); `Case` (agent-owned).
- **Goal:** the `RESOLVE_SAFELY` objective ("stop possible fraud first; otherwise send the most specific article").
- **Constraints:** `card_is_active`, `not_repeated_in_24h`.
- **Actions:** `FreezeCard` and `SendArticle`, each declaring effect semantics and a postcondition `achieved(observation)`.
- **Authority:** `AGENT_AUTHORITY` allows exactly these two action types.

**Same program, two bindings** (configuration only, [`config.py`](../../examples/support_router/config.py)):

```python
# deterministic/local: rules -> TF-IDF logistic regression (abstains below a validation-chosen threshold)
tc.Runtime(config.bindings(learned=learned), policy=config.LOCAL_ONLY)
# optional general-model escalation: rules -> learned -> Qwen3-8B running locally (no egress)
tc.Runtime(config.bindings(learned=learned, model=config.local_model_classifier("Qwen/Qwen3-8B")), policy=config.WITH_LOCAL_MODEL)
```

**Happy path (local bindings).** From [`OUTPUT-local.txt`](../../examples/support_router/OUTPUT-local.txt):

```text
### happy path (m-001)
case: status=resolved intent=lost_or_stolen_card action=FreezeCard(card_id='card-a1')
      receipt=applied verification=holds reason='postcondition observed'
handle scenario=happy path message=m-001
  parse    -> SupportRequest(message_id='m-001', sender='alice@example.com', text="…  [0.19 ms total, 0.02 ms backend]
    - email-rules@1 answer
  classify -> Intent.lost_or_stolen_card  [1.07 ms total, 1.04 ms backend]
    - intent-keyword-rules@2 abstain: no_rule_matched
    - tfidf-logreg@1 answer
  choose   -> FreezeCard(card_id='card-a1')  [0.15 ms total, 0.01 ms backend]
    - utility-argmax@1 answer
  invoke   -> Receipt(action=FreezeCard(card_id='card-a1'), status='applied', retry…  [0.02 ms total, 0.00 ms backend]
    · key=m-001:FreezeCard
  verify   -> Verdict(status='holds', reasons=('postcondition observed',), evidence…  [0.01 ms total, 0.00 ms backend]
```

**Ambiguous, unknown, excluded, and failing paths** (same run):

| Scenario | Outcome | Why (from the trace) |
| --- | --- | --- |
| Bob, two active cards: "My wallet was stolen with my cards in it." | `needs_human: no single safe action: tie_within_margin` | Two `FreezeCard` options with equal utility. The chooser abstains instead of freezing an arbitrary card. |
| Same, "…including the card ending in 2222." | `resolved`, `FreezeCard(card-b2)` | The parser extracted `last4`, leaving one option. |
| "Does your company sponsor the city marathon this year?" | `needs_human: intent unknown: below_threshold` | Rules: `no_rule_matched`; learned: `below_threshold`. No guess is made. |
| "the shop says my card isn't accepted there, what gives?" | `needs_human: intent unknown: below_threshold` | Learned top-2 were `card_acceptance` 0.37, `declined_card_payment` 0.30 |
| Carol, card already frozen | `resolved`, `SendArticle(kb/lost-card)` | `· excluded FreezeCard(card_id='card-c1'): card_is_active`; one feasible option, no backend consulted |
| Executor returns 503 | `needs_human`, `receipt=failed verification=fails` | The router does not retry; bounded retry belongs to §4.4 |
| Executor applies, reply lost | `resolved`, `receipt=indeterminate verification=holds` | The observation, not the receipt, decided |
| Blank email | `needs_human: could not parse message: empty_message` | |

At the end of the run, the bank's ground-truth effect list held exactly four effects:
freeze a1, freeze b2, the article to Carol, and freeze e1. The failing, ambiguous,
and unknown cases produced none.

**With the local model** ([`OUTPUT-with-local-model.txt`](../../examples/support_router/OUTPUT-with-local-model.txt)),
two scenarios changed. The program did not.

```text
### learned tier unsure (m-009)
case: status=resolved intent=card_acceptance action=SendArticle(to='alice@example.com', article='kb/where-cards-work')
  classify -> Intent.card_acceptance  [523.80 ms total, 523.75 ms backend]
    - intent-keyword-rules@2 abstain: no_rule_matched
    - tfidf-logreg@1 abstain: below_threshold
    - chat:Qwen3-8B@classify-v1 answer

### unknown intent (m-004)
case: status=needs_human ... reason='intent unknown: model_declined'
  classify -> {'unknown': 'model_declined', ...}  [1019.23 ms total, 1019.19 ms backend]
    - chat:Qwen3-8B@classify-v1 abstain: model_declined
```

One good escalation is an illustration, not evidence. On the full Banking77 test
split, the same escalation tier was right on only **38.2%** of the items it answered
and lowered overall selective accuracy ([05](05-evaluation.md#61-classify-cascade-on-banking77)).
The deployable policy is therefore "rules → learned → human" until a model tier
measures better on this slice.

**Persistence and effect semantics.**
- **Write-ahead.** `cases.begin(...)` writes the case with its idempotency key *before* `invoke`. After a crash, a re-run finds an `in_progress` case and can reuse the same key.
- **The key scopes the effect.** It is `message id + action type`, and the simulated bank honors it (a replay returns the stored receipt without a second effect).
- **Exactly-once is not claimed.** If the real executor ignored keys, the only safe way to finish an `in_progress` case would be to observe first, which §4.4 does. Retrying safely is not the same as the physical effect happening once.
- **Persistence here is illustrative.** `CaseLog` is a JSON file on the reference `Store`. A production host would supply its own persistence, audit, and actuation.

## 4.2 Structural knowledge program

*Two conflicting observations about one printer. Attach evidence and time, detect
the contradiction, query the subgraph, preserve uncertainty. No model adjudicates.*

**Main program (runs):** [`knowledge/program.py`](../../examples/knowledge/program.py)

```python
def observe_poll(world: tc.Store, poll: MonitorPoll) -> tc.Ref:
    source = world.put(tc.Ref(f"obs:poll-{poll.target}-{poll.at:%H%M}"), poll)
    status = Status.online if poll.reachable else Status.offline
    world.tell(tc.Claim(tc.Ref(f"printer:{poll.target}"), "status", status, tc.Interval.at(poll.at)), tc.Evidence(source, poll.at, method=poll.poller))
    return source


def observe_note(world: tc.Store, note_id: str, note: TechNote) -> tc.Ref | tc.Unknown:
    source = world.put(tc.Ref(f"obs:note-{note_id}"), note)
    report = tc.parse(note, StatusReport)  # extraction, not adjudication
    if isinstance(report, tc.Unknown):
        return report  # the note stays on record as an observation with no claim
    claim = tc.Claim(tc.Ref(f"printer:{report.asset_id}"), "status", report.status, tc.Interval(report.since, report.until))
    world.tell(claim, tc.Evidence(source, note.written_at, locator=report.span, method="parse"))
    return source


def status_at(world: tc.Store, printer: tc.Ref, t: datetime) -> Status | tc.Unknown:
    live = world.claims(printer, "status", at=t)
    sources = Counter(rec.claim.object for rec in live for _ in rec.evidence)
    if len(sources) == 1:
        return next(iter(sources))
    if not sources:
        return tc.Unknown("no_evidence", f"no status claim covers {t:%H:%M}")
    total = sum(sources.values())
    return tc.Unknown("contested", f"{len(live)} incompatible claims", tuple((s, tc.Score(n / total, "vote_share")) for s, n in sources.items()))
```

**Output (runs):** [`knowledge/OUTPUT.txt`](../../examples/knowledge/OUTPUT.txt), excerpt

```text
== contradictions (functional predicate, overlapping validity, same scope)
  offline during 10:02-10:02                 source=obs:poll-PRN-3-1002 method=snmp-poller-2 locator=None
  online during 08:00-10:04                  source=obs:note-n-17 method=parse locator=text[15:21]
  overlap                                    10:02-10:02
== queries
status_at(PRN-3, 09:00)                      Status.online
status_at(PRN-3, 10:02)                      Unknown(reason='contested', detail='2 incompatible claims', candidates=((Status.offline, Score(0.5, 'vote_share')), (Status.online, Score(0.5, 'vote_share'))))
status_at(PRN-3, 10:30)                      Unknown(reason='no_evidence', detail='no status claim covers 10:30')
printers on floor 2 with an offline claim @10:02 ['printer:PRN-3']
neighborhood(PRN-3): claims                  [('status', 'offline'), ('status', 'online'), ('located_on', 'floor:hq-2')]
== checks (unknown is not false)
check PRN-3 online @10:02                    unknown
check PRN-3 offline @10:02                   unknown
check PRN-3 online @09:00                    holds
check PRN-4 offline @10:02                   fails
== later evidence about a different time does not resolve the earlier conflict
status_at(PRN-3, 10:06)                      Status.online
status_at(PRN-3, 10:02)                      contested
== the author retracts; the change is proposed, then committed
proposed patch (not yet applied): conflicts  1
after commit r17: conflicts                  0
status_at(PRN-3, 10:02)                      Status.offline
retracted claims kept for audit              1
re-applying the same patch                   StaleRevision: patch based on revision 16, store is at 17
== persistence
records                                      10 entities, 6 claims, 5040 bytes JSON
round trip equal (entities, claims, lossless) (True, True, True)
load with an empty registry                  opaque values: 13; example: Opaque(type_name='Printer', ...)
```

The `vote_share` score counts sources. It is not a probability that either side is
right. The appropriate next step for a contested fact is *more observation* (a fresh
poll), and the demo shows that a poll at 10:06 settles 10:06 but not 10:02. A parser
may be a model; the note parser here is rules. The note "One of the printers upstairs
seems slow" yields `Unknown("no_asset_mentioned")` and is kept as an observation with
no claim.

## 4.3 Context selection

*Retrieve structured facts and documents, rank relevance, remove redundancy, pack
within a budget, never show one side of a contradiction.*

**Main program (runs):** [`context_select/program.py`](../../examples/context_select/program.py)

```python
def select_context(question: str, subject: tc.Ref, world: tc.Store, documents: Sequence[Snippet], *, budget: int) -> tc.Packed[Snippet] | tc.Unknown:
    facts = [as_snippet(rec) for rec in world.neighborhood(subject).claims]
    ranked = tc.rank(question, facts + list(documents))
    if isinstance(ranked, tc.Unknown):
        return ranked

    contested = {rec.id for conflict in world.conflicts(subject) for rec in (conflict.a, conflict.b)}
    required = [f for f in facts if f.claim_id in contested]
    kept, duplicates = tc.dedupe(ranked, similarity=lambda a, b: tc.shingle_similarity(a.text, b.text), threshold=0.5, keep=lambda s: s in required, key=lambda s: s.id)
    return tc.pack(kept, budget=budget, cost=lambda s: tc.approx_tokens(s.text), required=required, key=lambda s: s.id, dropped=duplicates)
```

**Output (runs):** [`context_select/OUTPUT.txt`](../../examples/context_select/OUTPUT.txt)

```text
== budget 100 approx tokens
used 96/100
  + [21] printer:PRN-3 status offline at 10:02 per obs:poll-PRN-3-1002
  + [22] printer:PRN-3 status online 08:00-10:04 per obs:note-n-17
  + [17] Floor 2 users report PRN-3 print jobs stuck in the queue since about 10am.
  + [12] The floor 2 kitchen will be closed for cleaning on Friday.
  + [24] Change 311: network switch sw-hq-2b firmware upgrade scheduled 09:55-10:10 on floor 2.
  - tkt-881-fwd: near-duplicate of 'tkt-881'
  - kb-12: budget: needs 22, 4 left
  - kb-40: budget: needs 18, 4 left

== budget 60 approx tokens
used 60/60
  + [21] printer:PRN-3 status offline at 10:02 per obs:poll-PRN-3-1002
  + [22] printer:PRN-3 status online 08:00-10:04 per obs:note-n-17
  + [17] Floor 2 users report PRN-3 print jobs stuck in the queue since about 10am.
  - tkt-881-fwd: near-duplicate of 'tkt-881'
  - lunch: budget: needs 12, 0 left
  - chg-311: budget: needs 24, 0 left
  ...
== budget 20 approx tokens
Unknown: required_evidence_exceeds_budget (required items cost 43 > budget 20)
```

This output also shows a weakness honestly. BM25 ranks the kitchen memo ("floor 2")
above the knowledge-base article on dropped network printers, and at 60 tokens the
change notice that likely explains the outage is dropped. `rank` is a contract, so a
better ranker (embeddings, a cross-encoder, a model) can be bound without touching
this program. [05](05-evaluation.md#62-context-selection-on-hotpotqa) measures the
lexical ranker on 7,405 HotpotQA questions.

## 4.4 Recovery agent

*Detect a failed action, distinguish retryable from terminal, choose a bounded
recovery step, escalate when evidence or budget is insufficient.*

**Main program (runs):** [`recovery/agent.py`](../../examples/recovery/agent.py)

```python
def credit_with_recovery(ep: Episode, *, executor: tc.actions.Executor, observe: Callable[[], object], key: str, sleep: Callable[[float], None]) -> Resolution:
    receipt = tc.invoke(ep.action, executor=executor, key=key)
    ep.invocations += 1
    for _ in range(ep.limits.max_invocations + ep.limits.max_observations):  # hard stop even if a constraint is wrong
        verdict = tc.verify(receipt, observe=observe, expect=ep.action.achieved)
        if verdict.holds:
            return Resolution("done", "; ".join(verdict.reasons), ep.invocations, ep.observations, ep.elapsed_s)
        ep.outcome = tc.classify(AttemptReport(receipt, verdict), Outcome)
        step = tc.choose(next_steps(ep), objective=RECOVER_SAFELY, given=ep, constraints=RECOVERY_POLICY)
        if isinstance(step, (Escalate, tc.Unknown)):
            return Resolution("escalated", step.reason, ep.invocations, ep.observations, ep.elapsed_s)
        sleep(step.after_s)
        ep.elapsed_s += step.after_s
        if isinstance(step, Retry):
            receipt = tc.invoke(ep.action, executor=executor, key=key)
            ep.invocations += 1
        else:
            ep.observations += 1
    return Resolution("escalated", "step limit reached", ep.invocations, ep.observations, ep.elapsed_s)
```

**Supporting policy** ([`recovery/domain.py`](../../examples/recovery/domain.py)):
- **The action.** `CreditAccount` is deliberately **not idempotent**, so a duplicate means money moved twice.
- **Classification (rules).** `classify` maps a receipt and its verdict to `transient | terminal | effect_unknown | effect_missing`.
- **Steps.** `Retry(backoff)`, `Reobserve(1s)`, and `Escalate(reason)`.
- **Constraints:**
  - `retry_is_safe`: allowed when transient; when the effect is missing *and* the read is authoritative; or when the action is idempotent or the executor honors keys.
  - `reobserve_is_useful`
  - `within_limits`: invocations, observations, deadline.
- **Facts about the target system, stated explicitly:** `Limits.executor_honors_keys` and `Limits.observation_is_authoritative`. The first version of this example lacked the second flag and escalated a case it could safely retry, so the flag was added.

**Outputs (runs):** [`recovery/OUTPUT.txt`](../../examples/recovery/OUTPUT.txt)

| Scenario | Result | Invocations / observations | Ground-truth effects |
| --- | --- | --- | --- |
| transient 503, then success | done | 2 / 0 | 1 |
| terminal 403 | escalated (`terminal`) | 1 / 0 | 0 |
| reply lost after commit; ledger query works | done (the observation shows the credit) | 1 / 0 | 1 |
| reply lost before commit; replica read may lag | escalated (`effect_missing`) after 2 re-observations | 1 / 2 | 0 |
| reply lost before commit; authoritative read | done (safe retry) | 2 / 0 | 1 |
| reply lost; ledger query down; keys **not** honored | escalated (`effect_unknown`), **no retry** | 1 / 2 | 1 |
| reply lost; query down then up; keys honored | done after 2 re-observations | 1 / 2 | 1 |
| persistent 503 | escalated after 3 invocations (backoff 2 s, 4 s) | 3 / 0 | 0 |

```text
### reply lost; query down; keys NOT honored
result: escalated (effect_unknown after 1 invocation(s)); invocations=1 observations=2 simulated_s=2.0
ground truth: effects applied = 1
  invoke   -> Receipt(action=CreditAccount(...), status='indeterminate' ...)
  verify   -> Verdict(status='unknown', reasons=('observation unavailable: ledger_query_unavailable',) ...)
  classify -> Outcome.effect_unknown
  choose   -> Reobserve(after_s=1.0)
    · excluded Retry(after_s=2.0): retry_is_safe
  ... (second re-observation, still unavailable) ...
  choose   -> Escalate(reason='effect_unknown after 1 invocation(s)')
    · excluded Retry(after_s=2.0): retry_is_safe
    · excluded Reobserve(after_s=1.0): within_limits
    · only one feasible option; no backend consulted
```

The escalation carries its evidence: one indeterminate receipt, two failed
observations, and the constraint that blocked a retry. A human, or an
effect-reconciliation process, then works from facts rather than from a guess. Trajectory
metrics over 5,000 simulated episodes against three baseline policies are in
[05](05-evaluation.md#63-recovery-trajectories).

## 4.5 What the examples changed in the API

Each change below was made because an example or measurement exposed a problem. None
was cosmetic.

| Friction found | Change |
| --- | --- |
| `Unknown.reason` said `exhausted`, burying `tie_within_margin` in the detail | The final reason is now the last real attempt's reason |
| Every in-process rule printed `usd=None (unknown)` | In-process implementations *declare* zero metered spend with a source. Energy remains unmeasured. |
| `invoke` and `verify` spans had no timing | `Span.close()` |
| Context packing lost duplicate drops from the report | `pack(dropped=...)`; `dedupe(key=...)` |
| The recovery agent escalated a safely retryable case | Explicit `observation_is_authoritative` fact |
| The filter mapping referred to a batch `check` that did not exist | `check.many` |
| Shared value aliasing across two records went unreported | The aliasing map is shared across a conversion |
| Store commits copied every entity (found by the benchmark) | Stage only touched entities |
