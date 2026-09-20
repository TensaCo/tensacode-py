# Whole-sentence goal boundary review

Verdict: changes requested for three callback/publication guard gaps. This review
is limited to the partial Task 4 goal boundary described in
`task-4-goals-report.md`, its supplied patch, and the root-owned changes in
`goal_interpretation.py`, `goal_learning.py`, and
`test_sentence_goal_teaching.py`. The four pending routing tests, other workers'
changes, and complete conversation routing are outside this verdict.

## Findings

### 1. P2: selection does not authenticate sentence content after selection/extraction callbacks

Location: `src/tensorcode/agent/goal_interpretation.py:415-420`, following the new
sentence-specific checks at lines 323-340.

The sentence snapshot is checked before `workspace.select`, dependency capture,
and extraction of the selected goal. The final check only calls
`validate_dependencies`, which compares identities, revision, selection, and
frontier; it does not compare the retained sentence's payload or source content.
Consequently a callback during selection can replace the parent reading with a
one-clause reading while preserving its comparison basis, and the function
returns a concrete goal with the old two-clause constraints and a live-looking
dependency. Both the new taught and learned sentence paths reproduce this.
The existing new tests expressly protect against this same content mutation
before selection, but do not cover it during selection.

Reproduction (repository root, `.venv/bin/python`):

```python
import runpy
from dataclasses import replace
from tensorcode.outcomes import Unknown
h = runpy.run_path('tests/test_sentence_goal_teaching.py')
for mode in ('taught', 'runtime'):
    agent = h['Agent']() if mode == 'taught' else h['trained_agent']()
    row = h['example'](agent, 'late-mutation-' + mode)
    gid = h['teach'](agent, row) if mode == 'taught' else h['runtime_proposal'](agent, row)
    original = agent.interpretations.select
    def selecting(*args, **kwargs):
        result = original(*args, **kwargs)
        if args[0] == gid:
            parent = agent.interpretations._groups[row[3].id]
            changed = replace(parent.candidates[0], payload=replace(
                row[4].payload, acts=row[4].payload.acts[:1]))
            agent.interpretations._groups[parent.id] = replace(parent, candidates=(changed,))
        return result
    agent.interpretations.select = selecting
    result = h['select'](agent, gid)
    assert not isinstance(result.goal, Unknown)  # Unexpected success
    assert result.dependency is not None
```

Revalidate the sentence-specific snapshot after the late callback-bearing reads,
and finish with scalar comparisons for all participating commitments. Add a
regression for both sentence modes. This is an extension gap in the existing
selection tail, rather than a claim that the old scalar validator changed.

### 2. P2: taught retention can publish success after its final evidence read withdraws the parent

Location: `src/tensorcode/agent/goal_interpretation.py:182-188`.

After `_validate_taught_parent`, both `workspace.get_source` and `workspace.get`
can invoke copy callbacks. Only the newly created goal group's comparison is
checked afterward. A withdrawal of the sentence during those reads is therefore
missed, and the stale teaching is registered and its group ID returned as a
successful retention. The later selector rejects it, so this finding concerns
the promised retention boundary, not a demonstrated subsequent action.

Reproduction:

```python
import runpy
from tensorcode.outcomes import Unknown
h = runpy.run_path('tests/test_sentence_goal_teaching.py')
agent = h['Agent']()
row = h['example'](agent, 'late-withdrawal')
original = agent.interpretations.get_source
armed = True
def reading_source(source_id):
    global armed
    result = original(source_id)
    if armed and result.provider == 'explicit-goal-teaching':
        armed = False
        agent.interpretations.unset(row[3].id, reason='withdrawn during final teaching evidence read')
    return result
agent.interpretations.get_source = reading_source
gid = h['teach'](agent, row)
assert not isinstance(gid, Unknown)  # Unexpected success
assert agent.interpretations.get(row[3].id).selected_id is None
assert gid in agent._goal_proposal_groups
```

Perform the parent/support check after these final evidence copies, and finish
with comparison checks for both the goal group and dependencies. This shared
taught-helper gap also existed for single-frame teaching; the new sentence API
inherits it. Cover the new API and preserve single-frame behavior.

### 3. P2: learned sentence publication does not check the published goal group's state

Location: `src/tensorcode/agent/goal_interpretation.py:254-265`.

The new learned sentence path checks its reading dependencies after publication,
but never checks that the newly created goal group still has exactly the authored
proposal set, unselected state, and expected initial comparison. A `propose`
callback can select the new goal group; retention then registers and returns it
successfully as an already-selected group. This violates the stated proposal-only
boundary and differs from the taught helper's explicit initial-comparison check.

Reproduction:

```python
import runpy
from tensorcode.outcomes import Unknown
h = runpy.run_path('tests/test_sentence_goal_teaching.py')
agent = h['trained_agent']()
row = h['example'](agent, 'publication-selection')
original = agent.interpretations.propose
def publish(*args, **kwargs):
    candidate = original(*args, **kwargs)
    agent.interpretations.select(candidate.group_id, candidate.id, reason='callback selection')
    return candidate
agent.interpretations.propose = publish
gid = h['runtime_proposal'](agent, row)
assert not isinstance(gid, Unknown)  # Unexpected success
assert agent.interpretations.get(gid).selected_id is not None
```

Authenticate the expected source, candidates, and initial group comparison before
successful registration/return, with final scalar checks after dependency
validation. Extend the publication test to include selection or proposal-set
changes, not only reading withdrawal.

## Scope compliance and positive observations

- The request-sequence envelope retains every ordered frame and the supplied
  declarative goal, including invariants. It does not invent clause composition.
- `_frames` rejects skipped/unresolved content and mismatched Request/Frame acts;
  token anchors are checked against the original sentence when present.
- Learned runtime projection uses an admitted goal model without a lexical
  fallback. Both reading support and model dependencies are retained.
- The `goal_learning._task_state` change reuses sentence authentication when
  extracting retained supervision. No separate defect was found in that small
  change.
- The tests accurately describe their semantics and grounding as supplied
  fixtures. They establish the intended whole-input structural correspondence,
  not understanding learned from raw input or completed routing.
- The supplied source-text parameter on taught retention may differ from the
  parent source text. I verified that mismatch is accepted; I have not classified
  it as a defect because the existing explicit-teaching API already permits a
  supplied text label, and the authentic parent source remains in the snapshot.
  If this field is intended to be an exact sentence copy, specify and test that
  contract explicitly.

## Verification

Read the bounded report and patch first, then the relevant implementations,
dependency/snapshot helpers, and Task 4 requirements. Ran two narrow Python
diagnostics using the new test helpers; all three findings reproduced as shown.
Did not rerun the full suite, the intentionally RED routing tests, or review/edit
other workers' implementation. The reported 67-test pass is supplied evidence,
not independently rerun in this review. No source files were modified.
