# Causal learning retains variable outcomes

This correction follows the objective in [36](36-structured-cognitive-workspace.md):
observations should enter a revisable structured core with their uncertainty intact.
It changes the existing causal learning mechanism; it does not establish generalized
causal reasoning or integrate an experiment loop into the agent.

## The removed assumption

Previously `causal.learn` measured the fraction of paired intervention/control runs
that differed, then chose the **first treated outcome** as the learned effect. If
three treated trials produced red, blue, red against gray controls, the result
asserted red with strength 1.0. That conflated “the intervention changed this aspect
in every measured pair” with “the intervention always produces red.”

The first-outcome rule is removed. There is no compatibility switch.

## Current behavior

`Contrast.pairs` contains comparable treated/control observations. `Contrast.trials`
counts those pairs. `Contrast.outcomes` retains empirical treated outcome counts
over that sample. The original `with_act` and `without_act` sequences retain all
runs, including incomplete and unmatched observations.

`learn` returns a `Causal` record whose `contrast` retains that evidence. When the
comparable treated outcomes vary, `effect` is `Unknown("variable_outcome", ...)`.
Its candidates expose the observed outcomes and empirical shares, explicitly typed
as `vote_share`. No candidate is selected as the answer. When the comparable sample
has a single outcome, `effect` retains the scalar value for that observed regularity;
this is not a universal statement about future trials.

The link's `strength` is also now a `vote_share`: the fraction of comparable pairs
that changed. This is separate from each treated outcome's frequency. It is not a
calibrated probability that an arbitrary future execution will produce that outcome.

`experiment` represents an absent observation using `Unknown("unobserved", aspect)`.
Explicitly observed `None` remains a value. Unknown values and unmatched runs do not
vote in comparisons. No comparable evidence means no learned link.

`tell_causal` records treated observations, control observations, and outcome counts.
For variable outcomes it records `effect_unresolved`, without asserting an `effect`
claim containing a guessed outcome. Causal record identity includes its contrast so
separate empirical samples do not collapse into one record with contradictory trial
counts. Existing evidence source and dependency arguments apply to these records.

## Validation

The causal tests cover variable outcomes, distinct outcome/change frequencies,
constant samples, unmatched runs, missing versus explicitly observed null values,
incomplete pairs, evidence retention in the store, and distinct empirical record
identities. Existing intervention-versus-confound tests continue to pass.

These are deterministic authored experimental fixtures that isolate the mechanism.
They establish correctness of empirical summarization, not learned visual concepts
or inference from arbitrary language.

## Remaining limitations

The caller still supplies the action, observation aspects, control intervention, and
reset procedure. The mechanism trusts that paired runs start from comparable states;
it does not establish that experimental assumption. Probe selection, causal model
generation, explanatory mechanisms, and conditioning on context remain separate work.

The `least_effect` threshold is an explicit caller policy with an existing default,
not a learned decision boundary. Zero-effect and below-threshold contrasts remain
available to the caller but are not returned as learned links. A small unanimous
sample can still be unrepresentative; the retained evidence enables subsequent
assessment, but confidence intervals and future-outcome calibration are absent.

`correlations` remains a separate legacy observational frequency estimator. Its
probability labeling has not been revised by this bounded intervention correction.
Consumers must inspect `support`, `strength.kind`, and `effect` rather than assuming
all causal records are fixed executable predictions.
