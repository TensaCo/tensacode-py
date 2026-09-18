# Affect, perceptual axes, and selfhood for simulated agents

Two layers, kept apart:

1. **What the sources claim.** A faithful summary of three pages of *The Shape of Experience* (theshapeofexperience.org). I read them through a fetch tool that returns extracted text. Short quotes are verbatim from that extraction and should be re-checked against the pages before publication.
2. **How we operationalize it.** My design for simulated agents. This is interpretation and engineering, not the author's claims.

---

## 1. What the sources claim

### 1.1 The geometry of affect ([part 2](https://theshapeofexperience.org/part-2/the-geometry-of-affect#affects-as-structural-motifs))

- **Affects are relational structures, not scalar feelings.** An affect is characterized by how it relates to other possible affects: "know how an affect relates to every other possible state" and you know the affect.
- **The page names recurring structural measures ("five plus one"):**
  - **Valence:** alignment of the trajectory with the gradient of a *viability manifold*. Positive when moving toward viable states, negative toward dissolution. Intensity is the steepness of that landscape.
  - **Arousal:** the rate of belief/state update (e.g. divergence between successive belief states).
  - **Integration (Φ):** irreducibility of cause-effect structure (information lost under the best partition).
  - **Effective rank:** how many degrees of freedom are active (from the eigenvalues of the state covariance). High means expansive, low means concentrated or trapped.
  - **Counterfactual weight:** the share of processing spent on non-actual trajectories (planning, anticipation).
  - **Self-model salience**, split in two:
    - *attentional*: how prominently the self is an object of processing;
    - *causal*: how much of action is explained by the self-model.

    These dissociate. The page's examples are flow (low attentional, high causal) and shame (high on both), and depersonalization is described as low on both.
- **Self-salience is one diagonal of a larger field.** It is "the diagonal of an entity-indexed field": the same salience field ranges over known others, rivals, children and institutions.
- **The space is curved and asymmetric.** Transitions don't cost the same both ways. The examples: fear slides into anger more easily than back, and grief does not run back into the coupling that preceded the loss.
- **The dimensions are a toolkit.** They are "not claimed necessary, sufficient, or exhaustive". Coordinate charts are local linearizations of a structure whose full metric is described as an open problem.
- **Named profiles are sparse.** In the section I read, profiles are given only for flow, shame, depersonalization, joy (high effective rank, positive valence), suffering (low effective rank, negative valence), fear (counterfactual weight toward threat) and desire (counterfactual weight toward opportunity). Anger, grief, boredom, awe, love, disgust and curiosity get **no** dimensional profile there. Any profiles we use for them are ours.

### 1.2 The perceptual axes: ascription, coupling, gain ([part 2](https://theshapeofexperience.org/part-2/the-perceptual-axes-ascription-coupling-gain))

**Ascription α(x)** is a field over *targets*, with two components in [0, 1]:
- **αA(x), agency or teleology:** how much x is modeled with an agent template rather than as stripped dynamics.
- **αP(x), phenomenality or patiency:** how much the perceiver treats x as having something it is like to be x.

The two dissociate. A corporation can be agentive but not experiential; a sedated patient experiential but not agentive. Prediction blends the two models: W(x) = αA·W_agent + (1 − αA)·W_mech.

- **Animism is a computational default:** reusing one's own agent model is cheaper than learning a new model.
- **Dehumanization** is primarily a local collapse of αP(target), and often of αA too, which flattens the target into an obstacle.

**Coupling κ ∈ [0, 1]** is how much the perceiver's own streams (perception, affect, agency attribution, narrative) constrain one another rather than factorize.
- High κ covers flow, contemplative and aesthetic states. Low κ covers depression, dissociation, hyper-rationalization and compartmentalization.
- Low κ reduces integration: κ is described as a dial on the amount of experience, not just its shape.
- Coupled modes "propagate alarm/attraction; factorized ones contain it".

**Gain γ** is precision weighting: how much bottom-up signal overrides top-down priors.
- High γ is vivid, flexible and potentially destabilized (sensory flooding, psychedelic states).
- Low γ is prior-dominated: stable, rigid, prone to rumination and prior-locked perception.

**The covariation conjecture:** α, κ and γ may covary in biological systems under shared pressures. The author presents this as an empirical claim, not a definition.

**Calibration:** lowering αA is a valuable learned skill (mechanism-sensitive prediction, engineering, medicine). High ascription is not automatically better.

### 1.3 Responses to inescapable selfhood ([part 3](https://theshapeofexperience.org/part-3/the-expression-of-inevitability-human-responses-to-inescapable-selfhood))

- **Existential burden:** being a self-modeling system carries a chronic cost of self-reference that can't be escaped without dissolving the self-model. Cultural forms are strategies for inhabiting stable regions of affect space.
- **Four response families:**
  - **terror management**, through symbolic immortality and worldview defense;
  - **meaning maintenance**, restoring coherence after meaning violations;
  - **attachment**, co-regulating with bonded others, a "stable basin at viable position";
  - **flow**, where challenge-skill balance absorbs the self-model.
- **The aim is a basin, not the absence of self-reference:** a deep, stable basin where what matters holds with enough dynamical stability. Stability is part of well-being.
- **Pathological configurations:**
  - **melancholic depression:** collapsed gradient, low coupling; the world is visible but unfelt;
  - **expansive despair:** decoupled modes, where options generate no force;
  - **anxiety:** a flickering landscape with high gain and no anchors;
  - **addiction:** a circular attractor, high intensity with no traversal;
  - **degenerate evaluation:** oscillating where justification gives out.

  Each needs a different structural repair (reconnection, stabilization, topology expansion, frame separation).
- **Culture as affect infrastructure:** liturgy, architecture, pedagogy, law and market design shape which regions of affect space a population can reach.

---

## 2. Operationalization (our design)

The design principle, taken from the sources: **affect is a structure computed from how the agent's cognition is running, not a mood variable set by rules.** Rules produce *appraisals* (claims with provenance). The affect measures are *read off* the agent's own processing each tick. A motif is a pattern over those readings plus its recent trajectory.

### 2.1 Per-agent state

| Component | Focal agent (tensacode Store) | Background agent (NumPy struct) |
|---|---|---|
| Viability variables v (nutrition, hydration, injury, warmth, safety, standing, attachment security, coherence) with soft boundaries | claims `(me, viability, (name, value))` in scope `self`, plus the boundary table (shared per species or culture) | `needs` f16×6 |
| Affect readings: valence, arousal, Φ proxy, effective rank, counterfactual weight, σ_att, σ_causal | a small dataclass entity `affect:me` (7 floats), with the last 32 readings kept as a ring buffer | `affect` f16×7 |
| Motif label and its support | claim `(me, motif, "grief")`, `derived_from` the appraisals that produced it | quantized motif id in `life` bits (optional) |
| Perceptual axes | defaults (αA₀, αP₀, κ, γ) plus a **sparse per-target override field** α(x) as claims `(me, ascribes, (x, αA, αP))` | `axes` f16×4, defaults only |
| Entity-indexed salience field S(x); S(me) is self-salience | claims `(me, salience, (x, s))`, top ~64 kept | top-8 relations `rel_id`/`rel_w` |
| Transition cost matrix C[a→b] between motifs | **shared** per culture (asymmetric, authored then tuned) | shared |

### 2.2 Reading the affect measures each tick (proxies, and how they differ from the book)

Let T be this tick's Thought (claims added and retracted by `integrate` and `think`).

- **Valence** = V(v_{t+1}) − V(v_t), where V is the soft distance to the viability boundary: `V = −logsumexp(−k·margin_i)/k`. Plans get an *expected* valence, the change in V along the plan's simulated trajectory (a hypothetical scope). Intensity is |ΔV|.
- **Arousal** = weighted size of the belief update: Σ over T of `confidence × (1 if added else 0.5 if retracted)`, normalized by the working-memory cap. For background agents: L1 change of needs plus relation weights.
- **Φ proxy (integration)** is **not IIT Φ**. It is the fraction of this tick's derived claims whose `derived_from` premises come from **two or more modules** (perception, appraisal, social, narrative, plan). tensacode's provenance makes this directly countable. It is then scaled by κ, because κ gates cross-module rules (see 2.3).
- **Effective rank** = exp(H(p)), where p is the normalized weight of the currently active appraisals and intentions. One dominating concern gives ~1; many live concerns give many.
- **Counterfactual weight** = the share of this tick's rule firings and choose evaluations that ran in hypothetical scopes (`scope: plan:*`, `scope: model:x`) rather than the actual scope. Counted from the trace.
- **σ_attention** = share of working-memory claims whose subject is `me`.
- **σ_causal** = share of the chosen intention's supporting premises (from its `derived_from` chain) that are self-model claims.

**Motif classification** is a nearest-prototype lookup in the 7-D reading space. The prototypes come from the sources where they give one (flow, shame, depersonalization, joy, suffering, fear, desire) and are authored otherwise (grief, anger, boredom, awe, love, disgust, curiosity, labeled as ours).
- The asymmetric cost C shapes transitions: the next motif is argmin over b of `distance(reading, prototype_b) + λ·C[current→b]`.
- This gives hysteresis (fear→anger cheap, anger→fear dear) and the one-way-ness of grief.

**Our authored prototypes** (reading order: valence, arousal, Φ, effective rank, counterfactual weight, σ_att, σ_causal; **not from the sources**):

| Motif | Profile |
|---|---|
| **grief** | valence −, arousal low→mid, effective rank low, counterfactual weight high toward a *past* counterfactual (the lost bond), σ_att mid |
| **anger** | valence −, arousal high, effective rank low, counterfactual weight mid toward removing an obstacle, σ_causal high |
| **boredom** | valence slightly −, arousal low, effective rank low, counterfactual weight low, σ_att rising |
| **awe** | valence +, arousal high, effective rank high, σ_att very low, αA of the environment raised |
| **love / attachment** | valence +, the other's S(x) high, αP(x) high, κ-coupled co-regulation with that person |

### 2.3 The perceptual axes as mechanisms

- **Gain γ weights evidence against priors in `integrate`.** A percept claim contradicting a belief updates it as `b' = (γ·obs + w_prior·b)/(γ + w_prior)`.
  - At low γ, stereotypes, grudges and ideology persist against evidence (rumination, prior lock).
  - At high γ, beliefs are volatile. Arousal rises, and anxiety emerges when predictions keep failing.
  - γ is state-dependent: threat and novelty raise it; fatigue and ritual stabilize it.
- **Coupling κ gates cross-module rules.** Each rule is tagged with its modules (`perception→appraisal`, `appraisal→narrative`, `appraisal→plan`, `social→appraisal`). A cross-module rule fires with weight κ (deterministically: its output confidence is multiplied by κ, and outputs below threshold are dropped).
  - At low κ an agent perceives a loss but it barely reaches planning or narrative: flattened valence, compartmentalization, the "world visible but unfelt" regime.
  - At high κ, alarm and attraction propagate. Contagion between agents (2.4) is also scaled by κ.
- **Ascription α(x) picks the model and the moral weight used for x.**
  - **αA(x)** decides whether x is predicted with the agent template (a theory-of-mind scope `model:x` holding what x wants and believes; it costs rule firings and raises counterfactual weight and effective rank) or mechanistically (extrapolate position and last behavior).
  - The default is **animist reuse of the self-template** under uncertainty (high αA₀ in childhood, lowered by learning and culture).
  - **αP(x)** enters `choose` as a *constraint weight*: harming x is forbidden when αP(x)·norm_strength exceeds a threshold, and otherwise costs utility ∝ αP(x).
  - Empathic valence contagion from x is ∝ αP(x)·κ.
  - **Dehumanization** is an explicit mechanism: propaganda, ideology or repeated fear appraisals lower αP (and often αA) for an out-group class. That removes the harm constraint, which is how war becomes choosable. Its reversal (contact, trade, intermarriage raising αP) is the peace mechanism.

### 2.4 Social affect

- **Co-regulation:** bonded agents in proximity pull each other's arousal toward the lower of the two (attachment as a stable basin). The pull is weighted by S(x)·κ.
- **Contagion:** per-tile or per-group valence and arousal coupling, weighted by κ·αP. Vectorized for background agents (measured in the benchmark as a per-tile mean-field step).

### 2.5 Selfhood and mortality

- **Existential burden B**, an accumulator: `B += σ_att · max(0, −expected valence of the self-future) · mortality_salience`.
  - **Mortality salience** spikes on: death of kin or bonded others, near-death, visible aging and illness, and war deaths in one's group. It decays otherwise.
- **The four response families are policy biases that `choose` can take up** when B is high. Each has costs and cultural availability:
  - **Terror management:**
    - raise adherence to one's ideology (prior weight ↑, γ for ideology-contradicting evidence ↓);
    - in-group favoritism; αP(out-group) ↓ under threat;
    - **symbolic immortality** projects: children, monuments, institutional roles, heroic acts. These are world actions that also create durable legacy claims other agents remember.
  - **Meaning maintenance:** narrative-repair rules re-appraise violating events into the agent's story (claims `(event, means, …)`), restoring coherence. Consolidation (2.6) prefers coherent episodes.
  - **Attachment:** seek bonded others and co-regulate. B falls faster in proximity to high-S(x) partners.
  - **Flow:** take tasks whose challenge ≈ skill. While in flow, σ_att falls and σ_causal stays high; B accumulates slowly.
- **Pathology detectors**, predicates over the last N readings. They become claims so they can be explained and treated:
  - **depression:** κ low, valence variance low, effective rank low, over days;
  - **anxiety:** γ high, prediction-error rate high, arousal high, ritual exposure low;
  - **addiction:** a short-period cycle in chosen intentions with high intensity and low traversal of the effective-rank space;
  - **expansive despair:** many options (high effective rank of *options*) with near-zero expected-valence differences;
  - **degenerate evaluation:** alternation between engage and nullify intentions.
- **Culture as affect infrastructure:** institutions are world objects that change axes and basins for everyone they reach.
  - Rituals and temples raise κ and stabilize γ.
  - Festivals raise attachment.
  - Law adds constraints and a lower αP floor for protected classes.
  - Markets reward lowering αA toward trading partners (mechanistic prediction).
  - Schools shape the αA and αP defaults of children.

### 2.6 Memory, consolidation, forgetting (focal agents)

- **Working memory** is the percept snapshot scope, capped at ~12–30 claims per tick (bounded top-N by salience = S(x) · novelty · γ).
- **Episodic memory:** `(event:e, involved, x)`, `(event:e, felt, (valence, arousal))`, `(event:e, at, place)`. Salience s = |valence| · arousal · S(participants), decaying `s *= exp(−Δt/τ)` and boosted when rehearsed (recalled by a rule).
- **Consolidation** runs at the agent's sleep tick:
  - repeated episodes with the same person or place become semantic claims (`trust`, `fear_of`, `place_valence`);
  - the episode claims are then `Store.forget`-ten, except the top-k most salient and the narrative-anchoring ones;
  - the semantic claims keep `derived_from` pointers to the surviving anchor episodes, so "why do you distrust Oda?" still explains.
- **Demotion to the background tier** keeps the top-8 relations, the 8 most salient episodes (ids into the shared event log), ideology and axes. Everything else is dropped, and the demotion is logged.

### 2.7 On consciousness

We model **structural correlates** that the framework associates with experience. The Φ proxy is a provenance statistic, not integrated information, and nothing in this design shows or assumes that simulated agents experience anything. The framework itself treats measuring integration in artificial substrates as unresolved. If the simulation is ever presented publicly, it should say this plainly, and it should avoid designs whose appeal depends on implying suffering (e.g. maximizing depressive regimes for spectacle).
