# 28. A general agent built from parsing and perception, with plugins

*2026-09-18. Status: approved by the owner, being built.*

**The shape, in the owner's words: vision and language in → a cognitive core → action and language out.** The core is modality-agnostic:
- what is seen and what is said both become claims in one store;
- what is done and what is said both come out of the same wants, plans and verified results.

Neither modality has a private path around the core.

## 28.1 What was asked, and why the current assistant cannot do it

The owner wants TensorCode itself to be a **general agent architecture**:
- text and images in; text and actions out;
- every demo is a general chatbot with specialized plugins;
- no large-model core ("avoid heavy matmuls");
- no brittle regex matching, and none of the cheating the evidence audit found (doc 11).

The demo assistant is the opposite. `assistant/language.py` makes 143 regex calls to map a message onto one of 39 intents, and each intent runs a hand-written procedure. The grammar in `tensorcode.language` is used only on clauses the regexes gave up on. Given a 15-line engineering-design prompt, the assistant:
- **split it into 15 "commands"** at line breaks;
- **misfired on one**: "available processes include cnc milling" matched a regex for the word *processes* and listed Linux processes;
- **said "I didn't understand" 13 times.**

Two of those are architectural, not bugs:
- **Keyword matching turns a mention into an action.** "Processes" inside a description is not a request to list processes. Only a parse knows the difference.
- **An intent table has no way to say "I understood this, and it is outside what I can do."** Anything unmatched is either dropped or answered by the nearest pattern.

## 28.2 The architecture

```
 text ─► segment ─► chart parse ─► frames ─► speech acts ─┐
                   (grammar + lexicon,                     │
                    partial cover)                         ▼
 image ─► plugin perception ─► scene ─► claims ─────► claim store ◄── memory, recall
                                                           │
                                   wants (open requests / questions)
                                                           │
                      ground: unify frames with plugin capability schemas
                                                           │
                    plan: search over capability pre/effects ─► tc.Plan
                                                           │
                  act: tc.invoke via plugin ─► verify against a fresh observation
                                                           │
 text ◄── generate (grammar, reversed) ◄── answers, receipts, Unknowns, questions back
```

Every box is a TensorCode operation behind the runtime (`parse`, `classify`, `choose`, `rank`, `check`, `verify`, `invoke`). Each one is traced, typed, and allowed to return `Unknown`.

### Understanding text: a parse, not a pattern

- **Segmentation.** Split on sentence punctuation and paragraph breaks; never on "and"/"then" followed by a verb. Each sentence is parsed on its own, and discourse state (`language/discourse.py`) carries entities and pronouns across sentences.
  - This also removes the chart parser's silent 40-token cap (`chart.py:237`), which dropped the last 24 words of a 64-word prompt in the survey.
- **Chart parsing.** Earley parsing with feature unification (`language/chart.py`, `features.py`, `english.py`), plus the Viterbi cover. A sentence the grammar only partly covers still yields its parsed pieces, together with a coverage score. Nothing is guessed silently.
- **Grammar gaps to close.** The survey found these missing or broken:
  - relative clauses ("a device *that* can travel");
  - to-infinitives ("able *to* build");
  - subordinate clauses (if / because / while);
  - adverbs;
  - measure phrases ("250 g", "under $40");
  - comparatives ("less than").
  Each is a handful of general productions, not a phrase list.
- **Open vocabulary.** A large lexicon loaded as data (word lists derived from a public resource such as WordNet), plus morphology. An unknown word gets the category that best explains the sentence (`grammar.py` already guesses and marks these). Then it is written back to the lexicon with provenance: learned from context, not hand-listed.
- **Speech acts from grammatical mood, not keywords:**
  - imperative → a request (a want to change the world);
  - interrogative → a question (a want to know);
  - declarative → something told, stored as a claim scoped to the user (`to_claims` already scopes reported speech and negation).
- **Allowed and forbidden regexes.** Tokenization and morphology are allowed. Anything mapping surface strings to intents, topics or answers is forbidden. A test enforces this with an allow-list of regex call sites in the agent package.

### Understanding images: plugin perception, shared representation

A plugin supplies a perception provider that turns an image, or its environment's own state, into a `PerceivedScene` (`perception/protocol.py`). `mind.scene_graph` then turns the scene into claims.
- The computerworld plugin reads the engine's scene API; no pixels are needed.
- The pixel path (docTR OCR plus a small detector, about 236 ms per frame) is small CNNs, not a language model. It is optional.
  - The OmniParser detector is AGPL. It cannot ship in a default install.

### Plugins

A plugin is a Python module that declares:

| Part | Purpose |
| --- | --- |
| `vocabulary` | lexicon entries for its domain (e.g. *folder*, *terminal*), as data |
| `capabilities` | typed actions: parameters, a **frame schema** (which verb senses and role types it realizes), **preconditions** and **effects** as claim patterns, and effect kind / idempotence (existing `ActionSpec`) |
| `perceive()` | observation → `PerceivedScene` → claims |
| `execute(cap, args)` | carries out an action and returns a `tc.Receipt` |

**Grounding** is unification of a request frame with capability schemas: predicate sense plus role types. The request "make a folder called recipes on my desktop" parses to `make(object=folder{name=recipes, location=desktop})`. That unifies with the desktop plugin's `create_directory(path)` schema because `make` + an artifact-typed object + a location role fits. The frame decides, not the word *folder* anywhere in the string.
- If nothing unifies, the agent says so, once, stating what it did understand.

**Planning** is new code: bounded search (goal regression over effects) that yields a `tc.Plan`. The inputs mostly exist:
- `learning/induce` already induces preconditions and effects from traces;
- `expectation` predicts what an act should change;
- `metacognition.Repair` handles replanning.

Verification is always a fresh observation (`tc.verify`), never the executor's own report.

**First plugins.**
- **Desktop (computerworld).** Owns the engine on one worker thread, because its handle cannot cross threads. Its capabilities are the primitives: launch, click, type, key, run a command, read a file. The grader stays in the privileged `CwWorld` session, out of the agent's reach.
- **Self.** Introspection over the trace, to answer "what did you just do / why".

### Replies

`language/generate.py` realizes frames by running the same grammar backwards, with a round-trip check. Content comes from:
- answers;
- receipts and verdicts ("I created ~/Desktop/recipes and checked that it's there");
- `Unknown`s (what was not understood, and why);
- clarifying questions (`social.py` already decides when to ask and when to act).

**One message gets one reply turn**, however many sentences the message has.

### What this can and cannot do with the factory prompt

With this design, the factory prompt parses into a request, roughly `design(object=device{mass<250 g, range≥100 m, payload=50 g, cost<$40, …})`, plus the listed deliverables. No plugin has a `design` capability, and the claim store holds no knowledge of motors, batteries or processes. So the honest reply is one message:
- a restatement of the constraints it extracted;
- a statement that it has no capability or knowledge to design hardware;
- which plugin or knowledge source would be needed.

That is the ceiling without a large model or a domain plugin. The architecture's job is to reach that ceiling honestly instead of listing Linux processes. Producing a real engineering design would take a design plugin: a component catalog, a physics and cost model, and a planner over fabrication processes. That plugin could be built later on the same interface.

## 28.3 What gets deleted, kept and moved

| Code | Fate |
| --- | --- |
| `assistant/language.py` (regex parser), `procedures.py`, `procedure.py`, `interpreter.py`, `programs.py`, `learning.py` (LLM teacher), `teacher.py` | **delete** (git keeps them); their tests go with them |
| `language/domains/desktop.py` | the vocabulary moves into the desktop plugin as data; the hand-written frame→act mapping is deleted (it contains test-set-shaped entries, e.g. *weather/joke/sandwich* tagged off-topic to match the benchmark's negatives) |
| `semantics_bridge.py` regex cues (`\bif\b` on raw text), `answer_type.py` word lists | replaced by grammar productions and by `Question.asked` from the parse |
| `backends/neural.py` (ELECTRA transformers, 13.5M / 109M params) | out of the default agent (it's matmul-heavy and trained on a closed intent set); stays an optional backend |
| `language/{chart,features,grammar,english,semantics,generate,discourse}`, `records`, `cognition`, `memory`, `wants`, `control`, `metacognition`, `expectation`, `social`, `learning/*`, perception protocol, `CwBody`, `CwWorld` | **keep**; these are the building blocks |
| `change`, `permanence`, `frames` (desktop-leaning) | move to the desktop plugin, or generalize |

## 28.4 Evaluation that cannot be gamed

These rules are drawn from the thirteen patterns in docs 11 to 13:

1. **Held-out prompts are written by someone else or taken from public datasets,** and frozen with a recorded hash before any run. Candidates:
   - the task instructions of a public Ubuntu desktop benchmark (e.g. OSWorld);
   - NL2Bash descriptions for file and shell requests;
   - a public general-instruction set for out-of-scope prompts;
   - the owner's own prompts, like the factory one.
   No test case may be derived from the agent's grammar, lexicon or schemas.
2. **Three splits: dev, calibration and test.** Any code change after the test split is read retires that split.
3. **World-state grading only, through the privileged channel.** Never grade the agent's own report.
4. **Required columns in every headline:**
   - **false-action rate:** did something the user didn't ask for, which is the "processes" failure;
   - **honest-refusal rate:** said what it understood and couldn't do;
   - **task success;**
   - **controls:** always-refuse, random-capability and plain-baseline arms;
   - **Wilson intervals.**
   The first target is false-action rate, not success.
5. **Environments and graders are frozen at registration.** Changing one creates a new named configuration; the original stays the headline.
6. **Results files are append-only and content-addressed.** Nothing is overwritten.
7. **A fix counts only if it also holds on a dataset its author didn't use while developing.**

## 28.5 First vertical slice (built in this order: grammar induction and its non-regular test first, then the agent core)

1. `tensorcode.agent`: the loop above, the plugin protocol, one-reply-per-message, and generation from frames.
2. Sentence segmentation plus the grammar productions listed in 28.2, and the lexicon as data.
3. The desktop plugin on computerworld: primitives as capabilities with pre/effects, and grounding by unification.
4. A planner for single-goal requests (regression, depth ≤ 6), with verify-after-act.
5. The chat server and page, rewritten on the plugin interface. The page shows only what the agent emits: parse, frames, wants, plan steps, receipts. No per-prompt elements.
6. Evaluation per 28.4:
   - an out-of-scope set (target: false-action rate 0 with a reported interval);
   - a desktop file/shell set from NL2Bash descriptions, graded by world state (success rate reported, whatever it is).

**Done when:**
- the factory prompt gets one honest reply;
- no regex outside tokenization and morphology exists in `tensorcode.agent` or the plugins (enforced by a test);
- both evaluation sets have frozen hashes and reported numbers, including the unflattering ones.

## 28.5b Learning: the seed is where it starts, not where it stops

**Owner, 2026-09-18:** the agent must be able to *learn* language, including non-regular (hierarchical) structure. Vision must likewise learn hierarchical features, not match icons.

The hand-written English grammar and the WordNet lexicon are seeds in the leapfrog sense. The architecture must also be able to acquire structure the seed does not have.

**Language.** Grammar induction from positive examples, producing productions the same Earley parser uses.
- **First learner: distributional learning of substitutable context-free languages** (Clark & Eyraud 2007).
  - Two substrings that share a context are put in the same class; the classes become nonterminals.
  - It identifies that class of languages in the limit from positive data, in polynomial time, with no neural network.
  - That class includes non-regular languages such as aⁿbⁿ and the Dyck languages.
- **Extensions for natural text:** k,l-substitutability (Yoshinaka), and merging by minimum description length.
- **The falsifiable test:**
  - learn aⁿbⁿ, Dyck-1, Dyck-2 and center-embedding (w c wᴿ) from strings up to length L;
  - accept and reject correctly up to length 3L, against near-miss negatives;
  - beat a regular-language learner (n-gram or state-merging) on the long strings, which no finite-state model can get right.
- **Then English:** delete a construction from the seed grammar (e.g. relative clauses), expose the learner to text that uses it, and measure whether the parser recovers it on held-out sentences.

**Vision.** Hierarchical, compositional features learned layer by layer from unlabeled images: edges → parts → objects. Each layer's vocabulary is learned from co-occurring compositions of the layer below (as in learned compositional shape hierarchies), not from backpropagation through a large network.
- Tested the same way: learned on one set of images;
- evaluated on held-out images and classes;
- against a flat, non-hierarchical control with matched feature count.

## 28.6 Decisions (owner, 2026-09-18)

1. **Small CNNs for pixels are acceptable** (docTR OCR plus a small detector). The AGPL OmniParser detector stays out of any default install.
2. **The lexicon is seeded from WordNet 3.0.** This follows the "leapfrog seeding" approved in synthEX (`docs/leapfrog-seeding.md`): seed from curated structure instead of re-deriving it, then test generalization beyond the seed.
   - **Caution carried over from symbolic-ai-models:** curated classes did no better than a random partition of matched balance (`notes/questions/2026-08-13-seeding-balance-not-provenance.md`).
   - So WordNet is used for **coverage**: word forms, parts of speech, lemmas, is-a for role type checks.
   - Any claim that its **categories** improve behavior must beat a matched-balance random-partition control.
   - The data is read from a local WordNet download and never committed.
3. **Held-out prompts come from public datasets and must cover a broad variety of use cases:**
   - desktop and GUI tasks;
   - shell and files;
   - questions about what is on screen;
   - questions about images;
   - facts told in conversation;
   - general knowledge;
   - arithmetic;
   - open-ended requests such as design, writing and brainstorming;
   - ambiguous requests that need clarification;
   - multi-step instructions.
   They are assembled by an author who does not read the agent's code. The test split stays unread by the implementer.
