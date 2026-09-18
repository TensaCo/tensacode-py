# 7. Pixels → scene graph: vision perception for real desktops

Status on 2026-09-17: prototype with a measured baseline. **Text-labeled controls can be
found and clicked from pixels on apps and operating systems the rules never saw.**
**The desktop agent finishes and verifies 7 of 10 episodes from pixels alone** with the
fine-tuned recognizer (was 2 of 10 before verified typing), 21 of 30 checks pass (was 13),
and **no episode reports success on a misread name any more** (was 2 of 10); the DOM version
finishes 10 of 10. On a real Linux desktop, the AT-SPI provider verified a small task in a
scratch window end to end (7.6.3).

**Which numbers survive the move to computerworld.** Environment-independent: crop and
word accuracy of the recognizers (7.6.6), the provider fusion comparison on recorded frames
(7.6.2), the real-desktop probe and acting test (7.6.3), tooltip icon learning (7.6.5), and
the invariant measurement on recorded frames (7.6.7). Environment-dependent, to be re-run on
computerworld: every end-to-end episode count in 7.6, which used the Seed simulator.

Perception is a **protocol with interchangeable backends** (`examples/browser_agents/perception/`):
web DOM, Linux AT-SPI (live on this machine), visual segmentation, recorded fixtures, and
Windows/macOS stubs, with a fusion provider that merges them and keeps disagreement visible
(7.6.2). The visual backend itself is in `examples/browser_agents/vision/`. Evaluation is `eval/vision_capture.py` and
`eval/vision_perception.py`, with results in `eval/results/vision_perception.json` and
`eval/results/vision_desktop_e2e_{pixels,pixels_corroboration,dom}.json`. The tests are
`tests/test_vision_perceive.py` (synthetic images, no models) and, for the tentative-belief
policy, `tests/test_cognition.py`.

## 7.1 Why this exists

Browser-agent perception (`browser.py`) reads the DOM: roles, names, and hit-tested
click points. A real desktop has no DOM, and accessibility trees (AT-SPI, UIA, AX) are
often incomplete. This module takes **only a screenshot**. It produces the same `Screen`
the DOM path produces, and the same scene-graph claims. The mind, rules, and intention
code run unchanged.

## 7.2 Pipeline

```
screenshot ─┬─ OCR (docTR)                 words + boxes + confidence
            ├─ UI detector (OmniParser v2) interactable boxes + confidence
            └─ classical CV (OpenCV)       rectangles, window frames
                     │
        phrases (words on one baseline, split at wide gaps and window edges)
                     │
        role rules → controls (button / textbox / checkbox, name, click point, window)
                     │                       + icon memory: names for icon-only controls
        Screen  ──►  Fragment (snapshot scope, tc.Score per claim, rule named as basis)
```

The role rules are plain Python in `perceive.py`, and each inferred control records the
rule that produced it:

| Rule | What it recognizes |
| --- | --- |
| Square box left of a phrase | Checkbox. The square needs straight edges, and an interior that is either page-colored or filled with a mark. Dots and icons are rejected. |
| Centered text in an outlined or filled rect | Button |
| Left-aligned text in a bordered wide rect | Textbox. Faint text becomes the hint, dark text the value, and the label comes from the nearest phrase above or to the left. |
| Filled row without a border | Selected list item, treated as a button |
| Row holding another control, with its own label | List row |
| Detector box | Named by the text inside it, a label beside it, or (for large square icons) a label below it. Otherwise named by icon memory, or left unnamed. |
| `user@host:path$` line (also `$` misread as 5, S or §) | Command prompt. The textbox starts where the prompt text ends, because that is where the caret is. |
| Short free phrase (≤5 words) | Low-confidence "clickable text" candidate |

**Windows.** A long top edge must meet a long vertical side edge, with a title-like phrase
just below it (or just above, when the title bar blends into the desktop). Frames nested
inside another frame are treated as panes. An element belongs to the smallest window
containing it, preferring windows whose four edges were all seen.

**Icon memory** (`icon_memory.py`) is a nearest-neighbour store: a 24×24 thumbnail per
named icon, cropped to its content, matched by cosine similarity with a 0.9 threshold. It
learns from any labeled source. In the evaluation it was fit only on the tuning split's
DOM labels (78 examples, 45 names). It recognizes the same icon art again; it does not
generalize to a different icon set.

No LLM or VLM is used anywhere: not at runtime, and not for labeling.

## 7.3 Models, memory, latency

| Model | Role | Weights | CUDA allocated after load | Load time |
| --- | --- | --- | --- | --- |
| docTR `fast_base` + `crnn_mobilenet_v3_large` | Text detection and recognition | 60.4 MB (parameters) | 60.6 MB | 2.2 s |
| OmniParser v2 `icon_detect` (YOLOv8, via ultralytics) | Interactable-element boxes | 40.6 MB file | 81.8 MB | 0.4 s |

Everything else is numpy and OpenCV on the CPU. Both models expose `free()`. Total GPU
footprint is about 140 MB, which leaves the rest of the 121 GB machine free.

**Licensing.** OmniParser's `icon_detect` weights are AGPL-3.0 (inherited from YOLO).
That is fine for research, but it matters before shipping. The detector is behind a
one-function interface, so it can be swapped.

**Per-frame latency** over the 72 evaluation frames (1280×800 or 1100×760), GB10:

| Stage | p50 | p95 |
| --- | --- | --- |
| OCR | 156 ms | 280 ms |
| Detector | 29 ms | 31 ms |
| CV + role rules | 39 ms | 113 ms |
| **Total** | **236 ms** | **372 ms** |

In the agent loop, perception takes 0.76–1.14 s per cycle (p50 per episode). That
includes polling screenshots until the screen settles (two frames 120 ms apart that are
nearly identical).

**Recognizer choice.** Compared on the tuning frames:

| Recognizer (FAST-base detector) | Exact word recall | Digit-word exact | OCR p50 |
| --- | --- | --- | --- |
| crnn_mobilenet_v3_small | 0.871 | 0.831 | 243 ms |
| **crnn_mobilenet_v3_large** (chosen) | **0.885** | **0.867** | **258 ms** |
| crnn_vgg16_bn | 0.894 | 0.871 | 351 ms |
| parseq | 0.888 | 0.868 | 671 ms |
| vitstr_small | 0.889 | 0.866 | 489 ms |
| db_resnet50 + parseq | 0.878 | 0.854 | 662 ms |

RapidOCR (PP-OCRv4) was 3.5 s per frame here, because onnxruntime has only a CPU build
on aarch64. PaddleOCR wheels were not tried; tesseract is not installed.

## 7.4 Ground truth and splits

The frames were captured from Seed simulator computers created for this work
(`vision-ubuntu`, `vision-macos`, `vision-windows`) plus the demo web apps. Each app in
the launcher was opened, and on some frames:
- a terminal ran commands,
- a mail composer was opened,
- a window was dragged,
- the page was reloaded between apps.

For every frame, the screenshot is paired with two DOM reads, one taken just before it
and one just after; the frame is kept only if they agree.

Ground truth:
- **Words:** DOM text split into words, kept only if hit-testable at their own location. Text under another window does not count.
- **Controls:** DOM controls with an unobstructed click point. Invisible window-resize handles are excluded.
- **Windows:** Seed `app-window` frames with at least 30% of their area visible.

| Split | Frames | Content | Used for |
| --- | --- | --- | --- |
| tune | 15 | Ubuntu base / Files / Terminal / Mail; web apps access and shop | Writing and tuning every rule, and fitting icon memory |
| test_app | 25 | Other Ubuntu apps (Text Editor, Slack, Settings, Firefox, System Monitor, VS Code, App Center, Chromium, Rhythmbox, Wireshark); web apps recon, inbox, chart | Held out |
| test_os | 32 | macOS and Windows computers (different themes, window chrome, docks) | Held out |

**Honesty note on the held-out splits.** Held-out metrics were first computed after
rule tuning on `tune`. Some perception changes came afterwards, each driven by an
end-to-end failure, not by test metrics:
- keeping weak detections that icon memory recognizes;
- the broader prompt pattern and prompt input position;
- splitting phrases at window edges;
- the title-above-edge fallback;
- the recognizer switch from mobilenet_small to mobilenet_large.

The tables below are the final re-run. Compared with the first held-out run:
- word, control, and targeting metrics moved by at most 1 point;
- window title accuracy moved the most (test_app 0.591 → 0.636, test_os 0.380 → 0.352).

On seeing the final run, two more changes were made:
- the title comparison now ignores punctuation, because the larger recognizer inserts
  stray `:` and `.`;
- the title-above fallback is disabled for screen-wide frames, where it grabbed the
  system clock as a title.

## 7.5 Metrics

**Targeting by name** is the metric agents need. For each DOM control with a unique name
of 40 characters or fewer, we look it up among the predicted controls the way agents do:
`label_similarity`, minimum 0.55, margin 0.08. We then check that the predicted click
point lands inside the DOM box. "Wrong" means a confident click on a different control;
no match is an abstention.

Full configuration (text candidates on, icon memory fit on tune):

| | tune | test_app | test_os |
| --- | --- | --- | --- |
| Words: fuzzy recall / precision (IoU ≥ 0.5, similarity ≥ 0.8) | 0.923 / 0.820 | 0.827 / 0.776 | 0.766 / 0.624 |
| Words: exact recall | 0.885 | 0.766 | 0.709 |
| Controls detected: recall / precision | 0.880 / 0.348 | 0.781 / 0.435 | 0.850 / 0.614 |
| Button role correct, among matches | 0.984 | 0.984 | 0.996 |
| Textbox recall / role correct, among matches | 0.842 / 0.688 | 0.698 / 0.167 | 0.702 / 0.153 |
| **Targeting, text-labeled controls: right / wrong** (queries) | 0.731 / 0.080 (175) | **0.738 / 0.106** (282) | **0.724 / 0.064** (595) |
| Targeting, icon-only buttons: right / wrong | 0.746 / 0.006 (173) | 0.241 / 0.072 (319) | 0.085 / 0.111 (902) |
| Targeting, fields labeled outside the box: right / wrong | 0.816 / 0.000 (49) | 0.378 / 0.081 (37) | 0.094 / 0.062 (32) |
| Windows found / title correct | 0.643 / 0.571 (14) | 0.727 / 0.636 (22) | 0.563 / 0.352 (71) |

Ablations (same frames, same cached model outputs):

| Change | Effect on held-out targeting (right / wrong) | Effect on control precision |
| --- | --- | --- |
| No icon memory | Icon-only: test_app 0.241 → 0.000, test_os 0.085 → 0.001. Text-labeled unchanged (±1 pt). | ≈ same |
| No short-text candidates | Text-labeled: test_app 0.738 → 0.553 (wrong 0.106 → 0.089); test_os 0.724 → 0.550 (wrong 0.064 → 0.052) | test_app 0.435 → 0.665, test_os 0.614 → 0.729 |

**What the numbers say:**
- **Visible text is the reliable channel.** Finding a text-labeled control by name
  generalizes across unseen apps and unseen OS themes (~0.73 right). Most of the rest are
  abstentions; about 6–11% are wrong clicks.
- **Icons without text are not solved.** Icon memory only re-recognizes icon art it has
  seen. It helps on held-out Ubuntu apps that share the dock and toolbar icons (0.24), and
  barely at all on macOS/Windows (0.09).
- **Form fields are weak off the tuning set.** The textbox *role* is usually wrong on
  held-out frames (0.15–0.17). Search boxes, composer fields, and address bars without a
  crisp border come out as buttons or text.
- **Loose text candidates** raise recall a lot and cost a little in wrong clicks. The
  intended use is with the agent's own checks: `find` abstains on near ties, and the
  agent verifies outcomes before acting further.

## 7.6 End-to-end: the desktop chore agent from pixels

`python -m examples.browser_agents.vision.desktop_e2e --episodes 10 [--dom] [--corroboration]`
runs the unchanged agent from `tasks/desktop.py` on a fresh Seed computer, with the same
seeds for both modes:
1. open Terminal from the dock;
2. `ls` the Desktop;
3. `cat` a note describing a project;
4. run a structure-checked 8-step shell plan;
5. verify by reading the result back.

Two explicit adapters in `desktop_e2e.py` sit between pixels and the task:
- **Efference copy.** The body knows what it typed. A prompt line within 0.8 similarity of
  a typed command is read as that command, including when it wraps over several lines, when
  the OCR splits it into two phrases, and when the prompt's `:`, `~` or `$` is misread.
  Output text gets only two corrections for known OCR confusions: curly quotes, and `-/` or
  `"/` at a word start read as `~/`.
- **Vocabulary.** The prompt textbox inside the window titled "Terminal" is renamed
  "Shell input", the name the task looks for.

### Verified typing (new)

`Browser.fill` no longer types blind. It clicks, then confirms the keyboard is really going
into the field, and only then sends select-all and the text:
- **DOM body:** the active element is an editable field at the clicked point; failing that,
  a probe keystroke must show up in that field's value. The second check matters because
  Seed's terminal routes keys through a window-level handler, so the click leaves
  `activeElement` on `body` and typing still works.
- **Pixel body:** a probe keystroke must visibly change the field's box in the screenshot.
- Either way the probe character is removed with Backspace, so a keystroke that landed in
  some other field is undone.
- If focus is not confirmed, the control is perceived again (its point goes stale when the
  terminal re-renders) and clicked again, then at a point further along the box. If it still
  cannot be confirmed, **nothing is typed** and the receipt is `rejected`; `run_mind` turns
  that into an escalation instead of waiting forever on keystrokes that went nowhere.

This is what turned most of the old hangs into progress, and it removed the class where
Ctrl+A selected the whole page instead of the field.

### Results

Ten episodes, seeds 72001-72010, each mode on its own fresh Seed computer:

| Mode | Finished and verified | All 3 checks correct | Simulator checks | Episode time p50 | Cycles p50 |
| --- | --- | --- | --- | --- | --- |
| DOM, before this change | 10 / 10 | 10 / 10 | 30 / 30 | 6.6 s | 57 |
| DOM, after | 10 / 10 | 10 / 10 | 30 / 30 | 7.1 s | 27.5 |
| Pixels, before (typed blind, single-frame beliefs) | 2 / 10 | 3 / 10 | 13 / 30 | 38.8 s | 46.5 |
| **Pixels, after (verified typing + prompt-reading fixes)** | **6 / 10** | **7 / 10** | **21 / 30** | **20.2 s** | **13** |
| Pixels, after, plus tentative beliefs (`--corroboration`) | 5 / 10 | 5 / 10 | 15 / 30 | 31.1 s | 23 |

"Finished and verified" means the agent read the result back and said it was done; the next
column counts episodes where the simulator's own three checks all pass, which is one more in
two runs (the agent did the work but would not call it finished).

Failure classes per run, same seeds:

| Class | Pixels before | Pixels after | Pixels after + corroboration |
| --- | --- | --- | --- |
| Note misread, agent escalates honestly | 3 | 1 | 1 |
| Keyboard focus missed: silent hang (before) / honest escalation (after) | 3 | 0 | 1 |
| Command completion not recognized: hangs to the cycle limit (work correct in 1 of 2) | 2 | 0 | 1 |
| **Reported success on a misread project name** | 0 | **2** | **2** |
| Work correct, but the read-back check did not match, so the agent escalated | 0 | 1 | 0 |
| Finished and verified | 2 | 6 | 5 |

The two "reported success" episodes are the serious ones. The agent reads the project name
from the note, the recognizer turns a `0` into an `8` (`signal-desk-72804` for
`signal-desk-72004`), and everything downstream is consistent with that name: the plan, the
shell commands, and the read-back check all use it. The agent says it verified the work, and
the simulator scores 0 of 3 because the project it was asked for does not exist. Before this
change those episodes hung instead, which scored the same but at least did not claim success.

### Tentative beliefs: measured, and worse here

The new `Corroboration` policy (see 7.6.1) was meant to fix exactly that misread. It does
not, and it costs progress, so it is **off by default** for this task (`--corroboration`
turns it on). Two reasons, both measured:

1. **The OCR errors are systematic, not flickery.** On a terminal digit probe (12 frames,
   948 digit-bearing words with DOM ground truth), reading each frame a second way agrees
   with the first reading on 96.8-100% of words, and *agrees on a wrong reading* for 12.7-15.8%
   of them. Waiting for two agreeing frames therefore keeps the same wrong name.

   | Second look | Words | First read exact | Second read exact | Both agree | Agree and wrong |
   | --- | --- | --- | --- | --- | --- |
   | Shift 3 px | 924 | 0.844 | 0.844 | 1.000 | 0.156 |
   | Shift 7 px | 948 | 0.848 | 0.842 | 0.968 | 0.127 |
   | Scale 1.25x | 912 | 0.842 | 0.836 | 0.993 | 0.158 |
   | Scale 0.9x | 960 | 0.850 | 0.840 | 0.977 | 0.138 |

2. **It costs a cycle per reading.** Every text claim needs a second frame before the mind
   may act on it, so episodes take 31 s instead of 20 s, and the agent sometimes re-issues a
   command whose prompt line it cannot see yet (harmless for `cat`, but it used a cycle).

What it does still buy: a frame caught mid-render can no longer create a belief. That was a
real failure earlier in development (a listing read while the terminal was still drawing).
It is kept as an optional policy because that class exists, and because an agent whose
perception *is* noisy frame to frame is the case it was written for.

### 7.6.1 The policy itself

`tensorcode.cognition.Corroboration` (about 60 lines, optional, no change to `records.py`):

- `k` frames of the same claim before it may be acted on (default 2); `min_confidence`
  below which frames do not count at all; `predicates` to govern only some perceived
  predicates (here `reads`, so button geometry stays instant).
- `Thought.established` reports claims that crossed the threshold, so `think` treats them as
  new and `established_only` rules fire then.
- Derived claims inherit: a derivation is established only when some line of support is.
- `confirm(mind, claim_id)` establishes a claim an action's outcome bore out.
- `policy.view(mind)` is a read-only view that hides tentative claims; `run_mind` passes it
  to the intention function, so deliberation cannot act on a single glance.
- Contradiction retracts rather than keeps: in a snapshot scope by no longer being
  perceived, elsewhere by a different object for a functional predicate.

Tests are in `tests/test_cognition.py`: establishment after `k` frames, a contradicted
glance retracted with its replacement starting over, functional contradiction outside
snapshots, inheritance plus a strict rule waiting, low confidence needing `confirm`, and
that ungoverned predicates and a policy-free mind behave exactly as before.

## 7.6.2 Interchangeable perception providers

Perception is no longer "the DOM" or "pixels": it is a protocol with several backends that
can be swapped per body or per task, and combined.

```python
class Provider(Protocol):
    name: str
    reliability: float               # prior on this source's structure; fusion uses it to resolve conflicts
    def available(self) -> bool: ...
    def perceive(self, target: Target) -> PerceivedScene: ...
```

`Target` says *what* to look at (a Playwright page, a screenshot or a way to grab one, an
application or window title, a coordinate offset). `PerceivedScene` holds `Element`s
(role, name, value, hint, section, box, hit point, state), `TextBlock`s, `Region`s
(windows, dialogs, graphics) and tables. Every item carries `provenance` (source, locator,
method, detail), a `confidence`, and `conflicts`. `scene.to_screen()` produces exactly the
`Screen` the existing agents consume, so nothing had to change for them, and
`ProvidedBrowser(page, provider, ...)` is a body whose `observe` uses any provider.

| Backend (`examples/browser_agents/perception/`) | Status | What it gives | Reliability |
| --- | --- | --- | --- |
| `web_dom.WebDomProvider` | live | Page DOM/ARIA roles, names, values, hit-tested points, live regions, tables (what the agents already used) | 0.95 |
| `atspi.AtspiProvider` | live on this machine | Linux desktop accessibility (AT-SPI 2 over D-Bus, via `gi`): real windows, roles, names, values, states, extents | 0.93 |
| `visual.VisionProvider` | live | OCR + UI detector + CV rules on any screenshot | 0.60 |
| `fixture.FixtureProvider` | live | A scene handed to it: tests, and replaying recorded frames | set per use |
| `web_dom.RecordedDomProvider` | live | A captured DOM scene, optionally *degraded* (names dropped from icon-only controls) for offline comparison | 0.95 |
| `uia.UiaProvider` | stub | Windows UI Automation: the module documents the exact calls, control-type mapping and `GetClickablePoint` semantics a real one needs | 0.95 |
| `ax.AxProvider` | stub | macOS AX: attributes, role mapping, and why occlusion must come from window order or vision | 0.90 |
| `fusion.FusedProvider` | live | Several providers merged by agreement | best of its parts |

**Fusion rules.** Structure follows the most reliable provider that saw an element. A
missing field is filled from a weaker provider, with its confidence and provenance. Two
different readings never collapse: the more reliable one is kept, its confidence drops to
at most the ratio of the two reliabilities, and the other reading is recorded as a
`Conflict`; if the providers are equally reliable the field becomes empty with both
readings kept, which is the scene-level way of saying Unknown. Elements only one provider
saw are kept. Agreement raises confidence (`1-(1-a)(1-b)`). `scene.uncertain` lists
everything with a conflict or confidence below 0.5.

### Measured: DOM vs vision vs fused

`eval/perception_fusion.py` on the same 72 frames (`eval/results/perception_fusion.json`).
The DOM is also the ground truth here, so its row is an upper bound by construction, not a
measurement; `dom-degraded` removes accessible names from controls whose name is not
visible as text, which is what a native toolkit with partial accessibility looks like.

| Configuration | words covered (tune / test_app / test_os) | targeting: text | icon | field | items flagged as conflicting |
| --- | --- | --- | --- | --- | --- |
| dom (= ground truth) | 0.78 / 0.66 / 0.69 | 0.90 / 0.87 / 0.91 | 0.99 / 0.95 / 0.86 | 1.00 / 0.95 / 0.91 | 0 |
| vision | 0.94 / 0.85 / 0.86 | 0.73 / 0.76 / 0.73 | 0.75 / 0.24 / 0.09 | 0.82 / 0.38 / 0.09 | 0 |
| **dom+vision** | **0.99 / 0.96 / 0.95** | 0.90 / 0.87 / 0.91 | 0.98 / 0.93 / 0.84 | 1.00 / 0.95 / 0.94 | 353 / 549 / 1255 |
| dom-degraded | 0.78 / 0.66 / 0.69 | 0.99 / 1.00 / 1.00 | **0.00 / 0.00 / 0.00** | 0.00 / 0.00 / 0.00 | 0 |
| **dom-degraded+vision** | 0.99 / 0.96 / 0.95 | 0.95 / 0.96 / 0.97 | **0.73 / 0.24 / 0.09** | 0.82 / 0.38 / 0.09 | 281 / 268 / 564 |

What this says:

* **Fusion is worth it for text coverage.** The DOM misses a quarter to a third of the
  words a person can see (canvas drawings, custom-painted widgets). Adding vision takes
  word coverage from 0.66-0.78 to 0.95-0.99 without touching targeting accuracy.
* **Fusion repairs a thin accessibility tree.** With icon names removed, naming
  icon-only controls goes from 0 to 0.73 (same icon set as the memory) and 0.24 on
  held-out apps; fields labelled outside their box go from 0 to 0.38.
* **Fusion costs a little precision and flags a lot.** Vision-only elements lower control
  precision (0.23 -> 0.20 on test_app), and 5-17 items per frame carry a conflict, nearly
  all of them OCR text differing from DOM text. They are visible, not silent, which is the
  point; an agent can consult `scene.uncertain` before acting on one.
* Pure `dom+vision` does slightly *worse* than DOM alone on icon targeting
  (0.93 vs 0.95 on test_app): extra vision-only elements occasionally win a name match.

## 7.6.3 Real desktop: read-only probe, and one acting test

`AT-SPI works on this machine` (X11 GNOME session, `gi`'s `Atspi`, no
`toolkit-accessibility` setting needed). Probed read-only on the user's own screen with
`$SP/vphysical.py`, one 1920x1080 screenshot and the live tree, driving nothing:

| | AT-SPI | Vision (CPU) | Fused |
| --- | --- | --- | --- |
| Interactive elements | 69 (61 named, 88%) | 144 | 191 (22 seen by both) |
| Text blocks | 28 | 118 | 143 |
| Regions | 322 | 2 | - |
| Time for the whole screen | **2.0 s** (609 nodes walked) | 163 s on CPU (0.25 s on GPU) | - |
| Conflicts flagged | - | - | 13 |

* AT-SPI named the things vision cannot: the real GNOME dock (`Activities`, `Files`,
  `Google Chrome`, `Terminal`, `jterm`), top-bar menus, tab strips.
* Vision found *something* at the same place for 55 of 61 named accessible elements (90%),
  but agreed on the **name for only 1 of them**: they are icon-only, so there is no text to
  read. This is the clearest argument for accessibility-first on real desktops.
* AT-SPI's gaps showed up too: 8 of 69 elements unnamed, including a 901x861 text area (the
  editor's document) - exactly where OCR has something to add.
* Roles seen: 37 button, 12 row, 8 tab, 6 textbox, 4 menuitem, 2 list.
### One acting test, in a scratch window this code launched

A small GTK window of its own making (`$SP/scratch_app.py`: a "Project name" entry, an
"Urgent" checkbox, a "Record project" button, a status label), placed in a screen corner.
The task: type a name, tick Urgent, press the button, then verify by reading the status
label back. Acting is real X11 input through `xdotool`, from
`perception/desktop_body.py`; perception is a provider.

| Provider | Verified the task | Perception p50 | Episode | Actions |
| --- | --- | --- | --- | --- |
| **AT-SPI** | **yes** - status read back as "recorded: ledger-lab-78077 (urgent)" | 48-92 ms | 2.1 s | 6 |
| AT-SPI + vision | no - the run refused to type (see below) | 177-200 ms | 2.0 s | 5 |

What simulation hid, all of it found in this one test:

* **X11 key events go to whoever holds input focus.** `xdotool type --window` uses
  XSendEvent, which GTK ignores outright, so typing must go through XTEST to the focused
  window. A screenshot grab (`import -window`) or a window-manager refusal moves that focus,
  and early runs therefore typed into whatever window was focused instead. The body now
  *verifies* focus is on its own window before any key event and refuses otherwise, which is
  why the fused run reports "rejected" rather than typing somewhere unknown.
* **AT-SPI's interfaces are module functions, not node methods**
  (`Atspi.Text.get_text(node, 0, n)`), and each node caches: a field just typed into reads
  stale until `node.clear_cache()`. Until that was fixed, every field read came back empty
  and verified typing could not confirm focus.
* **Coordinate frames differ per provider.** Accessibility reports screen coordinates;
  a window screenshot is window-relative. One `Target.image_origin` now puts both in screen
  space. Getting this wrong put a widget at y=1283 on a 1080-tall screen, and the body
  correctly refused to click outside its own window.
* **Fusion can add a rival for the same control**: vision saw the checkbox's *label* as a
  separate button. Fusion now merges a weaker provider's element that sits inside an
  interactive element the stronger provider already has.
* This is **one task on one window**, not a benchmark. Further real-desktop testing is
  **pending the user's go-ahead at a time that suits them**: they were working, so all
  display-touching work (screen capture, AT-SPI probes of the live session, pointer and key
  events) is stopped.

## 7.6.4 Killing false success: four attempts, the fourth works

The worst failure in 7.6 is the agent reading a project name wrong, doing the work under
that name, and reporting success (2 of 10 episodes). Four guards were built and measured.
**Three failed. The fourth - retraining the text recognizer on a wide corpus of free labels -
removed both false successes without costing any successes.**

| Attempt | Flag | Measured result |
| --- | --- | --- |
| Cross-place digit check | `--cross-check` | **Fails.** Unrelated identifiers legitimately differ by one digit (sequential seeds, ports), so 6 of 10 episodes lost the task note. Would not have caught the real case either: the misread name appears only once on screen. |
| Confidence gate | `--refuse-below 0.95` | **Fails.** On a 960-word probe, below 0.8 catches 0% of misreads; 0.95 catches 67% but refuses 6% of correct reads, and that 6% lands on the one filename the task needs: 0 of 10 episodes. Prompt lines were exempt (efference copy corroborates them) and it still failed. |
| Fine-tune on a *narrow* corpus | - | **Fails.** 4,423 crops of short terminal echoes only: crop accuracy rose (0.839 -> 0.986) but prose regressed (0.760 -> 0.744), `~` started reading as `"`/`#`, and end-to-end fell to 3 of 10. |
| **Fine-tune on a wide corpus** | `--reco-weights` | **Works.** See 7.6.6: false successes 2 -> 0, verified episodes 6 -> 7, all-checks 7 of 10 unchanged. |
| Task-stated invariant | (library) | **Promising, measured offline.** See 7.6.7. |

## 7.6.5 Icon names from tooltips

`examples/browser_agents/vision/tooltips.py` learns icon names the way a person does:
hover the icon, watch for text that appears next to it, OCR that, and store the name
against the icon's thumbnail in an `IconMemory`. No hand labels and no accessibility tree.
A tooltip is detected as *new text near the icon* between the frame before the hover and
the frame after, so it does not matter what draws it.

One 24-second tour of a Seed Ubuntu desktop: 27 icon-only controls hovered, 11 tooltips
read, 10 names learned, of which 8 correct (`Wireshark`, `Slack`, `Text Editor`,
`System Monitor`, `Firefox`, `App Center`, `Rhythmbox`, plus one more) and 2 OCR junk.

Icon targeting with that memory, against the 78-example memory fit from DOM labels
(`eval/results/vision_perception_tooltip_icons.json`):

| Icon memory | tune | test_app | test_os |
| --- | --- | --- | --- |
| none | 0.00 | 0.00 | 0.00 |
| **learned from tooltips** (10 entries, one tour, zero labels) | **0.31** | **0.06** | **0.00** |
| fit from DOM labels (78 entries over the whole tune split) | 0.75 | 0.24 | 0.09 |

So the mechanism works and needs no labels, but one tour of one screen learns only the
icons on that screen (the dock and the visible toolbars). To match the labelled memory it
would have to tour each app's toolbars, and Seed's macOS dock and Windows taskbar show no
tooltips at all, which is why `test_os` stays at zero. On the real GNOME desktop the dock
does show tooltips, and AT-SPI already names those buttons anyway - which is the better
route there (7.6.3).

## 7.6.6 A wide corpus of free labels, and what it fixed

Every label here is free: either the agent typed the string (so the pixels it rendered are
labelled), or the app's DOM reports the exact text at the same instant as the screenshot, or
this code drew the string itself offscreen in a known font.

| Source | Crops | What it covers |
| --- | --- | --- |
| Terminal echoes and files it wrote then `cat`-ed | 2,600 | the terminal font: paths with `~`, identifiers, prose lines, punctuation |
| Seed desktop apps (Files, Mail, Text Editor, Slack, Settings, System Monitor, App Center) and the demo web apps | 3,700 | UI fonts at their real sizes: menus, buttons, list rows, tables, labels |
| Offscreen rendering in 8 local fonts (DejaVu Sans/Mono/Serif, Cantarell, Liberation Sans/Mono), sizes 11-24, light-on-dark and dark-on-light, slight blur/noise/rescale | 9,000 | font variety, mixed case, digits in context, `~` paths, quoted phrases |

Held out: two fonts (LiberationMono, Cantarell-Regular) and eight apps never trained on.
Training is 5 epochs, ~15 s per epoch on the GB10.

| Recognizer | held-out apps: exact / digits / `~` | held-out fonts: exact / digits / `~` | pipeline: terminal digit words | held-out frames, all words | words with `~` |
| --- | --- | --- | --- | --- | --- |
| baseline (docTR crnn_mobilenet_v3_large) | 0.870 / 0.828 / 0.00 | 0.668 / 0.788 / 0.00 | 0.636 | 0.708 | 0/11 |
| fine-tuned, apps only | 0.891 / 0.877 / 0.83 | 0.658 / 0.694 / 0.52 | 0.727 | 0.703 | 7/11 |
| **fine-tuned, apps + rendered** | 0.880 / 0.862 / 0.83 | **0.794 / 0.882 / 0.72** | **0.727** | **0.711** | **8/11** |

The apps-only model is slightly better on the apps it saw but *worse than baseline on unseen
fonts* (0.658 vs 0.668): a narrow corpus buys accuracy where you trained and loses it
elsewhere. Adding rendered text fixes that (0.794) and keeps every other number at or above
baseline. The `~` class goes from never right to mostly right, which is what broke prompt
reading before.

End to end on the desktop chore (10 episodes, same seeds; **this benchmark moves to
computerworld next, so treat the episode counts as the last Seed numbers**):

| Recognizer | Finished and verified | All 3 checks | Checks | **Reported success on a misread name** | s/ep |
| --- | --- | --- | --- | --- | --- |
| baseline | 6 / 10 | 7 / 10 | 21 / 30 | **2** | 20.2 |
| fine-tuned, apps only | 7 / 10 | 7 / 10 | 21 / 30 | **0** | 23.1 |
| fine-tuned, apps + rendered | not measurable on Seed any more (see below) | | | | |

Same work done, and the two episodes that used to claim success now escalate honestly
("task note unreadable"). That is the first guard that helped rather than hurt.

The apps+rendered recognizer is better than apps-only on every held-out crop and pipeline
measure above, but its end-to-end number could not be taken: the Seed-based desktop harness
was retired while this was running (`tasks/desktop.py` no longer has a `SEED` endpoint, being
ported to computerworld). **The two rows above are the last valid Seed episode counts**, and
the apps+rendered weights need one confirming run on computerworld.

## 7.6.7 Task-stated invariants (the fourth guard, measured offline)

`examples/browser_agents/perception/invariants.py` lets a task declare what it believes
about what it reads, instead of perception guessing:

```python
project = SameIdentifier("project id", r"[a-z]+-[a-z]+-(\d{4,})", r"task-(\d{4,})\.txt")
shape = WellFormed("project name", r"project called ([^\s(]+)", r"[a-z]+-[a-z]+-\d{4,}")
verdict = check(lines, [project, shape])     # corroborated / violations / unseen
```

A value read the same way in two independent places is *corroborated*; two different
readings are a *violation* naming both, so the task can escalate instead of picking one;
if the places are not all on screen, that is *unseen*, not a violation. `WellFormed` is the
second kind and was added later (7.6.8): agreement cannot see a misreading that nothing
contradicts, such as a lost hyphen, but a task usually knows the *shape* of its own
identifiers. `mask(...)` rewrites a failing value out of the line so a grammar downstream
abstains rather than parsing a fragment of it.

Measured on the 11 recorded screenshots of failed pixel episodes (offline, no new runs):

| | frames | would refuse | corroborated | **corroborated but wrong** | invariant not visible |
| --- | --- | --- | --- | --- | --- |
| baseline recognizer | 11 | 1 | 7 | **0** | 3 |
| apps+rendered recognizer | 11 | 1 | 7 | **0** | 3 |

The single refusal is exactly the false-success frame (`px_72004`, readings `72004` vs
`72804`). It never corroborated a wrong value. One thing it needs to be told: **which lines
are data**. Checking every line flagged 4 of 11 frames, because the shell prompt carries the
*previous* task's working directory and the terminal still shows earlier episodes; excluding
the prompt's cwd brings false refusals to zero. So an invariant is only as good as the scope
the task gives it.

Measured live in 7.6.8, on the engine, this invariant alone refuses **nothing**: the
misreads there destroy the identifier instead of altering its digits, so there is no second
reading to disagree with. That is what `WellFormed` was added for.

## 7.6.8 The same run on the computerworld engine, where episodes are exact

Everything in 7.6 through 7.6.7 was measured against the Seed simulator, which the harness
has now replaced with the [computerworld](https://github.com/JacobFV/computerworld) engine
(0.1.0a1, built locally). The engine matters here for three reasons: a world is a definition
plus a seed, so one episode cannot contaminate the next; it renders 1280x800 itself in about
4 ms, so the visual pipeline can be pointed straight at it; and it is deterministic, so an
arm can be re-run and compared line by line.

`examples/browser_agents/vision/cw_e2e.py` runs the unchanged agent from `tasks/desktop.py`
on two bodies over the same worlds:

| Body | Perception |
| --- | --- |
| `CwBody` + `CwProvider` | the engine's own scene — a reference, not a reading |
| `CwPixelBody` + `CwPixelProvider` | the engine's rendered frame, through the same docTR + detector + CV pipeline |

Three things had to be built for the pixel body, and each is a claim about terminals rather
than about this world:

1. **Where to type.** This terminal's prompt is a bare `$`. One frame's recognizer simply
   missed that glyph, the agent lost the field, clicked the dock again, and escalated after
   doing all the work correctly. So the *caret* is used instead: a solid bright block found
   by shape (a letter fills about half its box, the caret fills all of it), and the console's
   whole pane — not the prompt line — becomes the `Shell input`, because clicking anywhere in
   a pane puts the keyboard in the shell and the prompt moves down as output arrives. The
   pane is also remembered between frames, since a terminal does not move on its own.
2. **Which text is output.** Output lines stand on a character grid; a title bar and a tab
   strip do not. The left margin is the leftmost text edge inside the window, and the pane
   starts at the first line standing on it.
3. **Wrapped lines, and whether the join takes a space.** A line whose right edge reaches the
   margin continues below. Whether a space belongs is read off the continuation's *indent*: a
   wrap inside a word resumes at the margin (`"notes/ w"` + `"ith"`), a wrap at a space
   carries that space onto the next row, one character in (`"as 'initial"` + `"commit'."`).
   Guessing "no space", as the first version did, produced a commit message of
   `initialcommit` that the agent then confirmed against its own misreading.

The terminal prints output but never echoes the command that produced it (neither does the
engine's own scene), so prompt lines are an **efference copy** — what the body typed, marked
`method="efference-copy"` in its provenance — in front of the OCR'd output that appeared
while it ran. Output is attributed to a command by how many lines had been read on screen
when it was typed, which is itself a pixel reading.

### Results: ten episodes per arm, two splits, everything else identical

Split A is the ten worlds the wrap rule and the invariants were debugged on. Split B is ten
worlds that were never looked at while writing either. "verified" is the agent saying it
finished; "ok" is verified **and** the world's own three checks passing; "false" is verified
with the checks failing — a claim of success on work that is wrong.

| Arm | Split | verified | ok | **false** | checks | refused | lines read exactly |
| --- | --- | --- | --- | --- | --- | --- | --- |
| engine scene (reference) | A | 10 | 10 | **0** | 30/30 | 0 | by construction |
| pixels, stock recognizer | A | 7 | 4 | **3** | 18/30 | 0 | 108/125 |
| pixels, apps fine-tune (`crnn_wide`) | A | 7 | 4 | **3** | 21/30 | 0 | 100/125 |
| pixels, apps+rendered fine-tune (`crnn_widest`) | A | 7 | 4 | **3** | 21/30 | 0 | 106/125 |
| **pixels, apps+rendered + task invariants** | A | 4 | 4 | **0** | 21/30 | 3 | 81/97 |
| engine scene (reference) | B | 10 | 10 | **0** | 30/30 | 0 | by construction |
| pixels, stock recognizer | B | 5 | 5 | **0** | 24/30 | 0 | 87/107 |
| pixels, apps fine-tune | B | 4 | 4 | **0** | 18/30 | 0 | 71/97 |
| pixels, apps+rendered fine-tune | B | 5 | 5 | **0** | 24/30 | 0 | 84/107 |
| **pixels, apps+rendered + task invariants** | B | 5 | 5 | **0** | 24/30 | 1 | 84/107 |

An episode costs 0.02 s on the engine's own scene and about 3 s through the recognizer; the
refused arms read fewer lines because they stop earlier. Timings and the reading score are
below.

Read honestly, three things follow, and only one of them is what was hoped for.

**The recognizer fine-tunes do not remove a single false success here.** They were expected
to: on the Seed task the apps-only fine-tune took false successes from 2 to 0 (7.6.6). On
this world all three recognizers make the same three false successes on split A, and the
fine-tunes only move the check count (18 → 21 of 30 on A; on B the apps-only fine-tune is
*worse*, 18 vs 24). The reason is visible in the failures: they are not the digit confusions
the wide corpus fixed. They are lost hyphens in the project name —

| Seed | Wanted | Read as | Directory the agent built |
| --- | --- | --- | --- |
| 31288 | `atlas-sync-31288` | `atlas sync-31288` | `~/Projects/atlas` |
| 61899 | `meadow-sync-61899` | `meadow-s sync-61899` | `~/Projects/meadow-` |
| 70452 | `meadow-tools-70452` | `meadow-1 ools-70452` | `~/Projects/meadow-1` |

— after which the plan, every command, and the agent's own read-back are all consistent with
the wrong name, so it reports success and the world scores 0 of 3.

**The wrap-indent rule removed one, before any guard.** The first run of split A had four
false successes; `72804` was the fused `initialcommit`, and reading the space off the indent
fixed it outright. That run is kept in `eval/results/cw_pixel_e2e.json` under
`before_two_fixes`.

**Agreement between two readings cannot see these, but shape can.** The digit invariant from
7.6.7 (`SameIdentifier`, the note's id must equal the file name's id) refuses **nothing**
here, and says so: `project id: not seen in every place`. A name read as `atlas` contains no
id to disagree with. What catches all three is a second kind of declaration, `WellFormed` — a
value whose shape the task knows:

```python
PROJECT_NAME = WellFormed("project name", r"(?:project called|project please:)\s+([^\s(]+)", r"[a-z]+-[a-z]+-\d{4,}")
```

On a violation the value is rewritten out of the line (`project please: <refused> sync-31288`),
so the note grammar abstains instead of parsing a fragment, and the agent escalates.

The obvious objection is that this guard was written after looking at the failures it
catches, and on split A it is fitted: 3 refusals, exactly the 3 false successes. Split B is
the answer to that. There it refuses once, on `99876` (`atlas-notes`, the id lost), an
episode that every arm already failed — so on ten unseen worlds the guard cost **no** correct
episode and claimed **no** false success. That is one held-out refusal, not a distribution;
it is evidence the rule is not merely memorised, not evidence of a rate.

### What the pixel body actually reads, and what it costs

The body scores its own reading against `terminal_lines()` — the engine's logical lines —
after every observation. It is reported and never used.

| | engine scene | pixels (apps+rendered) |
| --- | --- | --- |
| Perception per observation, p50 | 0.5 ms | 223–231 ms |
| Episode, p50 | 0.02 s | 3.0 s |
| Terminal lines read exactly (final frame, A / B) | by construction | 106/125 · 84/107 |
| Episodes with at least one misread line | 0/10 | 9/10 · 10/10 |

So about 85% of lines come back character-perfect, every episode has at least one that does
not, and episodes still finish — because what the task needs is a *parseable* line, not a
perfect one. The note's items and title survive a wrong character; the project name does not.
That asymmetry, not average word accuracy, is what decides this task. (Lines are aligned
before being counted. Counting them positionally instead reports 69/130, because one wrongly
joined line shifts every line after it — worth knowing if you write this measurement
yourself.)

Timings are CPU-load sensitive: the same arms measured 386–406 ms and 5.1–5.3 s per episode
while sharing the machine with other work. The engine rasterizes 1280x800 in about 4 ms, so
essentially all of the pixel cost is the recognizer and the detector.

**Repeatability.** Every arm was run three times. Per-episode status, checks and world
`state_hash` were identical every time, vision pipeline included, so a difference between two
arms here is the arm and nothing else.

**Engine shell gaps worked around rather than depended on** (the sibling fork's
`eval/results/computerworld_shell_gap.json` has the full probe): `find -name` ignores its
pattern and returns every path, so the agent lists `~/Desktop`; `grep -c` prints the matching
line instead of a count; `sed -n Np` is unsupported (only `s///`); `du`, `df`, `which` and
`clear` do not exist, so the body empties a terminal by closing and reopening the window;
`ls -la` ignores its flags. What this task does use, and which all work: `ls`, `cat`,
`mkdir -p`, `cd`, `echo >`/`>>`, `git init`/`add`/`commit -m`/`log --oneline`, and for scoring
`git status` plus the engine's own filesystem reads through a privileged session the agent
does not hold.

**Which earlier numbers this supersedes.** Nothing in 7.6–7.6.7 is retracted, but the Seed
rows in 7.6 and 7.6.6 are now the *old harness*: the simulator they ran on has been removed,
so they cannot be re-run, and the last valid ones are the two rows in 7.6.6. The table above
is the current end-to-end measurement. The one claim that does not carry over is 7.6.6's
"false successes: 2 → 0 with the apps-only fine-tune": on this engine the same fine-tune
leaves all three false successes in place, so that result was specific to that world's
failure mode (digits) and should not be read as a general fix.

## 7.7 What works on real desktops today, and what is missing

**Works today, from a screenshot alone:**
- Reading visible text: about 0.77–0.92 word recall; lower on dense small text.
- Finding and clicking text-labeled buttons, menu items, list rows, and tabs by name, with
  abstention when it is unsure.
- Windows with titles, most of the time.
- Command-prompt lines and where to type, including a bare `$` prompt and, when the glyph is
  missed, the caret next to it (7.6.8).
- Reconstructing a wrapped terminal line, space included, from the continuation's indent.
- Scored, provenance-carrying claims that the existing mind consumes.
- Recognizing icons it has been shown before.

**Missing before it is dependable:**
1. **A guard against confident misreads.** Still the top problem, and still only half
   answered. Two agreeing frames, a cross-place digit check, and a confidence gate all fail
   (7.6.4). A wide free-label corpus removed the digit-confusion class on the Seed task
   (7.6.6) and removed none of it on the engine (7.6.8), where the misreads are lost
   hyphens. What does work there is a task-stated *shape* invariant — 3 of 3 false successes
   turned into refusals, 1 refusal and no cost on ten held-out worlds — but that is a guard
   per identifier the task declares, not a general one. A general guard still needs
   per-character confidence, or a reading the agent can cross-examine by acting (open the
   directory it believes it made and see whether the name is there).
2. **Icons in general.** Tooltips work (7.6.5) but cover only what is toured. Learn names from hover tooltips (they render in screenshots, as
   seen on the Seed dock), or an icon captioner; icon memory alone does not transfer
   across themes — the engine's dock needed its own examples (7.6.8).
3. **Form fields without crisp borders**, and label association for them.
4. **More than one real-desktop test.** 7.6.3 is one read-only probe of a live GNOME session
   plus one acting test in a scratch window this code launched, and that is all; everything
   else is rendered (Seed themes, web apps, and now the engine's own rasterizer). The
   pipeline has no DOM dependency, but its thresholds were tuned on renderings.

## 7.8 Hooks

- `MindSpec.perceivers` already works for extra looks. Full pixel perception is done by
  subclassing `Browser` (`PixelBrowser.observe`), which needed no core change.
- **Done, in `perception/`:** the provider protocol, five live backends, two documented
  stubs, fusion, and `ProvidedBrowser` so a body's perception is selectable per task. The
  existing agents keep the DOM provider and needed no change.
- **Done, in `browser.py`:** a per-body typing strategy. `Browser.fill` clicks, confirms
  focus (`focused`, overridable per body; `field_text` reads a field's content), re-perceives
  a stale point, and returns a rejected receipt rather than typing blind. `TypeText` gained
  `replace`, so a probe keystroke does not select-all, and there is a `PressKey` action for
  the Backspace that undoes the probe. `mind.run_mind` turns a rejected motor receipt into an
  escalation (`MotorFailure`).
- **Done, in `tensorcode.cognition`:** `Corroboration` plus `Thought.established`,
  `Rule.established_only`, and the `corroboration=` argument on `integrate`/`think`.
  `MindSpec.corroboration` is a per-run factory, and intentions see the established view.
- **Done, in `perception/`:** `DesktopBody` (real X11 acting with focus verification and
  window-scoped clicks), `invariants.py` (task-stated invariants), and
  `vision/tooltips.py` (icon names from tooltips).
- **Still wanted:** making the settle policy part of `observe` for non-DOM bodies (today
  `PixelBrowser` does it), a cheap way to ask a recognizer for per-character confidence, and
  a `computerworld` structured/pixel provider pair (another fork is writing those against
  this protocol).

## 7.9 Reproduce

The venv shares torch and playwright with the existing venvs through a `.pth` file. It
adds docTR, ultralytics, OpenCV, onnx, and torchvision 0.28 (cu130, installed with
`--no-deps`).

```bash
PYTHONPATH=src:. python eval/vision_capture.py --out $DATA              # 72 frames, ~6 min
PYTHONPATH=src:. python eval/vision_perception.py --data $DATA --cache $CACHE --overlays $OVERLAYS
PYTHONPATH=src:. python -m examples.browser_agents.vision.desktop_e2e --episodes 10 --first-seed 72001 --hostname vision-e2e-6 --icon-memory icon_memory_tune.npz --out eval/results/vision_desktop_e2e_pixels.json
PYTHONPATH=src:. python -m examples.browser_agents.vision.desktop_e2e --episodes 10 --first-seed 72001 --hostname vision-e2e-7 --icon-memory icon_memory_tune.npz --corroboration --out eval/results/vision_desktop_e2e_pixels_corroboration.json
PYTHONPATH=src:. python -m examples.browser_agents.vision.desktop_e2e --dom --episodes 10 --first-seed 72001 --hostname vision-e2e-8 --out eval/results/vision_desktop_e2e_dom.json
PYTHONPATH=src:. python eval/perception_fusion.py --data $DATA --cache $CACHE      # DOM vs vision vs fused
PYTHONPATH=src:. python -m examples.browser_agents.vision.desktop_e2e --episodes 10 --first-seed 72001 --hostname vision-e2e-N --icon-memory icon_memory_tune.npz [--refuse-below 0.95 | --reco-weights crnn_terminal_ft2.pt | --cross-check | --corroboration]
PYTHONPATH=src:. python -m pytest tests/test_vision_perceive.py tests/test_perception_providers.py tests/test_invariants.py tests/test_cw_pixels.py tests/test_cognition.py
```

The `desktop_e2e` lines above need the Seed simulator, which the harness no longer has. The
current end-to-end run is on the engine (7.6.8) and needs no server at all:

```bash
SEEDS=10007,23145,31288,40613,52074,61899,70452,72804,84031,96720          # split A; B is in the results file
PYTHONPATH=src:. python -m examples.browser_agents.vision.cw_e2e --fit-icons cw_icons.npz --tuning-seeds 3
PYTHONPATH=src:. python -m examples.browser_agents.vision.cw_e2e --seeds $SEEDS --structured
PYTHONPATH=src:. python -m examples.browser_agents.vision.cw_e2e --seeds $SEEDS --icon-memory cw_icons.npz \
    [--reco-weights crnn_wide.pt | --reco-weights crnn_widest.pt] [--invariant] --out eval/results/....json
```

`icon_memory_tune.npz` is produced by `eval.vision_perception.fit_icon_memory` on the tune
split; `icon_memory_tooltips.npz` by the tooltip tour (`$SP/vtooltip_learn.py`); the
recognizer fine-tune by `$SP/vfont_data.py` (free labels from typed text) plus
`$SP/vfont_train2.py`. The real-desktop probe is `$SP/vphysical.py`, read-only. Those
scripts and weights live in the scratchpad, not the repo.

The recognizer fine-tune was the one guard that helped on the Seed task (7.6.6): pass
`--reco-weights` with the weights produced by `$SP/vcorpus.py` + `$SP/vrender_corpus.py` +
`$SP/vfont_train4.py`. On the engine it helps the check count and not the false successes
(7.6.8), where the guard that works is `--invariant`. The other guards in 7.6.4 measured
worse and stay behind flags. Corpus, weights, the icon memory and the real-desktop scripts
live in the scratchpad, not the repo.

Results files: `eval/results/cw_pixel_e2e.json` (7.6.8, both splits, both runs, the
pre-fix run, the shell gaps), `recognizer_comparison.json`, `invariant_eval.json`,
`real_screen_report.json`, `real_acting_report.json`, `perception_fusion.json`.
