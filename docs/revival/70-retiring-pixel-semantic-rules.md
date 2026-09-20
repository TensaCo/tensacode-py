# Retiring pixel semantic rules

The older desktop pixel path combined OCR and detector outputs with fixed geometry,
control-role, and terminal-prompt rules to construct actionable UI interpretations.
Those rules were a shortcut from measurements to meaning. Keeping that path beside
the structured cognitive workspace left an implicit fallback that could bypass
learned, revisable interpretation. This milestone removes it rather than renaming
it or retaining a compatibility switch.

This is a capability removal, not a newly learned visual replacement. Historical
pixel desktop episode results remain historical measurements. They do not describe
a runnable current screenshot-to-action path after this retirement.

## Removed path and retained components

The retired components include `examples/browser_agents/vision/perceive.py`,
the tooltip rule helper, pixel desktop and computerworld evaluation bodies,
`perception/visual.py`, `CwPixelProvider`, and their obsolete perception/fusion
runners. The fixed geometry-to-control and prompt parsing route is not preserved
as a fallback for uncertain learned models. Associated tests whose subject was
that removed implementation are retired too; a lower suite count is not itself
a regression in the remaining tested functionality.

OCR and detector model loading and measurement remain distinct from the removed
semantic interpretation. The learned icon association component is retained.
These components can supply measurements or learned associations; they do not by
themselves establish holistic scene understanding or authorize an action. Saved
result JSON files remain available for auditing the historical experiments.

The structured computerworld provider remains an environment adapter, with a
separate cleanup of invented semantic aliases and prompt rules:

- `CwProvider` preserves the engine's literal control labels; it does not rename
  a terminal control to “Shell input” or convert an unknown interactive role to
  a button.
- Window metadata comes from literal engine focus-region labels, without
  stripping a “Move” prefix or capitalizing a guessed title.
- The unused `split_prompt` regular expression and `terminal_texts` path are
  removed. `CwSurface.prompt` does not manufacture a fallback prompt.
- Terminal identification requires a unique explicit `:terminal-input`
  interaction and its window ID, rather than a title prefix. Missing or ambiguous
  interactions do not become an inferred terminal.
- `CwBody` identifies the terminal section through control provenance when using
  the engine's logical transcript. `browser.Control` retains that provenance so
  the general desktop adapter and older consumers can use explicit interaction
  identity instead of a label.

This is consumption of supplied engine metadata, not learned visual interpretation.
The logical transcript remains an explicit engine channel. Older authored task
policies are not all removed by this milestone; their existence must not be hidden
behind a claim that every hardcoded cognitive rule is gone. Verification of this
cleanup is covered by the focused verification below.

## What behavior is lost

The old screenshot pipeline no longer turns rectangular regions, OCR text,
control-like shapes, or terminal-looking strings into actionable desktop controls.
The deleted pixel bodies and evaluation runners cannot reproduce the old desktop
episodes from the current tree. The old visual/structured fusion runner is likewise
retired. Historical instructions for those commands have been removed from the
vision report instead of leaving broken or misleading recipes.

This does not remove browser document capture, structured computerworld evidence,
explicitly supplied visual scene proposals, neutral evidence graphs, or relational
learning over those graphs. It also does not imply that those structured adapters
observe the same facts as pixels. [Neutral evidence graphs](69-neutral-evidence-graphs.md)
keeps the source modality explicit; a DOM or engine interaction graph is not a
replacement claim of visual perception.

## Current visual capability and remaining work

The active workspace can retain visual scene proposals, relations and anchors,
source-backed alternatives, contradictory or unknown evidence, and learned
relational grounding/query-derived observation mechanisms. The demonstrated graph
learning still depends on supplied graphs and teaching. Literal real-browser
snapshots provide measured structural evidence through their explicit adapter.
Neither result establishes a learned pixel-to-scene model.

Holistic scene construction from images and video remains unfinished. The missing
capability includes organization, grouping, relational identity, temporal change,
affordances, uncertainty, and competing explanations, not merely replacement
control classification. A future pixel provider must supply revisable scene
hypotheses with evidence and evaluated limits; it must not restore the removed
fixed rules under another interface.

The combined focused run passed **60 tests, with one skipped**, in 8.59 seconds,
covering computerworld, recorded DOM, the general agent, explicit desktop arguments,
perception providers, and awareness/wants. Source searches across Python files in
`examples`, `src`, and `eval` found no references to the retired modules or pixel
providers. The full repository suite passed **2,403 tests, with two skipped**,
in 334.47 seconds (exit status 0).

The acceptance boundary is removal of imports and runnable entry points for the
retired path, preservation of measurement/learned-association components, and
continued functioning of structured providers and current workspace tests.
Deleting tests for removed functionality lowers the total; that is not a claim
of increased coverage. No accuracy gain or broader visual capability is claimed
from deleting code.

[Grounded browser actions](71-grounded-browser-actions.md) develops a separate
source-authenticated DOM action path from learned relational grounding. It does
not restore the retired pixel-rule pipeline or claim physical visual clicking.
