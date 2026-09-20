# Computer-using agents on the computerworld engine

> **Pixel pipeline retirement:** the fixed geometry/control/prompt semantic path,
> pixel providers, and associated end-to-end/fusion runners described in these
> historical results are retired. Saved measurements are not current runnable
> capabilities. See [70 — Retiring pixel semantic rules](70-retiring-pixel-semantic-rules.md)
> for the retained components and present vision limits.

The desktop agents no longer drive a separate simulator over HTTP. They run inside
[computerworld](https://github.com/JacobFV/computerworld): one deterministic Rust runtime,
embedded in this process through its Python binding. No Chromium, no simulator server, no
network hop.

This replaces the previous arrangement, where a Node simulator held long-lived state and
the agents reached it through a browser page and an HTTP API. The web-page task agents
(access, shop, recon, chart, inbox) still use Playwright, because those are real web pages
rather than a simulated computer.

## What the adapter looks like

Three small layers, all inside the existing perception protocol
(`examples/browser_agents/perception/protocol.py`):

| Piece | File | Job |
|---|---|---|
| `CwProvider` | `perception/computerworld.py` | `env.scene(w, h)` → `PerceivedScene` (elements, text, windows), each item carrying provenance and confidence |
| `CwPixelProvider` | `perception/computerworld.py` | the engine's rendered RGBA → the existing OCR/detector pipeline, so vision and fusion work here too |
| `Surface` + `CwExecutor` + `CwBody` | `perception/cw_body.py` | pointer/keyboard/application actions as engine envelopes; `Browser`'s shape (observe / click / fill / focused / field_text / screenshot) so minds run unchanged |
| `CwWorld` | `worlds/runtime.py` | the harness's handle: actor sessions for agents, privileged reads for scoring, snapshots and `state_hash` |
| world definitions | `worlds/desktop.json`, `worlds/desktop.py` | one Linux machine with a dock, terminal, files and editor; the task's note is an initial file |

A task becomes engine-backed by exporting a `world(seed) -> definition` factory; the
harness then builds the machine itself and never opens a browser (`Task.simulated`,
`harness.body_for`).

### Three deliberate mappings

A provider's job is to map its own vocabulary onto the protocol's, and these are the ones
that let one mind run on a web page, a real desktop and this engine:

1. **Roles.** Engine semantic roles (`heading`, `button`, `textbox`, `text`) map onto the
   protocol's roles.
2. **Sections.** A node inside window *n* is reported in the section named by that window's
   title, the same way the DOM provider reports a window or dialog label. Dock and top-bar
   chrome have no section.
3. **One name.** The terminal's command input is reported as `Shell input`; the engine calls
   it "Terminal command input". The engine's own label stays in the element's provenance.

### The terminal transcript is an efference copy

This terminal prints output but never echoes the command that produced it. So the prompt
lines an agent reads back (`agent@dev:$ ls ~/Desktop`) are the body's record of what it
typed, placed in front of the output that appeared while that command ran. Those blocks
carry `method="efference-copy"` in their provenance; output blocks carry `semantic.v1`.

Two details matter for correctness:

* **Attribution is positional.** Each command owns the lines printed while it ran, so long
  output cannot drift onto the next command.
* **Output is read unwrapped.** The scene draws terminal text hard-wrapped to the window,
  which splits words at the wrap. Joining those visual lines corrupted text: a commit
  message came back as `'initial commi t'`, and one note failed to parse at all. The
  engine's semantic channel carries the same text in logical lines, so that is what a
  reader gets. The wrapped visual lines remain on `last_scene` for anything working from
  pixels — a pixel-based agent here faces the same wrapping a real terminal has.

`clear` is not a program in this shell. Because a terminal's `clear` is a screen
affordance rather than a command, the body implements a typed `clear` by closing and
reopening the terminal window, which genuinely empties it. The shell's working directory
lives on the machine, not the window, so it survives (tested).

### Pixels work here too

The engine rasterizes its own desktop (1280×800 RGBA in 4.1 ms), so the OCR-plus-detector
pipeline runs on this world without a browser or a screen grab. Checked end to end in the
vision environment: on a frame with the terminal open, it read the clock, the dock's
`Terminal` label, the window title and the terminal's own output lines (`readme-first.txt`,
`task-1.txt`) — 18 elements and 11 text blocks. That makes DOM-free fusion measurable here
(`CwProvider` for structure, `CwPixelProvider` for pixels), which is the natural next
comparison for the vision work. `CwPixelProvider.available()` reports False wherever the
OCR package is not installed, rather than pretending.

## Measured: the desktop chore

40 episodes, seeds 1–40, one machine built per episode
(`eval/computerworld_bench.py` → `eval/results/computerworld.json`).

| | Previous simulator (DOM perception) | computerworld |
|---|---|---|
| Items correct | 30/30 | **120/120** |
| Episodes fully correct | 10/10 | **40/40** |
| Seconds per episode | 7.07 (p50) | **0.0158** (mean), 0.0158 p50 |
| Episodes per second | 0.14 | **60.2** |
| Perception per observation | not recorded | **0.41 ms** (p50) |
| Actions per second | — | 2,588 |
| Browser processes | 1 Chromium | none |
| Simulator processes | 1 Node server | none |
| Model calls | 0 | 0 |

The "previous simulator" column is the recorded DOM baseline for the same task and agent
in `eval/results/vision_desktop_e2e_dom.json` (10 episodes, 2026-09-17). It used different
seeds, so read it as an order-of-magnitude comparison of the same work, not a paired trial.
About **450× faster per episode**, and the episode no longer spends its time waiting for a
browser.

### Reproducibility, which the previous simulator could not offer

* **Same seed, same world.** Replaying a seed reaches the same `state_hash`; a different
  seed does not. Checked for two seeds in every bench run.
* **Fresh world per episode.** The note is part of the definition, so nothing an episode
  does can leak into the next one. Two workarounds existed only because the old simulator's
  state outlived the run, and both are now gone: the harness no longer sweeps stale notes
  off the Desktop, and the agent no longer tags its `ls` with a random marker to tell this
  episode's output from an older one. Dropping that marker is what made whole episodes
  reproducible — the agent had been injecting fresh randomness into every run.
* **Checkpoints.** `CwWorld.snapshot/restore/fork` expose the engine's copy-on-write
  checkpoints, which is what long-horizon work needs to retry a step without rebuilding.

## Where the engine's shell does not cover the chat assistant

The desktop chore uses only supported commands, which is why it scores 120/120. The chat
assistant asks for more. Probed directly
(`eval/results/computerworld_shell_gap.json`): **25 of 35** of its command patterns work.

**Works:** `ls -1pA`, `cat`, `head`, `tail`, `printf` with `>`/`>>`, `touch`, `mkdir -p`,
`rm -r`, `mv`, `cp`, `wc -l`, `grep` on a file, `pwd`, `cd` (persistent), `ps`, `uname`,
`hostname`, `whoami`, `date`, and `git init/add/commit/status/log`.

**Missing:** `stat`, `du`, `df`, `which`, `ip`, `nproc`, `uptime`, `sudo`. `grep` has no
recursive mode (it refuses a directory), `2>/dev/null` is unsupported, `git log` ignores
`-n`/`--oneline`, `date` prints the logical tick, and `apt` has no package catalogue in
this world definition.

**One silently wrong result, which matters more than the missing ones:** `find <root> -name
<pattern>` ignores the pattern and lists every file under the root. `find /home/agent -name
'nope'` returned all three files in the world. A search that returns wrong results instead
of failing cannot be detected by its caller, and the assistant's `resolve`, `find` and
`grep` programs all fall back to it.

Consequences observed when running the assistant on this engine:

* "what's on my desktop" answered **"There's no ~/Desktop"** — its file probe is
  `stat -c '%F|%s|%n'`, and a failed probe is read as "does not exist". A confident false
  statement, not an error.
* Resolving a name offered two wrong candidates ("I found 2 things called 'recipes'") because
  `find` returned unrelated files.

The better substitute is not a shell command: the engine exposes `filesystem.v1` with
`read`, `list` and `stat` (`stat` returns `is_dir`, `size`, `owner`, `mode`). `CwWorld`
already uses it for scoring. Whoever redesigns the assistant's programs should decide
whether a file probe is a *screen* action (type `ls` and read the terminal) or a *tool*
action (`filesystem.v1`), because the current shell-parsing approach has no reliable
equivalent here.

Until then the assistant's default engine is unchanged (`--engine seed`), so the running
chatbot keeps working; `--engine computerworld` (or `TENSORCODE_ENGINE=computerworld`) runs
it in-process with the engine's own frames for the viewer.


## Tracked engine gaps (noted, avoided, not worked around)

The owner's instruction is to keep these noted and avoid them for now, and to address
them all at once later, so nothing below is patched piecemeal and no task was made
easier to dodge one. `eval/results/computerworld_gaps.json` is the single list; add rows
there rather than working around a gap locally.

| Gap | Kind | What we do instead |
| --- | --- | --- |
| `find <root> -name <pattern>` ignores the pattern and lists everything | wrong answer | no agent path uses `find` on the engine; lookup is `ls`/`head` on a known path, or the harness's privileged filesystem read (never an agent action) |
| `stat -c`, `du`, `df`, `which`, `ip`, `nproc`, `uptime`, `sudo`, `clear`, non-recursive `grep`, `2>/dev/null`, `ls -la`, `sed -n Np`, `grep -c` absent; `date` is a logical tick | missing | the desktop task uses only the 25/35 working patterns; the chat assistant still defaults to `--engine seed` because its file probes need `stat` |
| the terminal never echoes a command and a bare `$` is not a recognizable prompt | behavioural | the body writes prompt lines itself, marked `method="efference-copy"` so a reader can tell the line is ours |
| the structured scene hard-wraps mid-word | wrong answer | read the engine's logical lines; keep wrapped lines only for pixel work, where the continuation indent makes the join decidable |
| the committed wheel predates the desktop shell | packaging | build from source (`maturin build --release`, ~14 s) and install `0.1.0a1` |


## What still refers to the old simulator

* `examples/browser_agents/assistant/server.py` — Seed remains the **default** engine on
  purpose, because the shell gaps above would break the live chatbot.
* The pixel desktop/computerworld runners and `CwPixelProvider` mentioned in the
  historical comparison are now retired. Saved pixel measurements do not imply
  an available rerun hook; see [70](70-retiring-pixel-semantic-rules.md).

The old simulator's clone is untouched, and its server was left running because the live
chatbot still uses it.

## Engine build note

The wheel in `target/wheels/computerworld-0.1.0-*.whl` predates the engine's desktop shell:
with it, a scene has no `window:*` or `shell:*` interactions and the engine's own
`examples/python/desktop_pixels.py` fails with "incompatible kernel checkpoint version".
Building the current source (`maturin build --release --manifest-path crates/python/Cargo.toml`,
14 s incremental) produces `computerworld-0.1.0a1-cp39-abi3-manylinux_2_34_aarch64.whl`,
which gives the full GNOME-style desktop: a left dock with launchers, window chrome,
resize regions, and `window:<id>:content:terminal-input`. The agents need that build.
