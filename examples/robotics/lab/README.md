# Robotics diagnosis lab

The maintained source for the standalone [lab](../../../docs/results/robotics/index.html)
is here. Rebuild rather than editing the generated results page:

```bash
python examples/robotics/lab/build.py
```

The builder validates local evidence/media and graph ports, embeds CSS, JavaScript
and experiment metadata, and derives compact timeline telemetry. The result opens
with `file://`; videos remain relative local files. No server or external assets
are required.

## Experiment contract

Add an entry to `experiments.json` and rebuild. Root fields are `schema_version: 1`
and `experiments`. Each experiment has `id`, `title`, `status` (`pass`, `fail`,
`ref`), `summary`, relative `evidence` path, `graph`, and nonempty `episodes`. Optional `graph_label` identifies which policy a paired
recording diagram describes; optional `outcomes` entries `{label,split,status}`
expose every condition without hiding failures. Alternatively `outcomes_source`
points to an evidence JSON file and `outcome_collections` lists its run arrays
(default `["runs"]`); entries with `id`, `role`, `success` generate that list at
build time. This keeps scores and development/regression labels tied to evidence.

Graph `blocks` maps stable IDs to `{col,row,kind,title,ins,outs,description?}`;
`ins` and `outs` are arrays of named ports. Graph `wires` is an array of pairs
such as `["camera.rgb", "detector.rgb"]`. Kinds are observed (`o`), authored or
estimated (`a`), pretrained/frozen (`f`) and actuation (`x`).

An episode has `id`, `title`, relative `video`, optional `poster`, and optional:

- `signals`: `{time,name,value,unit?,block?}` samples on the video clock.
- `events`: `{time,label,block?}` events on the video clock.
- `traces`: `{path,label,block?,video_start_s?,signal_blocks?}` build-time JSON or gzip JSON records.

Paths are relative to `docs/results/robotics`. Adding another controller or episode
needs data and evidence, not hand-coded HTML. Keep failed episodes in comparisons.

Trace ingestion accepts `t`, `stage`, `desired_hand`, `hand`, `pad_force_n`, and
optional `controller_state` fields. It generates stage events and samples signals
at 5 Hz. Position errors using recorded simulator hand poses are explicitly labeled
**evaluation truth**; these diagnostics are not controller inputs. Pad forces use
actual `sensor_state` measurements where recorded; legacy cube-filtered contact
forces are explicitly marked evaluation truth. Paired videos
name left/right trace sources separately. `video_start_s` subtracts a trace-time
offset for cropped recordings; otherwise trace timestamps must match video time.
Explicit signals/events support different controllers without changing the app.
Unrecorded GR00T internals are not fabricated.

## Diagnosis and notes

Select a graph block to inspect its ports and telemetry at the current video time.
Add a note to attach the time and selected block; click it to seek back. Notes
are scoped to experiment and episode and stored in browser localStorage. Export
downloads versioned JSON; import merges by annotation ID and requires matching
experiment, episode and media. Imported text is never HTML. If local storage is
unavailable, export before leaving. Annotations do not modify recorded evidence.

The warehouse decoration in current recordings has no physics or sensor role.

## Check the interface

```bash
uv run --no-project --with playwright python examples/robotics/lab/check_browser.py
```

The smoke check uses installed Chrome (`--browser` overrides its executable),
opens the built page via `file://`, and checks the current recorded examples,
playback, graph inspection, telemetry, annotations, persistence, import/export
and mobile layout. It writes `docs/results/robotics/browser-check.json` with source
hashes and keeps screenshots/annotation downloads in a temporary directory.

## Recorded task chapters and metrics

The viewport stage strip derives chapters from actual `controller_state.phase`
(or `stage`) changes in the first source trace. It uses that trace's video offset;
clicking a stage seeks to its recorded timestamp, and playback highlights the
current stage. Paired recordings use the first policy's stages, not an invented
merged timeline. Set a trace's `chapters` flag to choose another source, or supply
explicit episode `chapters: [{time,label,block?}]` on the video clock. The source
label is available on each chapter's tooltip. Playback defaults to 1×; explicit
2× and 4× controls change viewing speed without changing source timestamps.

Optional `metrics_source` reads scalar results directly from evidence. Its fields
are `path`, optional `run_id` (selects an entry in `runs`), and `fields`, an array
of `{key,label,unit?,scale?,digits?}`. For example, a measured meter-valued lift can
be shown in centimeters with `scale: 100`. A missing requested field fails the
build instead of substituting an invented value. Metrics should describe the exact
recording shown, not another policy version or a different trial.

A trace’s optional `signal_blocks` maps a raw telemetry field to a graph block ID
(for example, `payload_mass_estimate_kg` to `payload`). Other fields use the
trace’s default block. This keeps estimates and control diagnostics attached to
the component that actually produces them.

A metric field can name `compare_run_id` and optional `compare_label` to display
its value beside the same field from a matched run in the same evidence file.
Use matching units, sampling windows and conditions; the application does not
infer experimental comparability. Both values are read from evidence at build
time, and a tracking improvement must not be presented as a success-rate gap.

Episode `inspection_sources` is an optional list of `{title, path}` entries pointing
to local JSON evidence (scene hierarchy, grounded instructions, task plans or
coordination records). The build embeds their exact parsed contents; expandable
safe-text trees appear below the controller diagram, with source links. These are
recorded evidence inspections, not browser execution of simulator commands.

An experiment may opt into the local language preview with
`language_preview: {command: "<server startup command>"}`. Served over HTTP, the
form posts `{text, mode: "instruction" | "question"}` to
`/api/warehouse/interpret`. It accepts only responses scoped
`recorded_observation_preview`, displays the parser/status/scene revision and
renders the actual response as a safe-text tree. This previews grounding against
the server's fixed recorded observation; it does not execute simulator commands.
The file:// version disables submission and shows the startup command.

Set `multi_stage_tracks: true` on a shared-scene episode to offer one recorded
stage track per trace (for example, r0/r1). Selecting a track changes the chapter
strip; it does not change the recording or its clock. `metrics_source.collection`
selects a named evidence collection instead of the default `runs`, allowing
package-derived paired comparisons without hardcoded summary numbers.

Experiment `metadata` provides the sidebar facets: `task`, `fleet` (humanoid
count), `policy`, and `observation`. Status uses the existing experiment status.
Only supplied metadata produces filter options; filtering never invents a run.
Recorded event arrays remain available for time-local block diagnosis but are not
shown as an unsolicited action log.

For the actual local preview service, start the command displayed for the
warehouse experiment, then run `uv run --no-project --with playwright python
examples/robotics/lab/check_preview.py`. It checks real instruction grounding,
observed-count QA and abstention through HTTP, records non-execution and unchecked
motion feasibility, and writes `preview-browser-check.json`.

An inspection source may supply `image` and `caption` for a collapsed visual
comparison, and `fields` to embed only named top-level JSON fields while linking
the complete evidence file. This keeps large per-observation datasets out of the
standalone HTML. Use `metadata.study` for evidence-study filtering. Label offline
perception comparisons explicitly; they do not change the recorded controller.

For shared-world archives, a trace may specify `robot: "r0"`; the build selects
that robot from each `{t, robots: {...}}` record while keeping the source clock.
A metric field may use `count_where` to count records matching exact evidence
fields (for example, independently validated strict settling), and `compare_path`
to read the same metric from a separate matched reference file. Missing fields
fail the build instead of silently becoming failures or successes.

`metadata` facet values may also be arrays, for example several supporting-study
tags on one recording. `inspection_sources.link` points to a separate local
interactive diagnostic; the build validates that artifact without rewriting it.
A caption keeps historical/offline scope visible when its collapsed item opens.

Repeated-job traces may set `job_labels: true` to prefix recorded phases with the
source `current_job` and preserve transitions even when a new job starts in the
same phase. Nested per-robot/per-job validation uses `count_depth: 2` with
`count_where`; it counts actual matching job records, never a claimed task total.

Presentation excerpts use `video_start_s` and optional exclusive `video_end_s`
(source-clock seconds) on every trace. The builder subtracts the start offset and
omits later telemetry, so chapters cannot seek into events outside the excerpt.
Label the source window and presentation-only scope explicitly in its selector.

A completed-media experiment can opt into `policy_comparison_source`, pointing to
the independent final policy audit. Its collapsed selector/table retains every
policy/condition row. `audited` rows expose actual elapsed time and independent
physical outcomes; `invalid`, `incomplete` and `unrun` stay distinct. Comparative
claims require the report's `complete_and_matched` flag and no contradictory
rows or matching errors. Selecting an audit row does not pretend to change the
video. No pending comparison card or result is published merely by enabling this
source-level capability.

Montages use `timeline_source` on the episode, pointing to renderer metadata
`sequence.chapters` with contiguous `video_start_s`/`video_end_s` and corresponding
`source_start_s`/`source_end_s`. Only 1× segments are admitted. Each cut resets the
phase sample and adds a cut chapter; omitted source intervals never appear in
telemetry. The viewport and annotation clock show both video and source time;
exported notes retain `source_time`, recalculated on import from the bound mapping.
Keep the full continuous recording as the first/default episode.

Native reference comparisons use `native_comparison_source` plus explicit
`native_required_arms`. The builder reads the native packager's episodes/counts
and preserves every completed clip; unfinished required arms block publication
unless the manifest explicitly sets `native_allow_partial`. Arm/condition/config
selectors choose actual clips. A collapsed paired table preserves unavailable
cells and the package's configuration-match/partial-reset caveats. This remains a
separate native tabletop task, never a matched warehouse comparison.

The targeted reset-clearance study uses `placement_comparison_source`. Its
four-row table reads the packaged v5/v6 counts, leaves unmatched baseline cells
as N/A, highlights regressions, and reports whole-episode counts separately from
correlated robot placements. The continuous paired recording retains each
source's actual end time; telemetry stops when that source stops.

Publishing an incomplete policy audit additionally requires
`policy_allow_partial: true`. The viewport displays the audited/expected count
and refusal to admit a complete comparison. The matrix pivots all conditions
against every candidate; unknown cells remain distinct from physical failures.
Pair metrics can use exact `where` / `compare_where` policy-and-condition filters;
zero or multiple matches fail rather than selecting an arbitrary row.
