# Shared-scene rendering

`render.py` replays one recorded MuJoCo model and its complete `qpos` vector.
It never combines independent robot episodes, advances physics, or modifies
robot poses. `records.json` timestamps determine playback rate; per-robot phase
badges come from the same trace. Camera parameters are explicit CLI arguments.

`scenery.py` creates original procedural racks, cartons, labels, and pallets.
The simulator adds them before rollout, so they appear in physical collisions
and sensor images. Small label/barcode faces have no collision response.
These fixtures are inventory context; only objects named in the task trace are
claimed as manipulated. No third-party texture or warehouse asset is included.
Robot meshes remain subject to the upstream
[Unitree asset license](../g1/UNITREE-LICENCE) and the pinned provenance in
[the asset loader](../g1/fetch_assets.py); rendering does not redistribute new
copies of them.

`render_still.py` reuses the shared Blender Cycles exporter to path-trace one
recorded state. This is a presentation still, separate from the rasterized
video and the robot's camera input. Material shading and lighting differ from
the simulator's sensor renderer. Source geometry and recorded poses are retained.

Both renderers write source hashes and timing metadata. Run CPU rendering
through `../g1/budget.py`, retain its resource report, and inspect previews
before starting full videos. A preview is not evidence of task completion.

`render_hardware.py` selects NVIDIA EGL without changing GPU or MPS settings.
It paces frame generation independently of playback rate and monitors aggregate
GPU utilization, pausing at 75% until utilization drops below 60%. Use it inside
the CPU/RAM guard. Preserve the guard's raw report: its historical renderer label
is static and may say Mesa even for a hardware child. The video's `gl_backend`
records the actual active driver; the wrapper writes a separate GPU sample log.

## Recorded-state Cycles sequences

`export_cycles_sequence.py` exports static geometry and per-frame transforms
from one complete shared-scene recording. It selects the nearest recorded state
at the requested frame rate and reports the maximum timestamp error. It does not
interpolate motion or generate physics steps. Pass its `geometry.json` and
`poses.npz` to `../transfer/warehouse_photoreal.py` with `--scene` and `--poses`.
`--frame-start` and `--frame-count` permit a bounded benchmark before a full run.
The Blender script retains source hashes, device selection and per-frame timing.

Use `--radiance-dir` to retain linear EXR and RGB PFM frames. The optional
`denoise_cycles_frames.py` invokes a separately installed official
[Open Image Denoise](https://github.com/RenderKit/oidn) executable on CPU with
one or two threads. Retain its executable hash, source release, license and
resource report alongside the experiment. The development environment uses
OIDN 2.5.1 built from its official source release; this repository does not
bundle the binary. `tonemap_cycles_frames.py`, run inside Blender, applies AgX
to the denoised linear PFM frames. `assemble_cycles_video.py` requires every
expected frame, preserves the source clock, and labels the result as Cycles
with OIDN. This is a presentation replay, not the robot's camera input.

CUDA rendering needs a separately coordinated GPU budget. The historical
`g1/budget.py` deliberately masks CUDA visibility, so it must not be used as
proof that a CUDA child has a working GPU allocation. Record the actual device,
MPS allocation where applicable, GPU measurements and CPU/RAM guard separately.
On the current host, Blender's system Python requires `PYTHONHOME=/usr` to load
its NumPy extension consistently. Use absolute script paths inside resource
wrappers that change the working directory. A failed benchmark must not be
reported as a rendered sequence.

For a montage, repeat `--segment START DURATION LABEL` instead of
`--start`/`--duration`. The export retains chapter boundaries and each selected
source timestamp; the assembled video labels its source-time cuts. An optional
Blender `--camera-schedule` JSON list assigns `first_frame`, `frame_count`,
`location` and `lookat` per shot. Camera movement is presentation only. Review
one actual frame from every shot before scheduling the complete render.

Sequence rendering caches decompressed pose arrays and Blender object handles.
It updates only transforms that change in the recording after setting all
transforms on the first rendered frame. Resuming a segment therefore preserves
static and moving geometry exactly; retain each segment's source and resource
reports when a render spans multiple invocations.
