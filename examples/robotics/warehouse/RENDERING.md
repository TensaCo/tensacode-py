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
