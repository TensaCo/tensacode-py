# Shared actuator-cap development diagnostics

Two physical conditions (nominal and validation offset), each with two
co-simulated robots. Authored recovery placed 4/4 objects; each GRU and
diffusion variant placed 0/4. Geometry-model diagnostics are retained in the
adjacent geometry bundle and also placed 0/4. These are development results,
not the five training-excluded final evaluation cases.

Every recorded arm target passed the independent shared .018 radian per
20 ms slew audit. The cap limits commanded targets; it does not certify
forces, collision avoidance or task success. All failures and early fall
stops remain in the probes. Each probe binds its exact runtime source; the
geometry diagnostic used the later common final runtime snapshot. Full
raw traces remain in the campaign evidence directories.
