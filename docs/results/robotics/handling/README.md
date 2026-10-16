# G1 visual pick, transport and place

The diagnosis interface replays a physical MuJoCo episode: G1 with a simulated parallel gripper walks to the
workcell, detects an orange cube and a separate magenta receiving surface from
RGB-D, grasps, transports, detects tactile load transfer, opens and withdraws.
The warehouse scenery is decorative rendering; the robot, payload and support
contacts come from the recorded simulation.

The featured `high-receiver-v5` episode is a development result, not a held-out
benchmark. It walks 0.952 m and completes in 48.04 simulated seconds. Independent
scoring checks a released cube settled on the receiving surface for 4.24 s, a
1.587 mm XY center error and 184 mm hand clearance. A known-color calibrated
workcell and fixed geometric priors are required. This is an authored controller
with a frozen learned walker, not learned visual ICL or a matched VLA comparison.

Tactile forces in the robot's own pad frames feed an estimated-kinematics payload
observer. Its latter-half transport median is 80.15 g for an 80 g environment
payload; no environment mass enters the controller. Estimated load feeds a
Jacobian-transpose torque correction. A sustained measured load drop near the
visually located support permits release without assuming a fixed grasp offset.
True object/receiver poses, counterpart contact names and velocities are used
only for independent scoring.

Every completed development failure and declared check is retained in
`results.json` and `runs/`, with source snapshots and resource logs. Early low
receiver goals were kinematically unreachable with the fixed wrist orientation;
raising the fixture created a reachable workcell. Version 4 physically placed the
cube but timed out at an unnecessarily tight withdrawal waypoint. Version 5 changes
only the withdrawal tolerance to 12 mm; placement's physical criteria are unchanged.
See the matched compensation ablation and declared receiver/payload conditions in
`results.json`; a few deterministic episodes do not establish a general success rate.

Reproduce with `examples/robotics/transfer/README.md`. The shared runner supports
`--mode handling`; the shared evidence packager supports `--task handling`.

The matched 80 g compensation ablation also succeeds. Over the latter half of
transport, median hand-command tracking error falls from 8.106 mm without load
feedforward to 2.230 mm with it (72.5% lower). Both use the same frozen source,
fixture and mass; this is a tracking comparison, not evidence of a success-rate
advantage. Final placement errors are 2.091 mm and 1.587 mm respectively.

| Frozen-controller condition | Result | Placement XY error | Released/settled duration |
| --- | --- | ---: | ---: |
| Development nominal, 80 g | Complete | 1.587 mm | 4.24 s |
| Same case, load compensation disabled | Complete | 2.091 mm | 4.14 s |
| New receiver position and height | Complete | 2.839 mm | 4.12 s |
| Unknown 200 g payload | Complete | 1.409 mm | 4.14 s |

The last two conditions were declared before execution and ran without controller
changes. The 200 g payload observer estimates 200.269 g over the latter half of
transport; its median hand tracking error is 2.257 mm. These are two individual
prospective checks, not broad distributional validation. All five earlier
completed development failures remain visible in the evidence bundle.

All nine host physics jobs used six-core, 12 GiB limits. Resource samples record a
maximum aggregate host CPU utilization of 38.4% and minimum available RAM of
83.4%; GPU computation was disabled. Rendering has separate peer resource logs.
No physics job remains running after this suite.
