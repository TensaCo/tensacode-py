# π0 native ALOHA reference

Two fixed-seed executions of Physical Intelligence's released `pi0_aloha_sim`:
**0/2 transfers completed**. Both reached the right-hand lift stage (reward 2),
then timed out at the native 300-step / six-second limit. All footage is retained.
This is a small runtime integration result, not a general π0 performance estimate.

These are **bimanual arms, not a humanoid**, on a different task from TensorCode's
warehouse. There is no matched baseline advantage. The policy receives overhead
RGB, 14 ideal joint/gripper positions, and the fixed “Transfer cube” prompt.
Cube poses/contact identities enter evaluation only.

`validation.json` checks all 600 actions against saved model predictions and all
reported rewards against physical contact traces. Native completion requires left
finger contact with an off-table cube; it does not require sustained holding.
Both videos have 301 frames at 50 fps. Source checkpoint object generations/hashes,
converted weight hash, exact source archives, runtime versions, and resource logs
are included. The historical official simulator dependency lock differs from this
ARM runtime; the native timeout is unchanged and independently checked.

Inference used 32 of 48 GPU multiprocessors via private MPS, four CPU cores, and a
48 GiB client cgroup. MPS GPU allocation is outside that cgroup; global memory was
monitored separately, with a stop threshold of 25% available RAM. The private MPS
server and all experiment jobs were stopped afterward.

Reproduction entry points: `examples/robotics/policies/pi0/README.md` in the Python
repository. The inference runner is also retained in `runner-source.tar.gz`.
