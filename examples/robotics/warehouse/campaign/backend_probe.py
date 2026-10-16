"""Record the actual GL backend and effective cgroup limits in the worker."""

import json
import os
from pathlib import Path

import mujoco
from OpenGL import GL

model = mujoco.MjModel.from_xml_string(
    '<mujoco><worldbody><geom type="box" size=".1 .1 .1"/></worldbody></mujoco>'
)
renderer = mujoco.Renderer(model, height=32, width=32)
renderer.update_scene(mujoco.MjData(model))
renderer.render()
control = Path("/sys/fs/cgroup") / Path("/proc/self/cgroup").read_text().strip().split(
    "::"
)[1].lstrip("/")
print(
    json.dumps(
        {
            "gl_vendor": GL.glGetString(GL.GL_VENDOR).decode(),
            "gl_renderer": GL.glGetString(GL.GL_RENDERER).decode(),
            "gl_version": GL.glGetString(GL.GL_VERSION).decode(),
            "affinity": sorted(os.sched_getaffinity(0)),
            "cgroup": str(control),
            "cgroup_limits": {
                name: (control / name).read_text().strip()
                for name in ["cpu.max", "memory.max", "memory.swap.max"]
            },
        },
        indent=2,
    )
)
renderer.close()
