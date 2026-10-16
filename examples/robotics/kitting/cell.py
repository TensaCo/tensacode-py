"""Small Cartesian pick-and-place cell with free objects and contact-only gripping."""

from __future__ import annotations
import numpy as np
import mujoco

COLORS = [".95 .35 .20 1", ".95 .35 .20 1", ".18 .55 .95 1", ".95 .35 .20 1"]
SOURCES = np.array([[-0.30, -0.18], [-0.10, -0.18], [0.10, -0.18], [0.30, -0.18]])
SLOTS = np.array([[-0.23, 0.22], [0, 0.22], [0.23, 0.22]])
REJECT = np.array([0.43, 0.04])


def scene(seed=0):
    rng = np.random.default_rng(seed)
    positions = SOURCES + rng.uniform(-0.012, 0.012, (4, 2))
    objects = "".join(
        f'<body name="part{i}" pos="{x} {y} .425"><freejoint/><geom name="part_geom{i}" type="box" size=".022 .022 .022" mass="{rng.uniform(0.065, 0.095)}" rgba="{COLORS[i]}" friction="1.5 .02 .001"/></body>'
        for i, (x, y) in enumerate(positions)
    )
    markings = "".join(
        f'<geom type="box" pos="{x} {y} .401" size=".055 .055 .001" contype="0" conaffinity="0" rgba=".12 .65 .48 .8"/>'
        for x, y in SLOTS
    )
    return f"""<mujoco model="TensorCode order kitting cell"><option timestep=".004" integrator="implicitfast"/><compiler angle="radian"/>
<visual><global offwidth="960" offheight="720"/><headlight ambient=".4 .4 .4"/></visual>
<default><joint damping="5"/><geom friction="1.5 .02 .001" solref=".008 1"/></default>
<worldbody><light pos="0 -1 3"/><geom type="plane" size="3 3 .1" rgba=".12 .16 .22 1"/>
<geom name="table" type="box" pos="0 0 .2" size=".58 .42 .2" rgba=".25 .30 .37 1"/>{markings}
<geom type="box" pos=".43 .04 .401" size=".055 .055 .001" contype="0" conaffinity="0" rgba=".85 .28 .15 1"/>
<geom type="box" pos="0 .40 .88" size=".56 .025 .025" rgba=".45 .5 .58 1" contype="0" conaffinity="0"/>
<geom type="box" pos="-.55 .40 .64" size=".025 .025 .24" rgba=".45 .5 .58 1" contype="0" conaffinity="0"/>
<geom type="box" pos=".55 .40 .64" size=".025 .025 .24" rgba=".45 .5 .58 1" contype="0" conaffinity="0"/>
<body name="carriage" pos="0 0 .65"><joint name="x" type="slide" axis="1 0 0"/><joint name="y" type="slide" axis="0 1 0"/><joint name="z" type="slide" axis="0 0 1"/>
<geom name="palm" type="box" size=".055 .065 .018" mass=".6" rgba=".15 .60 .75 1"/>
<body pos="0 -.054 -.045"><joint name="left" type="slide" axis="0 1 0" range="0 .043" damping="2"/><geom name="left_pad" type="box" size=".032 .008 .035" mass=".1" rgba=".1 .8 .6 1"/></body>
<body pos="0 .054 -.045"><joint name="right" type="slide" axis="0 -1 0" range="0 .043" damping="2"/><geom name="right_pad" type="box" size=".032 .008 .035" mass=".1" rgba=".1 .8 .6 1"/></body>
</body>{objects}</worldbody>
<actuator><position joint="x" kp="600" kv="55"/><position joint="y" kp="600" kv="55"/><position joint="z" kp="1200" kv="65"/><position joint="left" kp="180" kv="3"/><position joint="right" kp="180" kv="3"/></actuator></mujoco>"""


class Cell:
    def __init__(self, seed=0, record=False):
        self.xml = scene(seed)
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.data = mujoco.MjData(self.model)
        self.record = record
        self.frames = []
        self.events = []
        self.target = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        self.frame_counter = 0
        self.move(np.array([0.0, 0.0, 0.65]), False, 0.4, "settle")

    def position(self, i):
        return self.data.body(f"part{i}").xpos.copy()

    def move(self, xyz, closed, seconds, label):
        goal = np.r_[
            xyz - np.array([0, 0, 0.65]), [0.037, 0.037] if closed else [0.0, 0.0]
        ]
        start = self.target.copy()
        steps = round(seconds / self.model.opt.timestep)
        for k in range(steps):
            t = min(1, (k + 1) / max(1, steps * 0.75))
            smooth = t * t * (3 - 2 * t)
            self.data.ctrl[:] = start + (goal - start) * smooth
            mujoco.mj_step(self.model, self.data)
            self.frame_counter += 1
            if self.record and self.frame_counter % 10 == 0:
                self.frames.append(
                    (self.data.qpos.copy(), float(self.data.time), label)
                )
        self.target = goal

    def transfer(self, item, destination, miss=False, feedback=True):
        source = self.position(item)
        xy = source[:2] + ([0.08, 0] if miss else [0, 0])
        high = 0.65
        low = 0.478
        self.move(np.r_[xy, high], False, 0.7, f"approach part {item}")
        self.move(np.r_[xy, low], False, 0.6, f"descend part {item}")
        self.move(np.r_[xy, low], True, 0.5, f"close gripper / part {item}")
        self.move(np.r_[xy, high], True, 0.7, f"lift / verify part {item}")
        lifted = bool(self.position(item)[2] > 0.52)
        if feedback and not lifted:
            self.move(np.r_[xy, high], False, 0.3, "grasp failed: release and retry")
            return False
        self.move(np.r_[destination, high], True, 1.0, f"transport part {item}")
        self.move(np.r_[destination, low], True, 0.6, f"place part {item}")
        self.move(np.r_[destination, low], False, 0.4, f"release part {item}")
        self.move(np.r_[destination, high], False, 0.5, f"verify placement part {item}")
        return bool(
            np.linalg.norm(self.position(item)[:2] - destination) < 0.045
            and 0.40 < self.position(item)[2] < 0.46
        )
