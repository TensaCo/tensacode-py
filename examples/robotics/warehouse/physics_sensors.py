"""Per-robot sensor boundary for one shared multi-robot MuJoCo world."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np

_spec = spec_from_file_location(
    "warehouse_transfer_sensors", Path(__file__).parents[1] / "transfer/sensors.py"
)
_module = module_from_spec(_spec)
_spec.loader.exec_module(_module)
SensorBelief = _module.SensorBelief


class SharedSensorBelief(SensorBelief):
    """Keep global physical truth behind rendering and virtual transducers.

    ``prefix`` identifies only the robot's own calibrated joints/geometries.
    ``offset`` is a calibration prior (scalar lane Y or world XYZ translation).
    Local kinematics operate in the shared global frame, using estimated root
    translation. No live floating-base translation or object pose is copied.
    """

    def __init__(self, local_model, shared_model, prefix, offset, **options):
        offset = np.asarray(offset, dtype=float)
        if offset.ndim == 0:
            offset = np.array([0.0, float(offset), 0.0])
        if offset.shape != (3,):
            raise ValueError("offset must be a lane Y scalar or XYZ vector")
        super().__init__(
            local_model, initialposition=np.array([0.0, 0.0, 0.79]) + offset, **options
        )
        self.renderer.close()
        self.renderer = mujoco.Renderer(
            shared_model, height=self.height, width=self.width
        )
        if not np.isclose(local_model.vis.global_.fovy, shared_model.vis.global_.fovy):
            self.renderer.close()
            raise ValueError("Local and shared camera fovy calibration must match")
        self.camera.lookat[:] += offset
        self.shared_model = shared_model
        self.prefix = prefix.rstrip("/") + "/"
        self.proprio = mujoco.MjData(local_model)
        self._shared_root_joint = shared_model.joint(
            self.prefix + "floating_base_joint"
        ).id
        self._shared_root_qadr = int(shared_model.jnt_qposadr[self._shared_root_joint])
        self._shared_root_dadr = int(shared_model.jnt_dofadr[self._shared_root_joint])
        self._local_root_dadr = int(
            local_model.jnt_dofadr[local_model.joint("floating_base_joint").id]
        )
        self._joint_map = []
        for j in range(local_model.njnt):
            if int(local_model.jnt_type[j]) not in (
                int(mujoco.mjtJoint.mjJNT_HINGE),
                int(mujoco.mjtJoint.mjJNT_SLIDE),
            ):
                continue
            name = mujoco.mj_id2name(local_model, mujoco.mjtObj.mjOBJ_JOINT, j)
            shared_j = shared_model.joint(self.prefix + name).id
            self._joint_map.append(
                (
                    int(local_model.jnt_qposadr[j]),
                    int(shared_model.jnt_qposadr[shared_j]),
                    int(local_model.jnt_dofadr[j]),
                    int(shared_model.jnt_dofadr[shared_j]),
                )
            )
        self._shared_floor = shared_model.geom("floor").id
        self._shared_foot_map = {}
        for geoms in self._foot_geoms:
            for local_g in geoms:
                name = mujoco.mj_id2name(local_model, mujoco.mjtObj.mjOBJ_GEOM, local_g)
                self._shared_foot_map[shared_model.geom(self.prefix + name).id] = (
                    local_g
                )
        self._shared_pads = [
            mujoco.mj_name2id(
                shared_model, mujoco.mjtObj.mjOBJ_GEOM, self.prefix + name
            )
            for name in ("pad_a", "pad_b")
        ]
        self._stance_proxy = []
        self._shared_data = None

    def _odometry(self, data):
        # Only this odometer consumes remapped stance contacts. The base class's
        # tactile loop receives an empty list and cannot call shared physics
        # with a local model/contact index.
        super()._odometry(SimpleNamespace(qpos=data.qpos, contact=self._stance_proxy))

    def _camera_measurement(self, _):
        return super()._camera_measurement(self._shared_data)

    def observe(self, shared_data):
        self._shared_data = shared_data
        p = self.proprio
        p.time = float(shared_data.time)
        q, sq = self._root_qadr, self._shared_root_qadr
        d, sd = self._local_root_dadr, self._shared_root_dadr
        p.qpos[q + 3 : q + 7] = shared_data.qpos[sq + 3 : sq + 7]
        p.qvel[d + 3 : d + 6] = shared_data.qvel[sd + 3 : sd + 6]
        for lq, gq, lv, gv in self._joint_map:
            p.qpos[lq] = shared_data.qpos[gq]
            p.qvel[lv] = shared_data.qvel[gv]
        self._stance_proxy = []
        for contact in shared_data.contact:
            pair = (int(contact.geom1), int(contact.geom2))
            if self._shared_floor in pair:
                for geom in pair:
                    if geom in self._shared_foot_map:
                        self._stance_proxy.append(
                            SimpleNamespace(
                                geom1=self._floor, geom2=self._shared_foot_map[geom]
                            )
                        )
        result = super().observe(SimpleNamespace(qpos=p.qpos, time=p.time, contact=[]))
        p.qpos[q : q + 3] = self.root_position
        mujoco.mj_forward(self.model, p)
        norms, local_forces = np.zeros(2), np.zeros((2, 3))
        for index, contact in enumerate(shared_data.contact):
            pair = (int(contact.geom1), int(contact.geom2))
            if not any(pad >= 0 and pad in pair for pad in self._shared_pads):
                continue
            force = np.zeros(6)
            mujoco.mj_contactForce(self.shared_model, shared_data, index, force)
            world = np.asarray(contact.frame).reshape(3, 3).T @ force[:3]
            for i, pad in enumerate(self._shared_pads):
                if pad >= 0 and pad in pair:
                    norms[i] += np.linalg.norm(force[:3])
                    rotation = np.asarray(shared_data.geom_xmat[pad]).reshape(3, 3)
                    local_forces[i] += (
                        rotation.T @ world * (1 if pad == pair[1] else -1)
                    )
        result["pad_force_n"] = norms.tolist()
        result["pad_force_local_n"] = local_forces.tolist()
        return result
