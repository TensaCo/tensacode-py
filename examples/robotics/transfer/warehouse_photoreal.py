"""Path-trace an unchanged recorded humanoid state with Blender Cycles (CPU).

Requires Blender 4.0+ on PATH. Static warehouse props are presentation only.
Example: python warehouse_photoreal.py --run RUN --assets ASSETS --output still.png
For resource limits invoke through g1/budget.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def export(args):
    import mujoco
    import numpy as np
    from warehouse_render import load

    work = args.output.parent / (args.output.stem + "-scene")
    model, data, qpos, records = load(args.run, args.assets, work)
    index = min(range(len(records)), key=lambda i: abs(records[i]["t"] - args.time))
    data.qpos[:] = qpos[index]
    mujoco.mj_forward(model, data)
    geoms = []
    for i in range(model.ngeom):
        if model.geom_group[i] == 3:
            continue
        material = model.geom_matid[i]
        rgba = model.mat_rgba[material] if material >= 0 else model.geom_rgba[i]
        item = {
            "name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or f"geom{i}",
            "type": int(model.geom_type[i]),
            "pos": data.geom_xpos[i].tolist(),
            "rot": data.geom_xmat[i].reshape(3, 3).tolist(),
            "size": model.geom_size[i].tolist(),
            "color": rgba.tolist(),
        }
        if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH:
            mesh = model.geom_dataid[i]
            va, vn = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
            fa, fn = model.mesh_faceadr[mesh], model.mesh_facenum[mesh]
            item.update(
                vertices=model.mesh_vert[va : va + vn].tolist(),
                faces=model.mesh_face[fa : fa + fn].tolist(),
            )
        geoms.append(item)
    path = work / "blender-scene.json"
    path.write_text(json.dumps(geoms))
    subprocess.run(
        [
            args.blender,
            "--background",
            "--factory-startup",
            "--python",
            str(Path(__file__).resolve()),
            "--",
            "--scene",
            str(path.resolve()),
            "--output",
            str(args.output.resolve()),
            "--samples",
            str(args.samples),
            "--threads",
            str(args.threads),
        ],
        check=True,
    )
    args.output.with_suffix(".json").write_text(
        json.dumps(
            {
                "renderer": "Blender Cycles CPU",
                "state_sha256": hashlib.sha256(
                    (args.run / "probe.npz").read_bytes()
                ).hexdigest(),
                "records_sha256": hashlib.sha256(
                    (args.run / "records.json").read_bytes()
                ).hexdigest(),
                "samples": args.samples,
                "recorded_time": records[index]["t"],
                "state_modified": False,
                "visual_only_scenery": True,
                "qpos": np.asarray(qpos[index]).tolist(),
            },
            indent=2,
        )
    )


def path_trace(args):
    import bpy
    from mathutils import Matrix, Vector

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = args.samples
    scene.cycles.use_denoising = False
    scene.render.threads_mode = "FIXED"
    scene.render.threads = args.threads
    scene.render.resolution_x, scene.render.resolution_y = 1280, 720
    scene.render.resolution_percentage = 100
    scene.world.color = (0.18, 0.21, 0.25)
    materials = {}
    for geom in json.loads(args.scene.read_text()):
        color = tuple(
            v / 12.92 if v < 0.04045 else ((v + 0.055) / 1.055) ** 2.4
            for v in geom["color"][:3]
        ) + (geom["color"][3],)
        if color not in materials:
            mat = bpy.data.materials.new(str(color))
            mat.diffuse_color = color
            mat.use_nodes = True
            bs = mat.node_tree.nodes.get("Principled BSDF")
            bs.inputs["Base Color"].default_value = color
            bs.inputs["Roughness"].default_value = 0.48
            if 0.4 < color[0] < 0.5 and abs(color[0] - color[1]) < 0.02:
                bs.inputs["Metallic"].default_value = 0.5
                bs.inputs["Roughness"].default_value = 0.24
            materials[color] = mat
        kind, size = geom["type"], geom["size"]
        if kind == 7:
            mesh = bpy.data.meshes.new(geom["name"])
            mesh.from_pydata(geom["vertices"], [], geom["faces"])
            mesh.update()
            obj = bpy.data.objects.new(geom["name"], mesh)
            bpy.context.collection.objects.link(obj)
            for poly in mesh.polygons:
                poly.use_smooth = True
        elif kind == 6:
            bpy.ops.mesh.primitive_cube_add(size=2)
            obj = bpy.context.object
            obj.scale = size
            bevel = obj.modifiers.new("soft edges", "BEVEL")
            bevel.width, bevel.segments = 0.006, 2
        elif kind == 0:
            bpy.ops.mesh.primitive_plane_add(size=35)
            obj = bpy.context.object
        elif kind == 2:
            bpy.ops.mesh.primitive_uv_sphere_add(
                segments=16, ring_count=8, radius=size[0]
            )
            obj = bpy.context.object
        else:
            continue
        obj.name = geom["name"]
        obj.location = geom["pos"]
        obj.rotation_euler = Matrix(geom["rot"]).to_euler()
        obj.data.materials.append(materials[color])
    for location, power, size, color in [
        ((1, -1, 4), 1000, 4, (1, 0.91, 0.8)),
        ((-3, -2, 3), 700, 3, (0.73, 0.85, 1)),
        ((3, 2, 4), 1200, 3, (1, 1, 1)),
    ]:
        bpy.ops.object.light_add(type="AREA", location=location)
        light = bpy.context.object
        light.data.energy, light.data.size, light.data.color = power, size, color
        light.data.shape = "DISK"
        light.rotation_euler = (
            (Vector((0.6, 0, 0.7)) - light.location).to_track_quat("-Z", "Y").to_euler()
        )
    bpy.ops.object.camera_add(location=(3, -3.8, 2))
    camera = bpy.context.object
    camera.rotation_euler = (
        (Vector((0.65, 0, 0.8)) - camera.location).to_track_quat("-Z", "Y").to_euler()
    )
    camera.data.lens = 48
    scene.camera = camera
    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = str(args.output)
    bpy.ops.render.render(write_still=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--assets", type=Path)
    parser.add_argument("--scene", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--time", type=float, default=19)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--blender", default="blender")
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else sys.argv[1:]
    args = parser.parse_args(argv)
    if not args.scene and not args.run:
        parser.error("--run is required")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    path_trace(args) if args.scene else export(args)


if __name__ == "__main__":
    main()
