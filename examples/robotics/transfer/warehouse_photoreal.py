"""Path-trace an unchanged recorded humanoid state with Blender Cycles (CPU).

Requires Blender 4.0+ on PATH. Static warehouse props are presentation only.
Example: python warehouse_photoreal.py --run RUN --assets ASSETS --output still.png
For resource limits invoke through g1/budget.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def geometry(model, data):
    """Export visible MuJoCo geometry once for stills or recorded sequences."""
    import mujoco

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
    return geoms


def export(args):
    import mujoco
    import numpy as np

    if args.shared_scene:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "warehouse"))
        from render import load
    else:
        from warehouse_render import load

    work = args.output.parent / (args.output.stem + "-scene")
    model, data, qpos, records = load(args.run, args.assets, work)[:4]
    index = min(range(len(records)), key=lambda i: abs(records[i]["t"] - args.time))
    data.qpos[:] = qpos[index]
    mujoco.mj_forward(model, data)
    geoms = geometry(model, data)
    path = work / "blender-scene.json"
    path.write_text(json.dumps(geoms))
    subprocess.run(
        [
            args.blender,
            "--background",
            "--factory-startup",
            "--python-exit-code",
            "1",
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
            "--camera-location",
            *map(str, args.camera_location),
            "--lookat",
            *map(str, args.lookat),
            "--exposure",
            str(args.exposure),
            "--device",
            args.device,
        ]
        + (["--denoise"] if args.denoise else []),
        check=True,
    )
    args.output.with_suffix(".json").write_text(
        json.dumps(
            {
                "renderer": f"Blender Cycles {args.device}",
                "state_sha256": hashlib.sha256(
                    (args.run / "probe.npz").read_bytes()
                ).hexdigest(),
                "records_sha256": hashlib.sha256(
                    (args.run / "records.json").read_bytes()
                ).hexdigest(),
                "samples": args.samples,
                "recorded_time": records[index]["t"],
                "state_modified": False,
                "visual_only_scenery": not args.shared_scene,
                "shared_physics_scene": args.shared_scene,
                "camera_location": args.camera_location,
                "lookat": args.lookat,
                "exposure": args.exposure,
                "scene_sha256": hashlib.sha256(
                    (args.run / "scene.xml").read_bytes()
                ).hexdigest(),
                "presentation_shading": "Cycles materials and lighting; box edges have a 6mm visual bevel; no physics recomputation",
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
    if args.device == "CUDA":
        preferences = bpy.context.preferences.addons["cycles"].preferences
        preferences.compute_device_type = "CUDA"
        devices = preferences.get_devices_for_type("CUDA")
        if not any(device.type == "CUDA" for device in devices):
            raise RuntimeError("No CUDA Cycles device is available")
        for device in devices:
            device.use = device.type == "CUDA"
        scene.cycles.device = "GPU"
    else:
        scene.cycles.device = "CPU"
    scene.cycles.samples = args.samples
    scene.cycles.use_denoising = args.denoise
    scene.render.use_persistent_data = bool(args.poses)
    scene.render.threads_mode = "FIXED"
    scene.render.threads = args.threads
    scene.render.resolution_x, scene.render.resolution_y = 1280, 720
    scene.render.resolution_percentage = 100
    scene.view_settings.exposure = args.exposure
    scene.world.color = (0.18, 0.21, 0.25)
    materials = {}
    geometries = json.loads(args.scene.read_text())
    for geom in geometries:
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
        elif kind == 4:
            bpy.ops.mesh.primitive_uv_sphere_add(segments=24, ring_count=12, radius=1)
            obj = bpy.context.object
            obj.scale = size
        elif kind in (3, 5):
            bpy.ops.mesh.primitive_cylinder_add(
                vertices=32, radius=size[0], depth=2 * size[1]
            )
            obj = bpy.context.object
            if kind == 3:
                parts = [obj]
                for end in (-1, 1):
                    bpy.ops.mesh.primitive_uv_sphere_add(
                        segments=24,
                        ring_count=12,
                        radius=size[0],
                        location=(0, 0, end * size[1]),
                    )
                    parts.append(bpy.context.object)
                bpy.ops.object.select_all(action="DESELECT")
                for part in parts:
                    part.select_set(True)
                bpy.context.view_layer.objects.active = obj
                bpy.ops.object.join()
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
    bpy.ops.object.camera_add(location=args.camera_location)
    camera = bpy.context.object
    camera.rotation_euler = (
        (Vector(args.lookat) - camera.location).to_track_quat("-Z", "Y").to_euler()
    )
    camera.data.lens = 48
    scene.camera = camera
    scene.render.image_settings.file_format = "PNG"

    def retain_radiance(frame):
        if not args.radiance_dir:
            return
        import numpy as np

        args.radiance_dir.mkdir(parents=True, exist_ok=True)
        prefix = args.radiance_dir / f"radiance-{frame:05d}"
        scene.render.image_settings.file_format = "OPEN_EXR"
        scene.render.image_settings.color_depth = "32"
        bpy.data.images["Render Result"].save_render(
            str(prefix.with_suffix(".exr")), scene=scene
        )
        source = bpy.data.images.load(
            str(prefix.with_suffix(".exr")), check_existing=False
        )
        width, height = source.size
        pixels = np.empty(width * height * 4, dtype=np.float32)
        source.pixels.foreach_get(pixels)
        with prefix.with_suffix(".pfm").open("wb") as stream:
            stream.write(f"PF\n{width} {height}\n-1.0\n".encode())
            stream.write(
                pixels.reshape(height, width, 4)[:, :, :3].astype("<f4").tobytes()
            )
        bpy.data.images.remove(source)
        scene.render.image_settings.file_format = "PNG"
        scene.render.image_settings.color_depth = "8"

    frame_times = []
    if args.poses:
        import numpy as np

        # NPZ indexing decompresses arrays: cache once, outside the geometry loop.
        poses = np.load(args.poses)
        positions = poses["positions"]
        rotations = poses["rotations"]
        source_times = poses["source_times"]
        count = len(positions)
        end = (
            min(count, args.frame_start + args.frame_count)
            if args.frame_count
            else count
        )
        if not 0 <= args.frame_start < end:
            raise ValueError("Requested frame range is empty or outside pose sequence")
        bound_objects = [
            (i, bpy.data.objects.get(geom["name"])) for i, geom in enumerate(geometries)
        ]
        changing = np.any(positions[1:] != positions[0], axis=(0, 2)) | np.any(
            rotations[1:] != rotations[0], axis=(0, 2, 3)
        )
        camera_schedule = (
            json.loads(args.camera_schedule.read_text()) if args.camera_schedule else []
        )
        for frame in range(args.frame_start, end):
            for shot in camera_schedule:
                if (
                    shot["first_frame"]
                    <= frame
                    < shot["first_frame"] + shot["frame_count"]
                ):
                    camera.location = shot["location"]
                    camera.rotation_euler = (
                        (Vector(shot["lookat"]) - camera.location)
                        .to_track_quat("-Z", "Y")
                        .to_euler()
                    )
                    break
            transform_started = time.monotonic()
            for i, obj in bound_objects:
                if obj is not None and (frame == args.frame_start or changing[i]):
                    obj.location = tuple(positions[frame, i])
                    obj.rotation_euler = Matrix(rotations[frame, i].tolist()).to_euler()
            print(
                json.dumps(
                    {
                        "transform_frame": frame,
                        "transform_seconds": time.monotonic() - transform_started,
                    }
                ),
                flush=True,
            )
            scene.render.filepath = str(
                args.output.parent / f"{args.output.stem}-{frame:05d}.png"
            )
            started = time.monotonic()
            bpy.ops.render.render(write_still=True)
            retain_radiance(frame)
            frame_times.append(
                {
                    "frame": frame,
                    "source_time_s": float(source_times[frame]),
                    "render_seconds": time.monotonic() - started,
                }
            )
            print(json.dumps(frame_times[-1]), flush=True)
    else:
        scene.render.filepath = str(args.output)
        started = time.monotonic()
        bpy.ops.render.render(write_still=True)
        retain_radiance(0)
        frame_times.append({"render_seconds": time.monotonic() - started})
    args.output.with_name(args.output.stem + "-render.json").write_text(
        json.dumps(
            {
                "renderer": "Blender Cycles",
                "device": args.device,
                "cuda_devices": [device.name for device in devices if device.use]
                if args.device == "CUDA"
                else [],
                "cuda_mps_active_thread_percentage": os.environ.get(
                    "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"
                ),
                "cuda_mps_pipe_directory": os.environ.get("CUDA_MPS_PIPE_DIRECTORY"),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "blender_version": bpy.app.version_string,
                "samples": args.samples,
                "denoising": args.denoise,
                "persistent_scene": bool(args.poses),
                "camera_schedule": (
                    json.loads(args.camera_schedule.read_text())
                    if args.camera_schedule
                    else None
                ),
                "radiance_dir": str(args.radiance_dir) if args.radiance_dir else None,
                "radiance_format": "Scene-linear HDR RGB PFM, bottom row first, float32; EXR retained"
                if args.radiance_dir
                else None,
                "resolution": [1280, 720],
                "camera_location": args.camera_location,
                "lookat": args.lookat,
                "exposure": args.exposure,
                "presentation_shading": "Cycles materials and lighting; box edges have a 6mm visual bevel",
                "frames": frame_times,
                "geometry_sha256": hashlib.sha256(args.scene.read_bytes()).hexdigest(),
                "poses_sha256": hashlib.sha256(args.poses.read_bytes()).hexdigest()
                if args.poses
                else None,
            },
            indent=2,
        )
    )


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
    parser.add_argument("--shared-scene", action="store_true")
    parser.add_argument("--camera-location", type=float, nargs=3, default=[3, -3.8, 2])
    parser.add_argument("--lookat", type=float, nargs=3, default=[0.65, 0, 0.8])
    parser.add_argument("--exposure", type=float, default=0)
    parser.add_argument("--device", choices=["CPU", "CUDA"], default="CPU")
    parser.add_argument("--denoise", action="store_true")
    parser.add_argument("--poses", type=Path)
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-count", type=int, default=0)
    parser.add_argument("--radiance-dir", type=Path)
    parser.add_argument("--camera-schedule", type=Path)
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else sys.argv[1:]
    args = parser.parse_args(argv)
    if not args.scene and not args.run:
        parser.error("--run is required")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    path_trace(args) if args.scene else export(args)


if __name__ == "__main__":
    main()
