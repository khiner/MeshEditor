"""Time synchronized draws of Blender’s interactive EEVEE viewport in background mode."""

import json
import struct
import sys
import time
import zlib

import bpy
import gpu
from pathlib import Path
from mathutils import Matrix
import numpy as np

name, scene_file, directory, grid, shadows, *flags = sys.argv[sys.argv.index("--") + 1 :]
scene_path = Path(scene_file)
out = Path(directory)
recovery = "--recovery" in flags
recovery_trial = next((int(f.split("=")[1]) for f in flags if f.startswith("--recovery-trial=")), 0)
camera_log = out / ("recovery-0.jsonl" if recovery else "native-0.jsonl")
poses = [json.loads(line) for line in camera_log.read_text().splitlines()]
assert len(poses) == (104 if recovery else 120)
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)
bpy.ops.import_scene.gltf(filepath=str(scene_path))
scene = bpy.context.scene
scene.render.engine = "BLENDER_EEVEE"
camera_name = "QA Camera" if name == "foliage" else "Camera"
scene.camera = next(
    o for o in scene.objects if o.type == "CAMERA" and (o.name == camera_name or o.data.name == camera_name)
)
scene.render.resolution_percentage = 100
scene.render.film_transparent = True
scene.eevee.use_raytracing = True
scene.eevee.use_taa_reprojection = True
scene.eevee.taa_samples = 64
# The traced-bounce comparison uses EEVEE Screen-Trace across the full roughness range.
scene.eevee.ray_tracing_method = "SCREEN"
scene.eevee.use_fast_gi = name != "diffuse-gi"
scene.eevee.fast_gi_resolution = "2"
scene.eevee.fast_gi_distance = 10.0 if name in ("fast-gi", "atrium") else 0.0
scene.eevee.fast_gi_thickness_near = 0.1
scene.eevee.ray_tracing_options.resolution_scale = "4" if grid == "quarter" else "2"
scene.eevee.shadow_ray_count = int(shadows)
scene.eevee.ray_tracing_options.trace_max_roughness = 0.5
# Match Material Preview: forest HDRI, unit intensity, no punctual lights.
for obj in scene.objects:
    if obj.type == "LIGHT":
        obj.hide_render = True
        obj.hide_viewport = True
world = bpy.data.worlds.new("Matched forest")
world.use_nodes = True
scene.world = world
nodes, links = world.node_tree.nodes, world.node_tree.links
tex = nodes.new("ShaderNodeTexEnvironment")
tex.image = bpy.data.images.load(str(Path("res/images/studiolights/world/forest.hdr").resolve()))
# Blender's -atan2(y,x) azimuth matches native atan2(z,x) after glTF Y-up conversion.
links.new(tex.outputs["Color"], nodes["Background"].inputs["Color"])
nodes["Background"].inputs["Strength"].default_value = 1
if hasattr(world, "sun_threshold"):
    world.sun_threshold = 0
scene.view_settings.view_transform = "Standard"
scene.view_settings.look = "None"
scene.view_settings.exposure = 0
scene.view_settings.gamma = 1
conversion = Matrix(((1, 0, 0, 0), (0, 0, -1, 0), (0, 1, 0, 0), (0, 0, 0, 1)))


def apply_pose(pose):
    scene.render.resolution_x = pose["width"]
    scene.render.resolution_y = pose["height"]
    scene.camera.matrix_world = conversion @ Matrix(pose["cameraToWorld"])
    camera = scene.camera.data
    camera.type = "PERSP"
    camera.sensor_fit = "VERTICAL"
    camera.lens = camera.sensor_height * pose["projection"][1][1] / 2
    # Positive-forward depth coefficients of the native right-handed Metal projection.
    p = pose["projection"]
    camera.clip_start = p[2][3] / p[2][2]
    camera.clip_end = p[2][3] / (p[2][2] + 1) if p[2][2] != -1 else 1e6


def png(path, rgb):
    rgb = np.ascontiguousarray(np.clip(rgb * 255 + 0.5, 0, 255).astype(np.uint8))
    h, w, _ = rgb.shape

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))

    data = b"".join(b"\0" + row.tobytes() for row in rgb)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(data))
        + chunk(b"IEND", b"")
    )


def display_rgba(rgba, path):
    alpha = rgba[:, :, 3:4]
    color = np.maximum(rgba[:, :, :3] / np.maximum(alpha, 1e-8), 0)
    x = color.min(axis=2, keepdims=True)
    color -= np.where(x < 0.08, x - 6.25 * x * x, 0.04)
    peak = color.max(axis=2, keepdims=True)
    new_peak = 1 - 0.24 * 0.24 / np.maximum(peak + 0.24 - 0.76, 1e-8)
    compressed = color * new_peak / np.maximum(peak, 1e-8)
    g = 1 - 1 / (0.15 * (peak - new_peak) + 1)
    color = np.where(peak < 0.76, color, compressed * (1 - g) + new_peak * g)
    display = np.maximum(color, 0) ** (1 / 2.2)
    png(path, display * alpha + 0.25 * (1 - alpha))


# Match glTF thickness to EEVEE's material output for the transmitting surfaces.
for material in bpy.data.materials:
    if not material.use_nodes:
        continue
    for node in material.node_tree.nodes:
        if (
            node.type == "GROUP"
            and node.node_tree
            and node.node_tree.name == "glTF Material Output"
            and "Thickness" in node.inputs
        ):
            output = next(
                n for n in material.node_tree.nodes if n.type == "OUTPUT_MATERIAL" and n.is_active_output
            )
            thickness = node.inputs["Thickness"]
            output.inputs["Thickness"].default_value = thickness.default_value
            if thickness.is_linked:
                material.node_tree.links.new(thickness.links[0].from_socket, output.inputs["Thickness"])
            material.thickness_mode = "SPHERE"

area_cases = ("glass", "mixed", "lights-64", "blocker", "emitter")
if name in area_cases or name.endswith("skins-64"):
    background = nodes["Background"]
    links.remove(background.inputs["Color"].links[0])
    background.inputs["Color"].default_value = (0.05, 0.05, 0.05, 1)
    light = next(o for o in scene.objects if o.type == "LIGHT")
    light.hide_viewport = light.hide_render = False
    if name.endswith("skins-64"):
        light.data.energy = 3
        light.data.angle = 0
    else:
        glass = name in ("glass", "mixed")
        light.data.type = "AREA"
        light.data.shape = "RECTANGLE"
        light.data.size, light.data.size_y = (2.5, 1.5) if glass else (2, 2)
        light.data.energy = 500 if glass else (100 / 64 if name == "lights-64" else 100)
        if not glass:
            background.inputs["Strength"].default_value = 0
        if name == "lights-64":
            light.data.use_custom_distance = True
            light.data.cutoff_distance = 100
            for _ in range(63):
                scene.collection.objects.link(light.copy())
if name == "mixed":
    mat = bpy.data.materials["Clear glass"]
    bsdf = next(n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    bsdf.inputs["Base Color"].default_value = (0.6, 0.9, 1, 1)
    bsdf.inputs["Transmission Weight"].default_value = 0
    bsdf.inputs["Alpha"].default_value = 0.4
    mat.surface_render_method = "BLENDED"
if name == "alpha":
    description = json.loads(scene_path.read_text())
    for material in description["materials"]:
        if material.get("alphaMode") == "BLEND":
            bpy.data.materials[material["name"]].surface_render_method = "BLENDED"
target = None
if name in ("blocker", "emitter"):
    target = light if name == "emitter" else bpy.data.objects["Blocker"]
    bpy.context.view_layer.update()
    origin = target.matrix_world.copy()
scene.render.fps = 24


def animate(index):
    if target is not None:
        matrix = origin.copy()
        matrix.translation.x += -1.2 + 2.4 * ((index + 1) % 48) / 48
        target.matrix_world = matrix
    elif name == "behind-glass":
        # MeshEditor frame one and the importer's frame zero are both glTF time zero.
        frame = (index + 1) % 96
        scene.frame_set(frame)
        expected_x = -2.3 + 4.6 * (1 - abs(frame - 48) / 48)
        assert abs(bpy.data.objects["Moving occluder"].matrix_world.translation.x - expected_x) < 1e-4
    elif name.endswith("skins-64") or name == "atrium":
        scene.frame_set(index + 1)


if name in ("flight-helmet", "sponza"):
    bpy.ops.object.select_all(action="DESELECT")
    meshes = [obj for obj in scene.objects if obj.type == "MESH"]
    for obj in meshes:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_mode(type="FACE")
    bpy.ops.mesh.select_all(action="SELECT")

gpu.init()
area = next(area for area in bpy.context.screen.areas if area.type == "VIEW_3D")
space = area.spaces.active
region = next(region for region in area.regions if region.type == "WINDOW")
space.shading.type = "RENDERED"
space.overlay.show_overlays = False
space.show_gizmo = False
space.region_3d.view_perspective = "PERSP"
space.shading.use_scene_world_render = True
space.shading.use_scene_lights_render = True
apply_pose(poses[0])
space.clip_start = scene.camera.data.clip_start
space.clip_end = scene.camera.data.clip_end
space.lens = scene.camera.data.lens
width, height = poses[0]["width"], poses[0]["height"]


def draw(offscreen, pose, image_render=False):
    bpy.context.view_layer.update()
    view = (conversion @ Matrix(pose["cameraToWorld"])).inverted()
    projection = Matrix(pose["projection"])
    # Blender's draw API accepts OpenGL clip depth; Metal remapping is internal.
    projection[2] = 2 * projection[2] - projection[3]
    offscreen.draw_view3d(
        scene,
        bpy.context.view_layer,
        space,
        region,
        view,
        projection,
        do_color_management=False,
        draw_background=False,
        is_image_render=image_render,
    )
    with offscreen.bind():
        # Synchronize using one pixel, without a full-frame readback in the timed region.
        gpu.state.active_framebuffer_get().read_color(0, 0, 1, 1, 4, 0, "FLOAT")


def capture(offscreen, path):
    with offscreen.bind():
        buffer = gpu.state.active_framebuffer_get().read_color(0, 0, width, height, 4, 0, "FLOAT")
    rgba = np.asarray(buffer, dtype=np.float32).reshape(height, width, 4)[::-1]
    display_rgba(rgba, path)


def warm_viewport():
    viewport = gpu.types.GPUOffScreen(width, height, format="RGBA16F")
    animate(0)
    draw(viewport, poses[0], image_render=True)
    for _ in range(64):
        draw(viewport, poses[0])
    return viewport


trials = []
count = next((int(flag.split("=")[1]) for flag in flags if flag.startswith("--trials=")), 3)
for trial in range(0 if "--images-only" in flags else count):
    viewport = warm_viewport()
    recovery_ms = []
    durations = []
    begin = time.monotonic()
    for index, pose in enumerate(poses):
        start = time.perf_counter()
        animate(index)
        draw(viewport, pose)
        now = time.perf_counter()
        durations.append((now - start) * 1000)
        if recovery and index >= 31:
            if index == 31:
                stopped = now
            recovery_ms.append((now - stopped) * 1000)
    if recovery:
        np.savetxt(out / f"eevee-{recovery_trial}.ms", recovery_ms)
    end = time.monotonic()
    trials.append(dict(start=begin, end=end, p95=float(np.percentile(durations, 95, method="higher"))))
    viewport.free()
if trials:
    (out / ("recovery-eevee-trials.json" if recovery else "eevee-trials.json")).write_text(
        json.dumps(
            dict(version=bpy.app.version_string, revision=bpy.app.build_hash.decode(), trials=trials),
            indent=2,
        )
        + "\n"
    )

if "--skip-image" not in flags:
    viewport = warm_viewport()
    for index, pose in enumerate(poses if recovery else poses[:32]):
        animate(index)
        draw(viewport, pose)
        if recovery and index >= 31:
            capture(viewport, out / f"eevee-{recovery_trial}-{index - 31:03}.png")
    if not recovery:
        capture(viewport, out / "blender.png")
    viewport.free()
