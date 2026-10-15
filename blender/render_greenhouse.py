"""Greenhouse model and sprite renderer.

Run from the repo root:
    blender -b --factory-startup --python blender/render_greenhouse.py -- [pass,pass,...]

With no pass list everything is rendered. Pass names: main-ns, main-ew,
shadow-ns, shadow-ew, <crop>-ns, <crop>-ew (all harvest stages of that crop), icon.

Outputs go to graphics/ and the sprite sizes and shifts
are written to prototypes/greenhouse-sprites.lua.

Projection: orthographic camera tilted THETA from vertical, the scene is
stretched along Y by 1/cos(THETA) so ground tiles come out square, and one
tile is TILE_PX pixels (sprites are used at scale 0.5).
"""

import bpy
import bmesh
import math
import os
import random
import sys
from mathutils import Euler, Matrix, Quaternion, Vector

ARGS = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ONLY = set(ARGS[0].split(",")) if ARGS else None
SAMPLES = int(os.environ.get("GH_SAMPLES", "128"))
# "Game look" (default): contact shadows, grime, bevels, material finishes, warm light and a
# colour grade, to look less like a raw render and more like Factorio's own sprites.
GAME_LOOK = os.environ.get("GH_LOOK", "game") != "raw"

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.environ.get("GH_OUT_DIR") or os.path.join(REPO, "graphics")
LUA_OUT = os.path.join(REPO, "prototypes", "greenhouse-sprites.lua")

TILE_PX = 64
SUPERSAMPLE = 2          # render this many times larger, then average down for crisp edges
THETA = math.radians(45)
STRETCH = 1 / math.cos(THETA)
SUN_ELEVATION = math.radians(40)

FW, FL = 5, 7            # footprint (north orientation): width x length in tiles
IW, IL = 4.7, 6.7        # outer size of the model
BASE_H = 0.35            # brick foundation height
WALL_H = 1.75            # eave height
RIDGE_H = 2.75
MARGIN = 0.4

# Harvest stages: 0 = empty beds, 1..STAGES = more and more plants, STAGES = full.
STAGES = 8

CROPS = ["wheat", "sugarcane", "cocoa", "cinnamon", "blueberries", "strawberries",
         "apples", "peaches", "lemons", "nuts", "pumpkin", "wood"]
ORIENTATIONS = {"ns": 0, "ew": 90}


# Scene ------------------------------------------------------------------------

def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = SAMPLES
    # No denoiser (it smears thin bars and stalks); supersampling averages the noise instead.
    scene.cycles.use_denoising = False
    scene.cycles.filter_width = 1.0
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.view_settings.view_transform = "Standard"
    scene.render.resolution_percentage = 100
    # GH_DEVICE=CPU renders on the processor, e.g. when a game is using the graphics memory.
    if os.environ.get("GH_DEVICE", "").upper() == "CPU":
        print("Using CPU")
        # Rebuild the ray-tracing structure from scratch for every render: updating it in
        # place between harvest stages crashes Embree (Blender 5.2) on the CPU.
        try:
            scene.cycles.debug_bvh_type = "STATIC_BVH"
        except (AttributeError, TypeError):
            pass
    else:
        try:
            prefs = bpy.context.preferences.addons["cycles"].preferences
            for device_type in ("OPTIX", "CUDA"):
                try:
                    prefs.compute_device_type = device_type
                    prefs.get_devices()
                    gpus = [d for d in prefs.devices if d.type == device_type]
                    if gpus:
                        for d in prefs.devices:
                            d.use = d.type == device_type
                        scene.cycles.device = "GPU"
                        print("Using", device_type)
                        break
                except TypeError:
                    pass
        except Exception as e:
            print("GPU setup failed, using CPU:", e)

    world = bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = (0.55, 0.65, 0.85, 1) if GAME_LOOK else (0.62, 0.7, 0.82, 1)
    bg.inputs["Strength"].default_value = 0.45

    sun_data = bpy.data.lights.new("Sun", "SUN")
    sun_data.energy = 2.8
    if GAME_LOOK:
        sun_data.color = (1.0, 0.92, 0.8)
    sun_data.angle = math.radians(3)
    sun = bpy.data.objects.new("Sun", sun_data)
    scene.collection.objects.link(sun)
    # Light travels east and slightly south, so shadows fall to the right.
    d = Vector((1, -0.18, 0))
    d.normalize()
    d.z = -math.tan(SUN_ELEVATION)
    sun.rotation_mode = "QUATERNION"
    sun.rotation_quaternion = d.to_track_quat("-Z", "Y")

    cam_data = bpy.data.cameras.new("Camera")
    cam_data.type = "ORTHO"
    cam = bpy.data.objects.new("Camera", cam_data)
    scene.collection.objects.link(cam)
    scene.camera = cam
    return scene


def collection(name):
    c = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(c)
    return c


# Materials ----------------------------------------------------------------------

def principled(m):
    nodes = m.node_tree.nodes
    for n in nodes:
        if n.type == "BSDF_PRINCIPLED":
            return n
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    out = next(n for n in nodes if n.type == "OUTPUT_MATERIAL")
    m.node_tree.links.new(bsdf.outputs[0], out.inputs["Surface"])
    return bsdf


def material(name, color, rough=0.6, alpha=1.0, metallic=0.0, noise=None, specular=None):
    """noise = (second_color, scale[, detail[, (ramp_start, ramp_end)]])."""
    m = bpy.data.materials.new(name)
    try:
        m.use_nodes = True
    except AttributeError:
        pass
    bsdf = principled(m)
    bsdf.inputs["Base Color"].default_value = (*color, 1)
    bsdf.inputs["Roughness"].default_value = rough
    bsdf.inputs["Metallic"].default_value = metallic
    bsdf.inputs["Alpha"].default_value = alpha
    if specular is not None and "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = specular
    if noise:
        nt = m.node_tree
        coords = nt.nodes.new("ShaderNodeTexCoord")
        tex = nt.nodes.new("ShaderNodeTexNoise")
        tex.inputs["Scale"].default_value = noise[1]
        tex.inputs["Detail"].default_value = noise[2] if len(noise) > 2 else 4
        ramp = nt.nodes.new("ShaderNodeValToRGB")
        ramp.color_ramp.elements[0].color = (*color, 1)
        ramp.color_ramp.elements[1].color = (*noise[0], 1)
        lo, hi = noise[3] if len(noise) > 3 else (0.35, 0.65)
        ramp.color_ramp.elements[0].position = lo
        ramp.color_ramp.elements[1].position = hi
        nt.links.new(coords.outputs["Object"], tex.inputs["Vector"])
        nt.links.new(tex.outputs["Fac"], ramp.inputs["Fac"])
        nt.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    return m


def _mix_multiply(nt, a_socket, b_socket):
    mix = nt.nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    mix.blend_type = "MULTIPLY"
    mix.inputs[0].default_value = 1.0
    a = next(i for i in mix.inputs if i.name == "A" and i.type == "RGBA")
    b = next(i for i in mix.inputs if i.name == "B" and i.type == "RGBA")
    nt.links.new(a_socket, a)
    nt.links.new(b_socket, b)
    return next(o for o in mix.outputs if o.type == "RGBA")


def _factor(nt, socket, lo, hi):
    """Remaps a 0..1 value to lo..hi."""
    mr = nt.nodes.new("ShaderNodeMapRange")
    mr.inputs["To Min"].default_value = lo
    mr.inputs["To Max"].default_value = hi
    nt.links.new(socket, mr.inputs["Value"])
    return mr.outputs["Result"]


GRIMY = {"frame", "trim", "brick", "metal", "wood"}
# Contact shadows outline small flat pieces lying on a surface (straw looked like toothpicks).
NO_CONTACT_SHADOW = {"glass", "straw", "straw_pale", "straw_old", "straw_mat"}
BEVELED = {"frame", "trim", "brick", "metal", "wood"}


def _set(mat, **inputs):
    """Sets Principled BSDF inputs by name, skipping any this Blender version lacks."""
    bsdf = principled(mat)
    for name, value in inputs.items():
        socket = bsdf.inputs.get(name.replace("_", " "))
        if socket is not None:
            socket.default_value = value


def _rough_variation(mat, lo, hi, scale=12):
    """Patchy roughness so surfaces don't look like uniform plastic."""
    nt = mat.node_tree
    bsdf = principled(mat)
    coords = nt.nodes.new("ShaderNodeTexCoord")
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = scale
    nt.links.new(coords.outputs["Object"], noise.inputs["Vector"])
    nt.links.new(_factor(nt, noise.outputs["Fac"], lo, hi), bsdf.inputs["Roughness"])


LEAVES = {"leaf", "leaf_dark", "leaf_blue", "leaf_red", "leaf_gloss", "leaf_flush", "leaf_light",
          "leaf_peach", "leaf_peach_dark", "leaf_sage", "cane_leaf", "cane_leaf_light",
          "cane_leaf_old", "pumpkin_leaf", "needles", "needles_light"}
WAXY = {"apple_red", "apple_mixed", "lemon", "lemon_green", "pod", "pod_red", "pod_yellow",
        "pod_green", "pumpkin", "pumpkin_deep", "berry_red", "berry_deep", "berry_light", "berry_unripe"}
VELVET = {"peach_red", "peach_gold"}
BLOOM = {"berry_blue", "berry_blue_dark", "berry_purple", "berry_green"}
MATTE = {"bark", "bark_red", "bark_grey", "bark_pine", "bark_peach", "soil", "floor", "path",
         "straw", "straw_pale", "straw_old", "straw_mat", "litter", "brick", "hull", "hull_dry"}


def _wrinkles(mat, strength=0.25, scale=14):
    nt = mat.node_tree
    bsdf = principled(mat)
    coords = nt.nodes.new("ShaderNodeTexCoord")
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = scale
    noise.inputs["Detail"].default_value = 3
    nt.links.new(coords.outputs["Object"], noise.inputs["Vector"])
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = strength
    nt.links.new(noise.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])


def apply_material_finishes(mats):
    """How each surface reacts to light: translucent leaves, waxy or velvety fruit,
    satin paint, and matte, patchy natural surfaces. No surface patterns."""
    if "plastic" in mats:
        _wrinkles(mats["plastic"])
        _rough_variation(mats["plastic"], 0.25, 0.5, scale=8)
    for key, m in mats.items():
        if key in LEAVES:
            # Thin leaves let some sun through, so lit leaves glow slightly.
            _set(m, Subsurface_Weight=0.25, Subsurface_Scale=0.02, Specular_IOR_Level=0.4)
            _rough_variation(m, 0.35, 0.65)
        elif key in WAXY:
            _set(m, Coat_Weight=0.35, Coat_Roughness=0.15, Subsurface_Weight=0.1, Subsurface_Scale=0.02)
        elif key in VELVET:
            _set(m, Sheen_Weight=0.8, Sheen_Roughness=0.5, Roughness=0.6,
                 Subsurface_Weight=0.15, Subsurface_Scale=0.02)
        elif key in BLOOM:
            # The pale waxy bloom on blueberries.
            _set(m, Sheen_Weight=1.0, Sheen_Tint=(0.75, 0.8, 0.95, 1), Sheen_Roughness=0.4, Roughness=0.55)
        elif key in MATTE:
            _set(m, Specular_IOR_Level=0.3)
            _rough_variation(m, 0.75, 1.0)
        elif key in ("frame", "trim"):
            # Painted metal: satin finish with a thin clear layer.
            _set(m, Coat_Weight=0.25, Coat_Roughness=0.3)
            _rough_variation(m, 0.3, 0.55, scale=6)
        elif key == "wood":
            _set(m, Specular_IOR_Level=0.35)
            _rough_variation(m, 0.6, 0.85)


def apply_game_look(mats):
    """Contact shadows on every material, plus grime near the ground and bevelled edges
    on the building parts."""
    for key, m in mats.items():
        nt = m.node_tree
        bsdf = principled(m)
        base_in = bsdf.inputs["Base Color"]
        if base_in.is_linked:
            color = base_in.links[0].from_socket
        else:
            rgb = nt.nodes.new("ShaderNodeRGB")
            rgb.outputs[0].default_value = base_in.default_value
            color = rgb.outputs[0]
        if key not in NO_CONTACT_SHADOW:
            ao = nt.nodes.new("ShaderNodeAmbientOcclusion")
            ao.samples = 16
            ao.inputs["Distance"].default_value = 0.35
            color = _mix_multiply(nt, color, _factor(nt, ao.outputs["AO"], 0.45, 1.0))
        if key in GRIMY:
            geo = nt.nodes.new("ShaderNodeNewGeometry")
            sep = nt.nodes.new("ShaderNodeSeparateXYZ")
            nt.links.new(geo.outputs["Position"], sep.inputs[0])
            height = nt.nodes.new("ShaderNodeMapRange")
            height.inputs["From Min"].default_value = 0.0
            height.inputs["From Max"].default_value = 0.6 if key == "frame" else 0.9
            nt.links.new(sep.outputs["Z"], height.inputs["Value"])
            noise = nt.nodes.new("ShaderNodeTexNoise")
            noise.inputs["Scale"].default_value = 6
            noise.inputs["Detail"].default_value = 8
            nt.links.new(geo.outputs["Position"], noise.inputs["Vector"])
            dirt = nt.nodes.new("ShaderNodeMath")
            dirt.operation = "MULTIPLY"
            nt.links.new(height.outputs["Result"], dirt.inputs[0])
            nt.links.new(_factor(nt, noise.outputs["Fac"], 0.7, 1.3), dirt.inputs[1])
            grime = nt.nodes.new("ShaderNodeMix")
            grime.data_type = "RGBA"
            nt.links.new(dirt.outputs[0], grime.inputs[0])
            next(i for i in grime.inputs if i.name == "A" and i.type == "RGBA").default_value = (0.3, 0.24, 0.17, 1)
            next(i for i in grime.inputs if i.name == "B" and i.type == "RGBA").default_value = (1, 1, 1, 1)
            color = _mix_multiply(nt, color, next(o for o in grime.outputs if o.type == "RGBA"))
            # Patchy colour variation, so nothing is one flat colour.
            patch = nt.nodes.new("ShaderNodeTexNoise")
            patch.inputs["Scale"].default_value = 2.5
            nt.links.new(geo.outputs["Position"], patch.inputs["Vector"])
            lo = 0.92 if key == "frame" else 0.82
            color = _mix_multiply(nt, color, _factor(nt, patch.outputs["Fac"], lo, 1.05))
        if key in BEVELED:
            bevel = nt.nodes.new("ShaderNodeBevel")
            bevel.inputs["Radius"].default_value = 0.012
            nt.links.new(bevel.outputs["Normal"], bsdf.inputs["Normal"])
        nt.links.new(color, base_in)


def read_pixels(img):
    """An image's pixels as an (h, w, 4) float array, bottom row first. Much faster than
    img.pixels[:], which builds a Python list of every value."""
    import numpy as np
    w, h = img.size
    px = np.empty(w * h * 4, dtype=np.float32)
    img.pixels.foreach_get(px)
    return px.reshape(h, w, 4)


def write_pixels(img, px):
    """Sets an image's pixels from an (h, w, 4) float array, bottom row first."""
    import numpy as np
    img.pixels.foreach_set(np.ascontiguousarray(px, dtype=np.float32).ravel())


def grade(path):
    """Factorio-style colour grade: slightly desaturated, warm, gentle contrast."""
    import numpy as np
    img = bpy.data.images.load(path)
    w, h = img.size
    px = read_pixels(img)
    bpy.data.images.remove(img)
    rgb = px[..., :3]
    lum = (rgb * np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)).sum(-1, keepdims=True)
    rgb = lum + (rgb - lum) * 0.86                       # desaturate
    rgb = rgb * np.array([1.03, 1.0, 0.92], dtype=np.float32)  # warm
    rgb = np.clip(rgb, 0, 1)
    rgb = rgb + 0.12 * (rgb - rgb * rgb) * (rgb - 0.5) * 4  # soft S-curve
    px[..., :3] = np.clip(rgb, 0, 1)
    out = bpy.data.images.new("graded", w, h, alpha=True)
    write_pixels(out, px)
    out.filepath_raw = path
    out.file_format = "PNG"
    out.save()
    bpy.data.images.remove(out)


def make_materials():
    return {
        "frame": material("frame", (0.72, 0.74, 0.7) if GAME_LOOK else (0.62, 0.64, 0.6), rough=0.4),
        "trim": material("trim", (0.22, 0.42, 0.28), rough=0.45),
        "brick": material("brick", (0.52, 0.26, 0.18), rough=0.85, noise=((0.66, 0.38, 0.27), 9)),
        "glass": material("glass", *BUILDING_GLASS[:1], rough=0.08, alpha=BUILDING_GLASS[1],
                          specular=BUILDING_GLASS[2]),
        "metal": material("metal", (0.55, 0.57, 0.6), rough=0.35, metallic=0.8),
        "floor": material("floor", (0.42, 0.34, 0.24), rough=0.95, noise=((0.55, 0.47, 0.34), 6)),
        "path": material("path", (0.6, 0.57, 0.52), rough=0.9, noise=((0.5, 0.48, 0.44), 14)),
        "wood": material("wood", (0.42, 0.27, 0.15), rough=0.8, noise=((0.5, 0.33, 0.19), 5)),
        "soil": material("soil", (0.2, 0.13, 0.08), rough=1.0, noise=((0.28, 0.19, 0.11), 18)),
        "straw": material("straw", (0.78, 0.66, 0.36), rough=0.9, noise=((0.66, 0.53, 0.26), 20)),
        "straw_pale": material("straw_pale", (0.82, 0.72, 0.47), rough=0.85),
        "straw_old": material("straw_old", (0.66, 0.56, 0.36), rough=0.95),
        "straw_mat": material("straw_mat", (0.24, 0.17, 0.1), rough=0.95, noise=((0.72, 0.61, 0.37), 10, 6, (0.5, 0.58))),
        "wheat": material("wheat", (0.86, 0.7, 0.32), rough=0.7),
        "wheat_stalk": material("wheat_stalk", (0.72, 0.66, 0.3), rough=0.7),
        "cane_green": material("cane_green", (0.5, 0.6, 0.22), rough=0.45),
        "cane_yellow": material("cane_yellow", (0.78, 0.7, 0.32), rough=0.45),
        "cane_purple": material("cane_purple", (0.48, 0.2, 0.24), rough=0.45),
        "cane_node": material("cane_node", (0.4, 0.32, 0.16), rough=0.6),
        "cane_wax": material("cane_wax", (0.7, 0.72, 0.6), rough=0.7),
        "cane_leaf": material("cane_leaf", (0.26, 0.5, 0.15), rough=0.55),
        "cane_leaf_light": material("cane_leaf_light", (0.42, 0.62, 0.2), rough=0.55),
        "cane_leaf_old": material("cane_leaf_old", (0.55, 0.58, 0.25), rough=0.6),
        "cane_dead": material("cane_dead", (0.68, 0.57, 0.34), rough=0.8),
        "litter": material("litter", (0.52, 0.42, 0.24), rough=0.95, noise=((0.4, 0.31, 0.17), 22)),
        "leaf": material("leaf", (0.16, 0.42, 0.14), rough=0.7, noise=((0.24, 0.5, 0.18), 6)),
        "leaf_dark": material("leaf_dark", (0.1, 0.3, 0.1), rough=0.7, noise=((0.16, 0.38, 0.14), 6)),
        "leaf_blue": material("leaf_blue", (0.17, 0.38, 0.22), rough=0.7),
        "leaf_red": material("leaf_red", (0.55, 0.22, 0.14), rough=0.7),
        "leaf_gloss": material("leaf_gloss", (0.11, 0.32, 0.11), rough=0.3),
        "leaf_flush": material("leaf_flush", (0.62, 0.34, 0.22), rough=0.4),
        "bark": material("bark", (0.33, 0.22, 0.14), rough=0.9),
        "bark_red": material("bark_red", (0.5, 0.26, 0.15), rough=0.9),
        "pod": material("pod", (0.95, 0.52, 0.08), rough=0.45),
        "pod_red": material("pod_red", (0.62, 0.16, 0.1), rough=0.45),
        "pod_yellow": material("pod_yellow", (0.95, 0.78, 0.15), rough=0.45),
        "pod_green": material("pod_green", (0.62, 0.7, 0.2), rough=0.45),
        "apple_red": material("apple_red", (0.72, 0.06, 0.06), rough=0.3),
        "apple_mixed": material("apple_mixed", (0.8, 0.3, 0.08), rough=0.3, noise=((0.88, 0.7, 0.18), 6)),
        "peach_red": material("peach_red", (0.98, 0.55, 0.12), rough=0.5, noise=((0.8, 0.1, 0.08), 7, 2)),
        "peach_gold": material("peach_gold", (1.0, 0.72, 0.2), rough=0.5, noise=((0.9, 0.3, 0.1), 7, 2)),
        "leaf_peach": material("leaf_peach", (0.24, 0.55, 0.16), rough=0.35),
        "leaf_peach_dark": material("leaf_peach_dark", (0.15, 0.4, 0.12), rough=0.35),
        "bark_peach": material("bark_peach", (0.5, 0.44, 0.38), rough=0.85, noise=((0.38, 0.33, 0.28), 20)),
        "lemon": material("lemon", (0.98, 0.84, 0.12), rough=0.45),
        "lemon_green": material("lemon_green", (0.62, 0.74, 0.18), rough=0.45),
        "hull": material("hull", (0.58, 0.7, 0.32), rough=0.95),
        "hull_dry": material("hull_dry", (0.62, 0.5, 0.32), rough=0.95, noise=((0.5, 0.38, 0.22), 30)),
        "nut_shell": material("nut_shell", (0.8, 0.58, 0.32), rough=0.75, noise=((0.66, 0.44, 0.22), 40)),
        "leaf_sage": material("leaf_sage", (0.33, 0.47, 0.3), rough=0.6),
        "bark_grey": material("bark_grey", (0.45, 0.42, 0.38), rough=0.9, noise=((0.3, 0.27, 0.24), 25, 6)),
        "bark_pine": material("bark_pine", (0.4, 0.27, 0.18), rough=0.9),
        "needles": material("needles", (0.09, 0.26, 0.13), rough=0.8, noise=((0.13, 0.33, 0.16), 25)),
        "needles_light": material("needles_light", (0.16, 0.38, 0.17), rough=0.8, noise=((0.2, 0.44, 0.2), 25)),
        "pumpkin": material("pumpkin", (0.95, 0.48, 0.06), rough=0.5),
        "pumpkin_deep": material("pumpkin_deep", (0.85, 0.36, 0.05), rough=0.5),
        "pumpkin_stem": material("pumpkin_stem", (0.42, 0.4, 0.22), rough=0.8),
        "pumpkin_leaf": material("pumpkin_leaf", (0.22, 0.45, 0.14), rough=0.7, noise=((0.32, 0.52, 0.2), 8)),
        "plastic": material("plastic", (0.025, 0.025, 0.028), rough=0.3, specular=0.6),
        "flower_yellow": material("flower_yellow", (0.98, 0.76, 0.1), rough=0.6),
                "berry_unripe": material("berry_unripe", (0.85, 0.88, 0.6), rough=0.4),
        "flower_center": material("flower_center", (0.95, 0.8, 0.15), rough=0.6),
        "leaf_light": material("leaf_light", (0.28, 0.55, 0.16), rough=0.6),
        "berry_blue": material("berry_blue", (0.3, 0.37, 0.62), rough=0.65),
        "berry_blue_dark": material("berry_blue_dark", (0.13, 0.15, 0.36), rough=0.5),
        "berry_purple": material("berry_purple", (0.45, 0.18, 0.38), rough=0.45),
        "berry_green": material("berry_green", (0.6, 0.72, 0.38), rough=0.45),
        "berry_red": material("berry_red", (0.9, 0.06, 0.06), rough=0.25),
        "berry_deep": material("berry_deep", (0.6, 0.02, 0.04), rough=0.25),
        "berry_light": material("berry_light", (0.96, 0.3, 0.16), rough=0.3),
        "flower": material("flower", (0.95, 0.95, 0.9), rough=0.6),
    }


# Mesh building ----------------------------------------------------------------

SMOOTH = {"sphere", "ico", "cyl", "cone", "ribbon", "pod", "pumpkin", "strawberry"}


def _build_part(bm, kind, matrix):
    """Adds one primitive to bm."""
    if kind == "cube":
        bmesh.ops.create_cube(bm, size=1, matrix=matrix)
    elif kind == "sphere":
        bmesh.ops.create_uvsphere(bm, u_segments=16, v_segments=10, radius=0.5, matrix=matrix)
    elif kind == "ico":
        bmesh.ops.create_icosphere(bm, subdivisions=2, radius=0.5, matrix=matrix)
    elif kind == "cyl":
        bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False, segments=10,
                              radius1=0.5, radius2=0.5, depth=1, matrix=matrix)
    elif kind == "cone":
        bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False, segments=10,
                              radius1=0.5, radius2=0.0, depth=1, matrix=matrix)
    elif kind == "pod":
        # Cocoa pod: rugby-ball shape with ten lengthwise ridges and a pointed tip.
        # Unit pod hangs from the origin down to z = -1, max radius ~0.4.
        rings, around = 12, 20
        top = bm.verts.new(matrix @ Vector((0, 0, 0)))
        tip = bm.verts.new(matrix @ Vector((0, 0, -1.08)))
        grid = []
        for i in range(1, rings + 1):
            t = i / (rings + 1)
            r = 0.4 * math.sin(math.pi * t) ** 0.75 * (1 - 0.25 * t)
            row = []
            for j in range(around):
                th = j / around * 2 * math.pi
                rr = r * (1 + 0.09 * math.cos(10 * th))
                row.append(bm.verts.new(matrix @ Vector((rr * math.cos(th), rr * math.sin(th), -t))))
            grid.append(row)
        for j in range(around):
            k = (j + 1) % around
            bm.faces.new((top, grid[0][k], grid[0][j]))
            bm.faces.new((tip, grid[-1][j], grid[-1][k]))
            for i in range(rings - 1):
                bm.faces.new((grid[i][j], grid[i][k], grid[i + 1][k], grid[i + 1][j]))
    elif kind == "pumpkin":
        # Ribbed, slightly flattened sphere of radius 0.5 centred on the origin.
        rings, around = 12, 24
        top = bm.verts.new(matrix @ Vector((0, 0, 0.36)))
        bottom = bm.verts.new(matrix @ Vector((0, 0, -0.4)))
        grid = []
        for i in range(1, rings + 1):
            phi = math.pi * i / (rings + 1)
            row = []
            for j in range(around):
                th = j / around * 2 * math.pi
                rr = 0.5 * math.sin(phi) * (1 + 0.07 * math.cos(12 * th))
                row.append(bm.verts.new(matrix @ Vector((rr * math.cos(th), rr * math.sin(th),
                                                         0.42 * math.cos(phi)))))
            grid.append(row)
        for j in range(around):
            k = (j + 1) % around
            bm.faces.new((top, grid[0][j], grid[0][k]))
            bm.faces.new((bottom, grid[-1][k], grid[-1][j]))
            for i in range(rings - 1):
                bm.faces.new((grid[i][j], grid[i + 1][j], grid[i + 1][k], grid[i][k]))
    elif kind == "strawberry":
        # Broad shoulders under the calyx, tapering to a rounded tip.
        rings, around = 12, 16
        top = bm.verts.new(matrix @ Vector((0, 0, 0)))
        tip = bm.verts.new(matrix @ Vector((0, 0, -1)))
        grid = []
        for i in range(1, rings + 1):
            t = i / (rings + 1)
            r = 0.5 * (1 - t) ** 0.75 * min(1.0, t * 5) ** 0.5
            row = []
            for j in range(around):
                th = j / around * 2 * math.pi
                row.append(bm.verts.new(matrix @ Vector((r * math.cos(th), r * math.sin(th), -t))))
            grid.append(row)
        for j in range(around):
            k = (j + 1) % around
            bm.faces.new((top, grid[0][k], grid[0][j]))
            bm.faces.new((tip, grid[-1][j], grid[-1][k]))
            for i in range(rings - 1):
                bm.faces.new((grid[i][j], grid[i][k], grid[i + 1][k], grid[i + 1][j]))
    elif kind == "ribbon":
        # Curved strip (leaves): matrix is (points, widths). The strip lies flat
        # sideways, i.e. its width is horizontal and perpendicular to the curve.
        pts, widths = matrix
        n = len(pts)
        rows = []
        for i, p in enumerate(pts):
            t = (pts[min(i + 1, n - 1)] - pts[max(i - 1, 0)]).normalized()
            side = t.cross(Vector((0, 0, 1)))
            if side.length < 1e-4:
                side = t.cross(Vector((1, 0, 0)))
            side.normalize()
            half = side * max(widths[i], 0.002) / 2
            rows.append((bm.verts.new(p - half), bm.verts.new(p + half)))
        for i in range(n - 1):
            bm.faces.new((rows[i][0], rows[i][1], rows[i + 1][1], rows[i + 1][0]))
    elif kind == "tri":
        # Triangular prism: unit triangle in XZ (base -0.5..0.5 at z=0, apex at z=1), thickness 1 in Y.
        vs = []
        for y in (-0.5, 0.5):
            vs.append(bm.verts.new(matrix @ Vector((-0.5, y, 0))))
            vs.append(bm.verts.new(matrix @ Vector((0.5, y, 0))))
            vs.append(bm.verts.new(matrix @ Vector((0, y, 1))))
        a0, b0, c0, a1, b1, c1 = vs
        for f in ((a0, b0, c0), (a1, c1, b1), (a0, a1, b1, b0), (b0, b1, c1, c0), (c0, c1, a1, a0)):
            bm.faces.new(f)


# Primitives that are a fixed shape placed by a matrix. Each is built once with bmesh and
# then copied with numpy: bmesh operators do bookkeeping over the whole mesh on every
# call, so adding thousands of leaves to one bmesh took minutes.
TEMPLATE_KINDS = {"cube", "sphere", "ico", "cyl", "cone", "pod", "pumpkin", "strawberry"}
_templates = {}


def _bm_arrays(bm):
    import numpy as np
    bm.verts.index_update()
    verts = np.array([v.co[:] for v in bm.verts], dtype=np.float64)
    faces = [tuple(v.index for v in f.verts) for f in bm.faces]
    return verts, faces


def _part_arrays(kind, matrix):
    import numpy as np
    if kind in TEMPLATE_KINDS:
        if kind not in _templates:
            bm = bmesh.new()
            _build_part(bm, kind, Matrix.Identity(4))
            _templates[kind] = _bm_arrays(bm)
            bm.free()
        verts, faces = _templates[kind]
        m = np.array(matrix, dtype=np.float64)
        return verts @ m[:3, :3].T + m[:3, 3], faces
    bm = bmesh.new()
    _build_part(bm, kind, matrix)
    out = _bm_arrays(bm)
    bm.free()
    return out


def compose(name, parts, mats):
    """parts: list of (kind, matrix, material_key). Builds one mesh with material slots."""
    import numpy as np
    slot_of = {}
    all_verts, faces, slots, smooth = [], [], [], []
    offset = 0
    for kind, matrix, mat_key in parts:
        verts, part_faces = _part_arrays(kind, matrix)
        slot = slot_of.setdefault(mat_key, len(slot_of))
        faces.extend(tuple(i + offset for i in f) for f in part_faces)
        slots.extend([slot] * len(part_faces))
        smooth.extend([kind in SMOOTH] * len(part_faces))
        all_verts.append(verts)
        offset += len(verts)
    me = bpy.data.meshes.new(name)
    me.from_pydata(np.concatenate(all_verts).tolist(), [], faces)
    me.polygons.foreach_set("material_index", slots)
    me.polygons.foreach_set("use_smooth", smooth)
    me.update()
    for key in sorted(slot_of, key=slot_of.get):
        me.materials.append(mats[key])
    return me


def M(loc=(0, 0, 0), rot=(0, 0, 0), scale=(1, 1, 1)):
    return Matrix.LocRotScale(Vector(loc), Euler(rot).to_quaternion(), Vector(scale))


def between(p1, p2, w, h=None):
    """Matrix that maps a unit cube/cylinder (Z-aligned) onto the segment p1-p2."""
    p1, p2 = Vector(p1), Vector(p2)
    d = p2 - p1
    q = d.to_track_quat("Z", "Y")
    return Matrix.LocRotScale((p1 + p2) / 2, q, Vector((w, h or w, d.length)))


def place(me, coll, parent, loc=(0, 0, 0), rot=(0, 0, 0), scale=(1, 1, 1)):
    o = bpy.data.objects.new(me.name, me)
    coll.objects.link(o)
    o.parent = parent
    o.location = loc
    o.rotation_euler = rot
    o.scale = scale
    return o


# Greenhouse -------------------------------------------------------------------

def greenhouse_parts():
    hw, hl = IW / 2, IL / 2
    frame, glass, floor = [], [], []
    t = 0.18
    # Foundation
    for sx in (-1, 1):
        frame.append(("cube", M((sx * (hw - t / 2), 0, BASE_H / 2), scale=(t, IL, BASE_H)), "brick"))
    for sy in (-1, 1):
        frame.append(("cube", M((0, sy * (hl - t / 2), BASE_H / 2), scale=(IW - 2 * t, t, BASE_H)), "brick"))
    floor.append(("cube", M((0, 0, 0.02), scale=(IW - 2 * t, IL - 2 * t, 0.04)), "floor"))
    floor.append(("cube", M((0, 0, 0.045), scale=(0.7, IL - 2 * t, 0.01)), "path"))

    b = 0.07
    ys = [-hl + i * IL / 7 for i in range(8)]
    pitch = math.atan2(RIDGE_H - WALL_H, hw)
    for y in ys:
        for sx in (-1, 1):
            frame.append(("cube", between((sx * hw, y, BASE_H), (sx * hw, y, WALL_H), b), "frame"))
            frame.append(("cube", between((sx * hw, y, WALL_H), (0, y, RIDGE_H), b), "frame"))
    for sx in (-1, 1):
        for z in (BASE_H, 1.05, WALL_H):
            frame.append(("cube", between((sx * hw, -hl, z), (sx * hw, hl, z), b), "frame"))
        # Purlin halfway up each roof slope
        frame.append(("cube", between((sx * hw / 2, -hl, (WALL_H + RIDGE_H) / 2),
                                      (sx * hw / 2, hl, (WALL_H + RIDGE_H) / 2), b * 0.8), "frame"))
        # Gutter
        frame.append(("cyl", between((sx * (hw + 0.07), -hl - 0.05, WALL_H - 0.03),
                                     (sx * (hw + 0.07), hl + 0.05, WALL_H - 0.03), 0.09), "metal"))
    frame.append(("cube", between((0, -hl - 0.03, RIDGE_H + 0.03), (0, hl + 0.03, RIDGE_H + 0.03), 0.1), "frame"))

    # Gable ends: studs and rails
    for sy in (-1, 1):
        y = sy * hl
        for x in [-hw + i * IW / 5 for i in range(6)]:
            top = WALL_H + (RIDGE_H - WALL_H) * (1 - abs(x) / hw)
            frame.append(("cube", between((x, y, BASE_H), (x, y, top), b), "frame"))
        for z in (BASE_H, 1.05, WALL_H):
            frame.append(("cube", between((-hw, y, z), (hw, y, z), b), "frame"))

    # Doors in the middle of both long sides (rotationally symmetric, so N==S and E==W)
    for sx in (-1, 1):
        x = sx * (hw + 0.02)
        frame.append(("cube", M((x, 0, (BASE_H + 1.55) / 2 - 0.15), scale=(0.08, 0.95, 1.75 - BASE_H)), "trim"))
        frame.append(("cube", M((x + sx * 0.02, 0, 1.15), scale=(0.06, 0.7, 0.55)), "glass"))
        frame.append(("cube", M((x + sx * 0.05, sx * 0.3, 0.8), scale=(0.04, 0.06, 0.06)), "metal"))

    # Glass walls and roof
    wall_h = WALL_H - BASE_H
    for sx in (-1, 1):
        glass.append(("cube", M((sx * hw, 0, BASE_H + wall_h / 2), scale=(0.02, IL, wall_h)), "glass"))
        slope = math.hypot(hw, RIDGE_H - WALL_H)
        # Rotating about Y by +pitch tilts local X down towards +X, so sx * pitch slopes
        # each pane down from the ridge to its own eave.
        glass.append(("cube", M((sx * hw / 2, 0, (WALL_H + RIDGE_H) / 2 + 0.045),
                                rot=(0, sx * pitch, 0), scale=(slope + 0.06, IL + 0.08, 0.02)), "glass"))
    for sy in (-1, 1):
        glass.append(("cube", M((0, sy * hl, BASE_H + wall_h / 2), scale=(IW, 0.02, wall_h)), "glass"))
        glass.append(("tri", M((0, sy * hl, WALL_H), scale=(IW, 0.02, RIDGE_H - WALL_H)), "glass"))

    # Roof vents, propped open (placed symmetrically under 180 degree rotation)
    # Roof vents: each one fills a roof panel exactly, from the ridge down to the purlin and
    # between two rafters, hinged at the ridge and propped open to a shallower angle. Alternate
    # bays on both sides, which is the same pattern after a 180 degree turn (N==S, E==W).
    opened = pitch - math.radians(16)
    vent_len = math.hypot(hw, RIDGE_H - WALL_H) / 2 - b
    for sx in (-1, 1):
        for bay in (1, 3, 5):
            y0, y1 = ys[bay], ys[bay + 1]
            width = (y1 - y0) - b * 1.5
            hinge = Vector((0, (y0 + y1) / 2, RIDGE_H + 0.06))
            d = Vector((sx * math.cos(opened), 0, -math.sin(opened)))
            center = hinge + d * (vent_len / 2)
            rot = (0, sx * opened, 0)
            vent = M(center, rot=rot)
            for dx, dy, w, l in ((vent_len / 2 - b / 2, 0, b, width), (-vent_len / 2 + b / 2, 0, b, width),
                                 (0, width / 2 - b / 2, vent_len, b), (0, -width / 2 + b / 2, vent_len, b)):
                frame.append(("cube", vent @ M((dx, dy, 0), scale=(w, l, 0.05)), "frame"))
            glass.append(("cube", vent @ M(scale=(vent_len - b, width - b, 0.02)), "glass"))
    return frame, glass, floor


# Crops ------------------------------------------------------------------------

BED_X = (-1.2, 1.2)
BED_W = 1.35
BED_L = 6.1


MOUND_H = 0.24       # height of a plastic-covered mound above the floor
MOUND_W = 1.2        # width of a plastic-covered mound


def plastic_mound_parts():
    """Plasticulture beds: long rounded earth mounds wrapped in black plastic film."""
    parts = []
    for x in BED_X:
        # A cylinder along the bed, squashed into a low rounded ridge; the lower half is
        # under the floor.
        parts.append(("cyl", M((x, 0, 0), rot=(math.pi / 2, 0, 0), scale=(MOUND_W, 2 * MOUND_H, BED_L)),
                      "plastic"))
    return parts


def mound_height(dx):
    """Top of a plastic mound at sideways offset dx from its centre line."""
    t = min(1.0, abs(dx) / (MOUND_W / 2))
    return MOUND_H * math.sqrt(1 - t * t)


def bed_parts():
    parts = []
    for x in BED_X:
        parts.append(("cube", M((x, 0, 0.1), scale=(BED_W, BED_L, 0.2)), "wood"))
        parts.append(("cube", M((x, 0, 0.2), scale=(BED_W - 0.1, BED_L - 0.1, 0.02)), "soil"))
    return parts


def wheat_clump(rng):
    parts = []
    for _ in range(9):
        base = Vector((rng.uniform(-0.07, 0.07), rng.uniform(-0.07, 0.07), 0))
        h = rng.uniform(0.6, 0.8)
        tip = base + Vector((rng.uniform(-0.08, 0.08), rng.uniform(-0.08, 0.08), h))
        parts.append(("cyl", between(base, tip, 0.022), "wheat_stalk"))
        parts.append(("sphere", M(tip + Vector((0, 0, 0.05)), scale=(0.05, 0.05, 0.16)), "wheat"))
    return parts


def arching_leaf(start, azimuth, length, width, start_angle, bend, mat, segments=12):
    """A grass-like strap leaf: leaves the stem at start_angle from vertical and bends
    further over by bend (radians) per segment, so long leaves arch over and droop."""
    out = Vector((math.cos(azimuth), math.sin(azimuth), 0))
    up = Vector((0, 0, 1))
    pts, widths = [], []
    p = Vector(start)
    step = length / segments
    for i in range(segments + 1):
        t = i / segments
        pts.append(p.copy())
        # Narrow where it leaves the sheath, widest a third of the way, tapering to a point.
        widths.append(width * min(1.0, 0.45 + t * 2.2) * (1 - t) ** 0.6)
        angle = start_angle + bend * i
        p = p + (out * math.sin(angle) + up * math.cos(angle)) * step
    return ("ribbon", (pts, widths), mat)


def cane_stalk(rng, base, height, lean_az):
    """One cane: jointed internodes with swollen nodes and waxy bands, leaves at the top,
    dried leaves hanging from the lower joints."""
    parts = []
    lean_az += rng.uniform(-0.4, 0.4)       # fan outwards from the middle of the clump
    lean = rng.uniform(0.08, 0.22)
    axis = Vector((math.cos(lean_az) * math.sin(lean), math.sin(lean_az) * math.sin(lean), math.cos(lean)))
    colour = rng.choice(["cane_purple", "cane_yellow", "cane_yellow", "cane_green"])
    n = rng.randint(10, 13)
    seg = height / n
    nodes = []
    p = Vector(base)
    for i in range(n):
        wobble = Vector((rng.uniform(-0.012, 0.012), rng.uniform(-0.012, 0.012), 0))
        q = p + axis * seg + wobble
        d = (q - p).normalized()
        r = 0.058 * (1 - 0.35 * i / n)
        mat = colour if i < n * 0.65 else "cane_green"
        parts.append(("cyl", between(p, q, 2 * r), mat))
        parts.append(("cyl", between(q - d * 0.01, q + d * 0.01, 2 * r * 1.14), "cane_node"))
        parts.append(("cyl", between(q - d * 0.045, q - d * 0.014, 2 * r * 1.04), "cane_wax"))
        nodes.append(q)
        p = q

    # Green leaves from the upper joints, alternating on two sides (two ranks).
    rank = rng.uniform(0, math.pi)
    top_nodes = nodes[-4:]
    for k, node in enumerate(top_nodes):
        side = rank + (math.pi if k % 2 else 0) + rng.uniform(-0.45, 0.45)
        young = k / (len(top_nodes) - 1)       # 0 = lowest (oldest) leaf, 1 = top
        length = rng.uniform(0.45, 0.58) * (0.8 + 0.3 * young)
        mat = rng.choice(["cane_leaf", "cane_leaf", "cane_leaf_light"])
        if k == 0 and rng.random() < 0.5:
            mat = "cane_leaf_old"
        parts.append(arching_leaf(node, side, length, rng.uniform(0.07, 0.085),
                                  start_angle=math.radians(rng.uniform(15, 30) - 8 * young),
                                  bend=math.radians(rng.uniform(5.5, 8)), mat=mat))
    # Spindle: the rolled young leaf pointing straight up from the tip.
    parts.append(("cone", between(p, p + axis * 0.22, 0.035), "cane_leaf_light"))
    # Dried leaves hanging down from the lower joints.
    for node in nodes[2:5]:
        if rng.random() < 0.6:
            parts.append(arching_leaf(node, rng.uniform(0, 2 * math.pi), rng.uniform(0.25, 0.4), 0.06,
                                      start_angle=math.radians(rng.uniform(60, 80)),
                                      bend=math.radians(rng.uniform(14, 20)), mat="cane_dead", segments=8))
    return parts


def cane_clump(rng):
    """A stool of canes growing from one spot, fanning slightly outwards."""
    parts = []
    for _ in range(rng.randint(8, 10)):
        a = rng.uniform(0, 2 * math.pi)
        r = rng.uniform(0.03, 0.16)
        base = Vector((math.cos(a) * r, math.sin(a) * r, 0))
        parts += cane_stalk(rng, base, rng.uniform(0.85, 1.15), lean_az=a)
    return parts


def leaf_crown(rng, center, radii, sprays, leaf_size, leaf_mats, flush_mat=None, flush_share=0.0,
               droop=(0.1, 0.35), bark="bark", core=0):
    """Tree crown made of leafy twigs. Each twig points outwards from the crown centre and
    carries leaves on alternating sides that lie roughly flat, angled forward and drooping
    a little, so from above the crown reads as overlapping leaves rather than a ball."""
    parts = []
    length, width = leaf_size
    if core > 0:   # shady filler so dense crowns have no see-through holes; 0 = no filler
        parts.append(("ico", M(center, scale=radii * core), "leaf_dark"))
    for _ in range(sprays):
        # Directions cover the whole upper crown, top included, so from above it doesn't
        # read as a ring; twigs start near the middle like real shoots off the branches.
        v = Vector((rng.gauss(0, 1), rng.gauss(0, 1), abs(rng.gauss(0, 1.1)) - 0.1)).normalized()
        base = center + Vector((v.x * radii.x, v.y * radii.y, v.z * radii.z)) * rng.uniform(0.1, 0.3)
        tip = center + Vector((v.x * radii.x, v.y * radii.y, v.z * radii.z)) * rng.uniform(0.95, 1.1)
        parts.append(("cyl", between(base, tip, 0.014), bark))
        heading = math.atan2(tip.y - base.y, tip.x - base.x)
        outer = v.z > 0.2
        n = rng.randint(6, 8)
        for k in range(n):
            t = 0.2 + 0.8 * k / (n - 1)
            side = 1 if k % 2 else -1
            yaw = heading + side * rng.uniform(0.5, 0.9)
            pos = base.lerp(tip, t) + Vector((math.cos(yaw), math.sin(yaw), 0)) * length * 0.45
            if flush_mat and outer and t > 0.6 and rng.random() < flush_share:
                mat = flush_mat
            else:
                mat = rng.choice(leaf_mats)
            s = rng.uniform(0.85, 1.1) * (0.8 if t > 0.85 else 1)
            parts.append(("sphere", M(pos, rot=(rng.uniform(-0.25, 0.25), rng.uniform(*droop), yaw),
                                      scale=(length * s, width * s, 0.012)), mat))
        # Leaf at the very tip, pointing along the twig.
        mat = flush_mat if (flush_mat and outer and rng.random() < flush_share) else leaf_mats[0]
        parts.append(("sphere", M(tip + Vector((math.cos(heading), math.sin(heading), 0)) * length * 0.4,
                                  rot=(0, 0.2, heading), scale=(length * 0.8, width * 0.8, 0.012)), mat))
    return parts


def cocoa_tree(rng):
    """Cocoa: bare lower trunk and branches with ridged pods hanging from them on short
    stalks, under a high, flat crown of large drooping leaves."""
    parts = []
    trunk_top = rng.uniform(1.0, 1.1)
    parts.append(("cyl", between((0, 0, 0), (0, 0, trunk_top), 0.13), "bark"))
    branch_ends = []
    for i in range(4):
        a = i / 4 * 2 * math.pi + rng.uniform(-0.4, 0.4)
        start = Vector((0, 0, rng.uniform(0.75, 0.95)))
        end = Vector((math.cos(a) * 0.38, math.sin(a) * 0.38, rng.uniform(1.45, 1.6)))
        parts.append(("cyl", between(start, end, 0.07), "bark"))
        branch_ends.append((start, end))
    parts += leaf_crown(rng, Vector((0, 0, 1.68)), Vector((0.46, 0.46, 0.16)), sprays=34,
                        leaf_size=(0.26, 0.09), leaf_mats=["leaf_gloss", "leaf", "leaf_dark"],
                        droop=(0.25, 0.55))
    # Pods grow straight out of the trunk and thick branches (cauliflory) on short stalks
    # and hang at an angle. Mixed ripeness: green-yellow, yellow, orange, maroon.
    spots = [(Vector((0, 0, rng.uniform(0.18, 0.8))), 0.065) for _ in range(13)]
    for start, end in branch_ends[:2]:
        spots.append((start.lerp(end, rng.uniform(0.2, 0.5)), 0.035))
    for anchor, trunk_r in spots:
        a = rng.uniform(0, 2 * math.pi)
        out = Vector((math.cos(a), math.sin(a), 0))
        stalk_start = anchor + out * trunk_r
        stalk_end = stalk_start + out * 0.04 - Vector((0, 0, 0.02))
        parts.append(("cyl", between(stalk_start, stalk_end, 0.02), "bark"))
        tilt = rng.uniform(0.25, 0.6)
        down = (out * math.sin(tilt) - Vector((0, 0, math.cos(tilt)))).normalized()
        size = rng.uniform(0.25, 0.3)
        mat = rng.choice(["pod", "pod", "pod_yellow", "pod_red", "pod_green"])
        parts.append(("pod", Matrix.LocRotScale(stalk_end, down.to_track_quat("-Z", "Y"),
                                                Vector((size, size, size))), mat))
    return parts


def cinnamon_tree(rng):
    """Evergreen cinnamon tree: reddish-brown trunk, dense rounded crown of glossy leaves
    on leafy twigs, with the pink-red young leaf flush at the outer twig tips."""
    parts = []
    trunk_top = rng.uniform(0.85, 0.95)
    parts.append(("cyl", between((0, 0, 0), (0, 0, trunk_top), 0.12), "bark_red"))
    parts.append(("cyl", between((0, 0, 0), (0, 0, 0.08), 0.17), "bark_red"))   # root flare
    for i in range(5):
        a = i / 5 * 2 * math.pi + rng.uniform(-0.3, 0.3)
        start = Vector((0, 0, rng.uniform(0.6, trunk_top)))
        end = Vector((math.cos(a) * 0.28, math.sin(a) * 0.28, rng.uniform(1.15, 1.35)))
        parts.append(("cyl", between(start, end, 0.055), "bark_red"))
    parts += leaf_crown(rng, Vector((0, 0, 1.3)), Vector((0.42, 0.42, 0.34)), sprays=56,
                        leaf_size=(0.15, 0.065), leaf_mats=["leaf_gloss", "leaf_gloss", "leaf", "leaf_dark"],
                        flush_mat="leaf_flush", flush_share=0.3, droop=(0.05, 0.3), bark="bark_red")
    return parts


def fruit_tree(rng, bark, trunk_h, crown_center, crown_radii, sprays, leaf_size, leaf_mats,
               fruit, fruit_count, droop=(0.1, 0.35), core=0):
    """Generic orchard tree: trunk and branches into a leafy-twig crown, with fruit hung
    around the outside of the crown (mostly its lower half, where the camera sees it)."""
    parts = [("cyl", between((0, 0, 0), (0, 0, trunk_h), 0.12), bark)]
    for i in range(4):
        a = i / 4 * 2 * math.pi + rng.uniform(-0.4, 0.4)
        start = Vector((0, 0, trunk_h * rng.uniform(0.7, 1.0)))
        end = crown_center + Vector((math.cos(a) * crown_radii.x * 0.6, math.sin(a) * crown_radii.y * 0.6,
                                     -crown_radii.z * 0.2))
        parts.append(("cyl", between(start, end, 0.06), bark))
    parts += leaf_crown(rng, crown_center, crown_radii, sprays=sprays, leaf_size=leaf_size,
                        leaf_mats=leaf_mats, droop=droop, bark=bark, core=core)
    for _ in range(fruit_count):
        v = Vector((rng.gauss(0, 1), rng.gauss(0, 1), rng.uniform(-0.7, 0.45))).normalized()
        pos = crown_center + Vector((v.x * crown_radii.x, v.y * crown_radii.y, v.z * crown_radii.z)) \
            * rng.uniform(1.02, 1.12)
        parts += fruit(rng, pos)
    return parts


def apple(rng, pos):
    mat = rng.choice(["apple_red", "apple_red", "apple_mixed"])
    return [("sphere", M(pos, scale=(0.11, 0.11, 0.1)), mat),
            ("cyl", between(pos + Vector((0, 0, 0.04)), pos + Vector((0, 0, 0.08)), 0.012), "bark")]


def peach(rng, pos):
    """Big, saturated peach: yellow-orange with red blush patches, a short stalk and a leaf."""
    r = rng.uniform(0.1, 0.115)
    mat = rng.choice(["peach_red", "peach_red", "peach_gold"])
    yaw = rng.uniform(0, 6.3)
    side = Vector((math.cos(yaw), math.sin(yaw), 0))
    return [
        ("sphere", M(pos, rot=(rng.uniform(-0.2, 0.2), rng.uniform(-0.2, 0.2), yaw),
                     scale=(2 * r, 2 * r * 0.95, 2 * r * 1.03)), mat),
        ("cyl", between(pos + Vector((0, 0, r * 0.8)), pos + Vector((0, 0, r * 1.3)), 0.014), "bark_peach"),
        ("sphere", M(pos + Vector((0, 0, r * 1.15)) + side * 0.06, rot=(0, 0.3, yaw),
                     scale=(0.13, 0.04, 0.012)), "leaf_peach"),
    ]


def lemon(rng, pos):
    mat = "lemon_green" if rng.random() < 0.2 else "lemon"
    tilt = (rng.uniform(-0.4, 0.4), rng.uniform(-0.4, 0.4), 0)
    return [("sphere", M(pos, rot=tilt, scale=(0.09, 0.09, 0.13)), mat),
            ("cone", M(pos, rot=tilt) @ M((0, 0, -0.07), rot=(math.pi, 0, 0), scale=(0.03, 0.03, 0.03)), mat)]


def almond(rng, pos):
    """Almond at harvest: most hulls have dried tan and split into two halves that gape open
    around the nut; a few are still closed and green."""
    yaw = rng.uniform(0, 6.3)
    if rng.random() < 0.25:
        return [("sphere", M(pos, rot=(0, rng.uniform(-0.4, 0.4), yaw), scale=(0.1, 0.075, 0.12)), "hull")]
    side = Vector((math.cos(yaw), math.sin(yaw), 0))
    parts = [("sphere", M(pos, rot=(0.25, 0, yaw), scale=(0.07, 0.05, 0.1)), "nut_shell")]
    for sgn in (-1, 1):
        parts.append(("sphere", M(pos + side * sgn * 0.045, rot=(0, sgn * 0.5, yaw),
                                  scale=(0.045, 0.085, 0.11)), "hull_dry"))
    return parts


def apple_tree(rng):
    return fruit_tree(rng, "bark", 0.8, Vector((0, 0, 1.3)), Vector((0.46, 0.46, 0.36)), 52,
                      (0.15, 0.08), ["leaf", "leaf", "leaf_dark"], apple, 18)


def peach_tree(rng):
    """Dwarf peach: slim, slightly crooked grey-brown trunk, compact rounded crown of long,
    glossy, arching leaves, and big blushing peaches hanging around its lower half."""
    parts = []
    points = [Vector((0, 0, 0))]
    for z in (0.35, 0.62, 0.82):
        points.append(Vector((rng.uniform(-0.05, 0.05), rng.uniform(-0.05, 0.05), z)))
    for i in range(len(points) - 1):
        parts.append(("cyl", between(points[i], points[i + 1], 0.085 - 0.01 * i), "bark_peach"))
    center = Vector((0, 0, 1.2))
    radii = Vector((0.44, 0.44, 0.34))
    for i in range(4):
        a = i / 4 * 2 * math.pi + rng.uniform(-0.4, 0.4)
        end = center + Vector((math.cos(a) * 0.28, math.sin(a) * 0.28, rng.uniform(-0.1, 0.1)))
        parts.append(("cyl", between(points[rng.choice([2, 3])], end, 0.045), "bark_peach"))
    # No filler core: through the long, narrow leaves it showed as a smooth green ball.
    # More twigs fill the crown instead.
    parts += leaf_crown(rng, center, radii, sprays=64, leaf_size=(0.22, 0.045),
                        leaf_mats=["leaf_peach", "leaf_peach", "leaf_peach_dark"], droop=(0.2, 0.45),
                        bark="bark_peach", core=0)
    # Peaches hang on the outside of the crown, mostly its lower half, so they read from above.
    for _ in range(12):
        v = Vector((rng.gauss(0, 1), rng.gauss(0, 1), rng.uniform(-0.6, 0.35))).normalized()
        pos = center + Vector((v.x * radii.x, v.y * radii.y, v.z * radii.z)) * rng.uniform(1.1, 1.22)
        parts += peach(rng, pos - Vector((0, 0, 0.04)))
    return parts


def lemon_tree(rng):
    return fruit_tree(rng, "bark_grey", 0.6, Vector((0, 0, 1.15)), Vector((0.44, 0.44, 0.42)), 58,
                      (0.13, 0.07), ["leaf_gloss", "leaf_gloss", "leaf_dark"], lemon, 16)


def almond_tree(rng):
    return fruit_tree(rng, "bark_grey", 0.75, Vector((0, 0, 1.25)), Vector((0.48, 0.48, 0.34)), 66,
                      (0.15, 0.045), ["leaf_light", "leaf", "leaf_sage"], almond, 28)


def pumpkin_leaf(rng, center, heading):
    """Big palmate pumpkin leaf: five pointed lobes fanned out, slightly cupped."""
    parts = []
    tilt = (rng.uniform(-0.25, 0.25), rng.uniform(-0.25, 0.25))
    for k, spread in enumerate((-1.2, -0.6, 0, 0.6, 1.2)):
        a = heading + spread
        length = 0.15 if k in (1, 2, 3) else 0.11
        c = center + Vector((math.cos(a) * length * 0.5, math.sin(a) * length * 0.5, 0.01 * abs(spread)))
        parts.append(("sphere", M(c, rot=(tilt[0], tilt[1] - 0.15 * abs(spread), a),
                                  scale=(length, length * 0.55, 0.018)), "pumpkin_leaf"))
    return parts


def pumpkin_plant(rng):
    """Sprawling vines with big lobed leaves on stalks, yellow flowers and ribbed pumpkins."""
    parts = []
    for i in range(7):
        a = i / 7 * 2 * math.pi + rng.uniform(-0.4, 0.4)
        p = Vector((0, 0, 0.02))
        for k in range(5):
            a += rng.uniform(-0.4, 0.4)
            q = p + Vector((math.cos(a) * 0.09, math.sin(a) * 0.09, 0))
            parts.append(("cyl", between(p, q, 0.025), "cane_leaf_light"))
            if k % 2 == 0:
                la = a + rng.uniform(-1.2, 1.2)
                leaf_pos = q + Vector((math.cos(la) * 0.06, math.sin(la) * 0.06, rng.uniform(0.16, 0.26)))
                parts.append(("cyl", between(q, leaf_pos, 0.018), "cane_leaf_light"))
                parts += pumpkin_leaf(rng, leaf_pos, la)
            if k == 3 and rng.random() < 0.5:
                parts.append(("cone", M(q + Vector((0, 0, 0.1)), scale=(0.07, 0.07, 0.08)), "flower_yellow"))
            p = q
    for _ in range(rng.choice([1, 2, 2])):
        a = rng.uniform(0, 2 * math.pi)
        d = rng.uniform(0.2, 0.32)
        r = rng.uniform(0.14, 0.18)
        c = Vector((math.cos(a) * d, math.sin(a) * d, r * 0.82))
        parts.append(("pumpkin", M(c, rot=(0, 0, rng.uniform(0, 6.3)), scale=(2 * r, 2 * r, 2 * r)),
                      rng.choice(["pumpkin", "pumpkin", "pumpkin_deep"])))
        parts.append(("cyl", between(c + Vector((0, 0, r * 0.6)), c + Vector((0.02, 0, r * 0.95)), 0.035),
                      "pumpkin_stem"))
    return parts


def conifer(rng):
    """Young timber conifer: straight trunk with whorls of thin, drooping needle branches."""
    parts = []
    h = rng.uniform(1.5, 1.75)
    parts.append(("cyl", between((0, 0, 0), (0, 0, h), 0.08), "bark_pine"))
    tiers = 11
    for t in range(tiers):
        z = 0.25 + (h - 0.3) * t / tiers
        radius = 0.4 * (1 - t / tiers) ** 0.9 + 0.05
        for k in range(12):
            a = k / 12 * 2 * math.pi + t * 0.5 + rng.uniform(-0.15, 0.15)
            base = Vector((0, 0, z + 0.05))
            tip = Vector((math.cos(a) * radius, math.sin(a) * radius, z - radius * 0.3))
            mat = "needles_light" if t >= tiers - 3 else rng.choice(["needles", "needles", "needles_light"])
            parts.append(("cone", between(tip, base, 0.07 + 0.05 * (1 - t / tiers), 0.03), mat))
    parts.append(("cone", between((0, 0, h - 0.1), (0, 0, h + 0.16), 0.07), "needles_light"))
    return parts


def blueberry_bush(rng):
    """Woody stems, small leaves, and bunches of berries hanging on the outside of the bush."""
    parts = []
    tips = []
    for i in range(6):
        a = i / 6 * 2 * math.pi + rng.uniform(-0.3, 0.3)
        base = Vector((math.cos(a) * 0.03, math.sin(a) * 0.03, 0))
        tip = Vector((math.cos(a) * rng.uniform(0.22, 0.3), math.sin(a) * rng.uniform(0.22, 0.3),
                      rng.uniform(0.55, 0.75)))
        parts.append(("cyl", between(base, tip, 0.03), "bark_red"))
        tips.append((base, tip, a))
    # Leaves along the upper part of each stem, plus a dense core so it reads as one bush.
    for base, tip, a in tips:
        for k in range(16):
            t = rng.uniform(0.35, 1.0)
            c = base.lerp(tip, t)
            la = rng.uniform(0, 2 * math.pi)
            c = c + Vector((math.cos(la) * 0.07, math.sin(la) * 0.07, rng.uniform(-0.03, 0.03)))
            key = "leaf_red" if rng.random() < 0.06 else rng.choice(["leaf", "leaf", "leaf_light"])
            parts.append(("sphere", M(c, rot=(rng.uniform(-0.7, 0.7), rng.uniform(-0.7, 0.7), la),
                                      scale=(0.11, 0.06, 0.02)), key))
    parts.append(("ico", M((0, 0, 0.3), scale=(0.3, 0.3, 0.3)), "leaf_dark"))
    # Berry bunches: mostly ripe dusty blue, the odd unripe green or purple one.
    for i in range(9):
        a = i / 9 * 2 * math.pi + rng.uniform(-0.25, 0.25)
        r = rng.uniform(0.24, 0.3)
        c = Vector((math.cos(a) * r, math.sin(a) * r, rng.uniform(0.25, 0.6)))
        parts.append(("cyl", between(c + Vector((0, 0, 0.08)), c, 0.012), "bark_red"))
        for k in range(rng.randint(5, 7)):
            off = Vector((rng.uniform(-0.05, 0.05), rng.uniform(-0.05, 0.05), rng.uniform(-0.06, 0.02)))
            roll = rng.random()
            key = "berry_green" if roll < 0.1 else "berry_purple" if roll < 0.2 else                 rng.choice(["berry_blue", "berry_blue", "berry_blue_dark"])
            size = rng.uniform(0.07, 0.085) if key.startswith("berry_blue") else 0.06
            parts.append(("sphere", M(c + off, scale=(size, size, size * 0.9)), key))
    return parts


def strawberry(rng, top):
    """One berry on its stalk, hanging at a random angle with a star-shaped calyx on top.
    top=True rests it on the leaves (visible from above), otherwise it hangs out at the side."""
    a = rng.uniform(0, 2 * math.pi)
    out = Vector((math.cos(a), math.sin(a), 0))
    r = rng.uniform(0.06, 0.12) if top else rng.uniform(0.18, 0.24)
    calyx = out * r + Vector((0, 0, rng.uniform(0.2, 0.24) if top else rng.uniform(0.08, 0.12)))
    # Tip direction: anywhere from hanging straight down to lying on its side, facing outwards.
    tilt = rng.uniform(math.radians(25), math.radians(100))
    turn = a + rng.uniform(-0.8, 0.8)
    tip_dir = Vector((math.cos(turn) * math.sin(tilt), math.sin(turn) * math.sin(tilt), -math.cos(tilt)))
    frame = Matrix.LocRotScale(calyx, tip_dir.to_track_quat("-Z", "Y"), Vector((1, 1, 1)))
    length = rng.uniform(0.1, 0.12)
    width = length * rng.uniform(0.85, 1.0)
    roll = rng.random()
    mat = ("berry_unripe" if roll < 0.08 else "berry_light" if roll < 0.3 else
           "berry_deep" if roll < 0.62 else "berry_red")
    parts = [
        ("cyl", between(Vector((0, 0, 0.1)) + out * 0.06, calyx + Vector((0, 0, 0.02)), 0.012), "leaf_light"),
        ("strawberry", frame @ M(scale=(width, width, length)), mat),
    ]
    for k in range(5):
        sa = k / 5 * 2 * math.pi + rng.uniform(-0.2, 0.2)
        parts.append(("sphere", frame @ M((math.cos(sa) * 0.025, math.sin(sa) * 0.025, 0.004),
                                         rot=(0, -0.3, sa), scale=(0.045, 0.014, 0.006)), "leaf"))
    return parts


def strawberry_plant(rng):
    """Trifoliate leaves on stalks, berries hanging out onto the straw, a couple of flowers."""
    parts = []
    for i in range(7):
        a = i / 7 * 2 * math.pi + rng.uniform(-0.25, 0.25)
        r = rng.uniform(0.1, 0.15)
        tip = Vector((math.cos(a) * r, math.sin(a) * r, rng.uniform(0.13, 0.18)))
        parts.append(("cyl", between((0, 0, 0.0), tip, 0.018), "leaf_light"))
        for da in (-0.55, 0, 0.55):
            la = a + da
            c = tip + Vector((math.cos(la) * 0.06, math.sin(la) * 0.06, 0.01))
            parts.append(("sphere", M(c, rot=(0, rng.uniform(-0.3, 0.1), la), scale=(0.12, 0.085, 0.022)),
                          "leaf" if rng.random() < 0.6 else "leaf_light"))
    for i in range(5):
        parts += strawberry(rng, top=i < 2)
    for _ in range(2):
        a = rng.uniform(0, 2 * math.pi)
        c = Vector((math.cos(a) * 0.13, math.sin(a) * 0.13, 0.2))
        for k in range(5):
            pa = k / 5 * 2 * math.pi
            parts.append(("sphere", M(c + Vector((math.cos(pa) * 0.03, math.sin(pa) * 0.03, 0)),
                                      scale=(0.035, 0.035, 0.012)), "flower"))
        parts.append(("sphere", M(c + Vector((0, 0, 0.008)), scale=(0.025, 0.025, 0.015)), "flower_center"))
    return parts


# How far plants must stay from the bed ends so wide crops don't poke through the end walls.
END_MARGIN = {"sugarcane": 0.5}

# Width across the bed that the plant rows span (default: most of the bed).
ROW_SPREAD = {"strawberries": 0.56}

CROP_LAYOUT = {
    # crop: (plant builder, rows per bed, spacing along bed, extra bed parts)
    "wheat": (wheat_clump, 5, 0.22, None),
    "sugarcane": (cane_clump, 2, 0.95, "litter"),
    "cocoa": (cocoa_tree, 1, 1.2, None),
    "cinnamon": (cinnamon_tree, 1, 1.2, None),
    "blueberries": (blueberry_bush, 2, 0.62, None),
    "strawberries": (strawberry_plant, 2, 0.36, "plastic"),
    "apples": (apple_tree, 1, 1.2, None),
    "peaches": (peach_tree, 1, 1.2, None),
    "lemons": (lemon_tree, 1, 1.2, None),
    "nuts": (almond_tree, 1, 1.2, None),
    "pumpkin": (pumpkin_plant, 1, 0.85, "loose_straw"),
    "wood": (conifer, 1, 0.9, "litter"),
}


def loose_straw_parts(rng):
    """Straw mulch over the bed soil: a patchy straw layer with soil showing through in
    gaps, and many thin, bent, flat-lying pieces on top for texture."""
    parts = []
    half_w, half_l = (BED_W - 0.16) / 2, (BED_L - 0.16) / 2
    for bx in BED_X:
        # Patchy layer: straw colour where its noise is high, soil showing through elsewhere.
        parts.append(("cube", M((bx, 0, 0.214), scale=(BED_W - 0.12, BED_L - 0.12, 0.012)), "straw_mat"))
        for _ in range(1100):
            x, y = rng.uniform(-half_w, half_w), rng.uniform(-half_l, half_l)
            length = rng.uniform(0.05, 0.12)
            a = rng.uniform(0, math.pi)
            bend = rng.uniform(-0.5, 0.5)
            c = Vector((bx + x, y, 0.2215 + rng.uniform(0, 0.002)))   # lying on the layer
            mat = rng.choice(["straw", "straw", "straw_pale", "straw_old"])
            # Two flat segments meeting at a slight angle: a bent, flattened straw.
            for sgn, ang in ((-1, a), (1, a + bend)):
                d = Vector((math.cos(ang), math.sin(ang), rng.uniform(-0.05, 0.05))) * (length / 2)
                parts.append(("cube", between(c, c + sgn * d, 0.016, 0.004), mat))
    return parts


CLEARANCE = 0.08   # gap kept between plants and the glass


def mesh_vertices(me):
    import numpy as np
    co = np.empty(len(me.vertices) * 3, dtype=np.float32)
    me.vertices.foreach_get("co", co)
    return co.reshape(-1, 3)


def fit_inside(verts, loc, yaw, scale):
    """Largest scale <= scale at which the plant stays under the roof and inside the walls."""
    import numpy as np
    hw, hl = IW / 2 - CLEARANCE, IL / 2 - CLEARANCE
    c, s_ = math.cos(yaw), math.sin(yaw)
    rot = np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1]], dtype=np.float32)
    local = verts @ rot.T
    for _ in range(40):
        p = local * scale + np.array(loc, dtype=np.float32)
        roof = WALL_H + (RIDGE_H - WALL_H) * (1 - np.abs(p[:, 0]) / (IW / 2)) - CLEARANCE
        ok = (np.abs(p[:, 0]) < hw).all() and (np.abs(p[:, 1]) < hl).all() and (p[:, 2] < roof).all()
        if ok:
            return scale
        scale *= 0.97
    return scale


def build_crop(name, mats, coll, parent):
    """Builds beds and plants; returns the plants as (object, stage) with stages spread
    evenly over both beds, so each stage adds plants all over the greenhouse."""
    builder, rows, spacing, mulch = CROP_LAYOUT[name]
    plants = []
    rng = random.Random(name)
    variants = [compose(f"{name}-{i}", builder(rng), mats) for i in range(4)]
    vertex_cache = {me.name: mesh_vertices(me) for me in variants}
    plastic = mulch == "plastic"
    beds = compose(f"{name}-beds", plastic_mound_parts() if plastic else bed_parts(), mats)
    place(beds, coll, parent)
    if mulch == "loose_straw":
        place(compose(f"{name}-mulch", loose_straw_parts(random.Random(name + "-straw")), mats), coll, parent)
    elif mulch and not plastic:
        mulch_me = compose(f"{name}-mulch", [
            ("cube", M((x, 0, 0.215), scale=(BED_W - 0.12, BED_L - 0.12, 0.02)), mulch) for x in BED_X
        ], mats)
        place(mulch_me, coll, parent)
    usable_w = ROW_SPREAD.get(name, BED_W - 0.35)
    n_along = int((BED_L - 0.3) / spacing)
    for bx in BED_X:
        for r in range(rows):
            x = bx if rows == 1 else bx - usable_w / 2 + usable_w * r / (rows - 1)
            for i in range(n_along):
                y = -(n_along - 1) * spacing / 2 + i * spacing + (spacing / 2 if r % 2 else 0) * (rows > 1)
                if abs(y) > BED_L / 2 - END_MARGIN.get(name, 0.15):
                    continue
                me = rng.choice(variants)
                x_jit = x + rng.uniform(-0.03, 0.03)
                z = mound_height(x_jit - bx) - 0.01 if plastic else 0.21
                loc = (x_jit, y + rng.uniform(-0.03, 0.03), z)
                yaw = rng.uniform(0, 2 * math.pi)
                s = fit_inside(vertex_cache[me.name], loc, yaw, rng.uniform(0.9, 1.1))
                plants.append(place(me, coll, parent, loc=loc, rot=(0, 0, yaw), scale=(s, s, s)))
    random.Random(name + "-stages").shuffle(plants)
    return [(o, i * STAGES // len(plants) + 1) for i, o in enumerate(plants)]


# Rendering --------------------------------------------------------------------

# Glass settings: (colour, alpha, specular). Looking at the crops you see through two or
# three panes plus the sky reflected in them, which washes the fruit out to grey, so the
# crop pictures use clearer, less reflective glass than the empty building.
BUILDING_GLASS = ((0.74, 0.92, 0.88), 0.08, 0.35)
CROP_GLASS = ((0.7, 0.88, 0.84), 0.03, 0.12)
if os.environ.get("GH_OLD_CROP_GLASS"):   # for before/after comparisons
    CROP_GLASS = BUILDING_GLASS


def set_glass(mat, settings):
    color, alpha, specular = settings
    bsdf = principled(mat)
    bsdf.inputs["Base Color"].default_value = (*color, 1)
    bsdf.inputs["Alpha"].default_value = alpha
    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = specular


def set_flags(objs, **flags):
    for o in objs:
        for k, v in flags.items():
            setattr(o, k, v)


def frame_camera(cam, scene, umin, umax, vmin, vmax):
    """Point the camera so the image covers screen rect [umin,umax]x[vmin,vmax] (tiles)."""
    w = math.ceil(umax - umin)
    h = math.ceil(vmax - vmin)
    umax, vmax = umin + w, vmin + h
    uc, vc = (umin + umax) / 2, (vmin + vmax) / 2
    scene.render.resolution_x = w * TILE_PX * SUPERSAMPLE
    scene.render.resolution_y = h * TILE_PX * SUPERSAMPLE
    cam.data.ortho_scale = max(w, h)
    up = Vector((0, math.cos(THETA), math.sin(THETA)))
    fwd = Vector((0, math.sin(THETA), -math.cos(THETA)))
    cam.location = Vector((uc, 0, 0)) + up * vc - fwd * 60
    cam.rotation_euler = (THETA, 0, 0)
    cam.data.clip_end = 200
    # Factorio shift: x right, y down
    return {"width": w * TILE_PX, "height": h * TILE_PX, "shift": (uc, -vc)}


def apply_mask(path, mask_path):
    """Multiply the alpha of the image at path by the alpha of the image at mask_path."""
    import numpy as np
    img = bpy.data.images.load(path)
    mask = bpy.data.images.load(mask_path)
    px = read_pixels(img).reshape(-1, 4)
    m = read_pixels(mask).reshape(-1, 4)
    px[:, 3] *= m[:, 3]
    write_pixels(img, px)
    img.filepath_raw = path
    img.file_format = "PNG"
    img.save()
    bpy.data.images.remove(img)
    bpy.data.images.remove(mask)


def downscale(path, factor):
    """Box-filter the PNG at path down by factor, averaging in premultiplied alpha."""
    import numpy as np
    img = bpy.data.images.load(path)
    w, h = img.size
    px = read_pixels(img)
    bpy.data.images.remove(img)
    px[..., :3] *= px[..., 3:4]
    px = px.reshape(h // factor, factor, w // factor, factor, 4).mean(axis=(1, 3))
    a = px[..., 3:4]
    px[..., :3] = np.where(a > 0, px[..., :3] / np.maximum(a, 1e-6), 0)
    out = bpy.data.images.new("downscaled", w // factor, h // factor, alpha=True)
    write_pixels(out, px)
    out.filepath_raw = path
    out.file_format = "PNG"
    out.save()
    bpy.data.images.remove(out)


def render(scene, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    scene.render.filepath = path
    bpy.ops.render.render(write_still=True)
    if SUPERSAMPLE > 1:
        downscale(path, SUPERSAMPLE)
    if GAME_LOOK and not path.endswith("shadow.png") and not os.path.basename(path).startswith("_"):
        grade(path)
    print("Wrote", path)


def wanted(name):
    return ONLY is None or name in ONLY


def main():
    scene = reset_scene()
    mats = make_materials()
    if GAME_LOOK:
        apply_material_finishes(mats)
        apply_game_look(mats)
    cam = scene.camera

    root = bpy.data.objects.new("ROOT", None)
    scene.collection.objects.link(root)
    model = bpy.data.objects.new("MODEL", None)
    scene.collection.objects.link(model)
    model.parent = root

    c_frame, c_glass, c_floor = collection("frame"), collection("glass"), collection("floor")
    frame_parts, glass_parts, floor_parts = greenhouse_parts()
    frame_obj = place(compose("frame", frame_parts, mats), c_frame, model)
    glass_obj = place(compose("glass", glass_parts, mats), c_glass, model)
    building = [frame_obj, glass_obj]
    floor = place(compose("floor", floor_parts, mats), c_floor, model)

    crop_colls = {}
    crop_plants = {}
    for crop in CROPS:
        c = collection("crop-" + crop)
        crop_plants[crop] = build_crop(crop, mats, c, model)
        c.hide_render = True
        crop_colls[crop] = c

    ground_me = compose("ground", [("cube", M((0, 0, -0.01), scale=(40, 40, 0.02)), "floor")], mats)
    ground = place(ground_me, scene.collection, root)
    ground.is_shadow_catcher = True
    ground.hide_render = True

    sprites = {}
    root.scale = (1, STRETCH, 1)
    for orient, angle in ORIENTATIONS.items():
        model.rotation_euler = (0, 0, math.radians(angle))
        fw, fl = (FW, FL) if angle == 0 else (FL, FW)
        top = RIDGE_H * math.sin(THETA) + 0.25
        main_rect = (-fw / 2 - MARGIN, fw / 2 + MARGIN, -fl / 2 - MARGIN, fl / 2 + top + MARGIN)

        name = "main-" + orient
        sprites[name] = frame_camera(cam, scene, *main_rect)
        if wanted(name):
            render(scene, os.path.join(OUT_DIR, f"greenhouse-{orient}.png"))

        name = "shadow-" + orient
        shadow_len = RIDGE_H / math.tan(SUN_ELEVATION) + 0.3
        sprites[name] = frame_camera(cam, scene, -fw / 2 - MARGIN, fw / 2 + MARGIN + shadow_len,
                                     -fl / 2 - MARGIN - 0.8, fl / 2 + MARGIN)
        if wanted(name):
            set_flags(building + [floor], visible_camera=False)
            ground.hide_render = False
            render(scene, os.path.join(OUT_DIR, f"greenhouse-{orient}-shadow.png"))
            ground.hide_render = True
            set_flags(building + [floor], visible_camera=True)

        for crop in CROPS:
            for stage in range(STAGES + 1):
                sprites[f"{crop}-{orient}-{stage}"] = frame_camera(cam, scene, *main_rect)
            if not wanted(f"{crop}-{orient}"):
                continue
            # Crops plus the floor under them, seen through the glass and lit exactly like
            # the building sprite. The frame cuts them out where it is in front but is not
            # drawn itself (the building sprite has it).
            frame_obj.is_holdout = True
            crop_colls[crop].hide_render = False
            set_glass(mats["glass"], CROP_GLASS)
            # Mask: the full crop without glass, i.e. where floor or plants are behind the
            # glass. Glass over open background is left to the building sprite so it is not
            # drawn twice. One mask per crop is shared by all stages to keep renders down.
            mask = os.path.join(OUT_DIR, "_mask.png")
            glass_obj.visible_camera = False
            render(scene, mask)
            glass_obj.visible_camera = True
            for stage in range(STAGES + 1):
                for obj, plant_stage in crop_plants[crop]:
                    obj.hide_render = plant_stage > stage
                out = os.path.join(OUT_DIR, f"greenhouse-{crop}-{orient}-{stage}.png")
                render(scene, out)
                apply_mask(out, mask)
            for obj, _ in crop_plants[crop]:
                obj.hide_render = False
            os.remove(mask)
            crop_colls[crop].hide_render = True
            frame_obj.is_holdout = False
            set_glass(mats["glass"], BUILDING_GLASS)

    if wanted("icon"):
        root.scale = (1, 1, 1)
        model.rotation_euler = (0, 0, math.radians(-35))
        crop_colls["wheat"].hide_render = False
        cam.rotation_euler = (math.radians(58), 0, 0)
        cam.data.ortho_scale = 8.0
        d = Euler(cam.rotation_euler).to_matrix() @ Vector((0, 0, 1))
        cam.location = Vector((0, 0, 1.0)) + d * 60
        scene.render.resolution_x = scene.render.resolution_y = 64 * SUPERSAMPLE
        render(scene, os.path.join(REPO, "graphics", "greenhouse-icon.png"))
        crop_colls["wheat"].hide_render = True

    write_lua(sprites)


def write_lua(sprites):
    lines = [
        "-- Generated by blender/render_greenhouse.py. Do not edit by hand; re-run the script instead.",
        "return {",
    ]
    for name in sorted(sprites):
        s = sprites[name]
        lines.append('    ["%s"] = { width = %d, height = %d, shift = { %.4f, %.4f } },'
                     % (name, s["width"], s["height"], s["shift"][0], s["shift"][1]))
    lines.append("}")
    with open(LUA_OUT, "w", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    print("Wrote", LUA_OUT)


# Only render when run as a script; render_barn.py imports this file for its helpers.
if __name__ == "__main__":
    main()
