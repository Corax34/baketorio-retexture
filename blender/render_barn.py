"""Barn (Baketorio+) as a pasture that shows as many animals as are in the barn.

Run from the repo root:
    blender -b --factory-startup --python blender/render_barn.py [-- only pig] [preview <dir>]

Renders the 6x6 paddock (fence, shed, hay and water troughs, pig feeder, wallow) as one
static picture, and each animal position of every kind (cows, pigs) on its own, animated
with a skeleton (cow_rig.py, pig_rig.py) over FRAMES frames. Every animal is split the way
vanilla machines do it: a still image of everything that never moves plus small patches
covering only the pixels that change (head, ears, tail). Each animal's shadow is a separate
animation, so the game can draw all shadows below all animals. scripts/barn.lua draws one
animal per animal in the barn, of the kind its recipe uses.

'only <kinds>' re-renders just those kinds (comma separated) and keeps the others' files.

Writes graphics/barn-base.png, barn-shadow.png, barn-icon.png (with cows),
barn-<kind>-N-*.png and prototypes/barn-sprites.lua plus barn-sprites-<kind>.lua.
With 'preview <dir>' it also writes frames rebuilt from those files (as the game layers
them) over grass, for checking.

Lua pipe positions (north orientation, Factorio y down) the scene is laid out for:
output north at (0.5, -2.5), inputs at west (-2.5, 0.5), east (2.5, 0.5), south (0.5, 2.5).
"""

import math
import os
import random
import shutil
import sys
import tempfile

import bpy
import numpy as np
from mathutils import Matrix, Vector

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import render_greenhouse as gh  # noqa: E402  (shared scene, materials and look)
import cow_model  # noqa: E402
import cow_rig  # noqa: E402
import pig_model  # noqa: E402
import pig_rig  # noqa: E402
import sheep_model  # noqa: E402
import sheep_rig  # noqa: E402
import chicken_model  # noqa: E402
import chicken_rig  # noqa: E402

M, between, compose, material, place = gh.M, gh.between, gh.compose, gh.material, gh.place
V = Vector

ARGS = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
PREVIEW_DIR = ARGS[ARGS.index("preview") + 1] if "preview" in ARGS else None

REPO = os.path.dirname(HERE)
GFX = os.path.join(REPO, "graphics")
LUA_DIR = os.path.join(REPO, "prototypes")

SIZE = 6                  # footprint in tiles (Baketorio+'s barn is 3x3; this is double)
FRAMES = 48
TILE_PX = gh.TILE_PX
TOP_H = 2.15              # highest point of the scene (shed roof), for framing
PATCHES_PER_ANIMAL = 2    # animated patches per animal after merging (memory vs number of layers)
ANIMAL_SCALE = 0.85       # animals a little smaller than life size, to suit the buildings
HEN_SCALE = 1.25          # hens larger than life: at life size they'd be specks in the paddock
MALE_SCALE = 1.06         # bulls, boars and rams are a little bigger than the females
HAY_H = 0.5               # height of the hay trough (fits the scaled cows' reach)
WATER_H = 0.48            # height of the water trough rim
FEEDER = (-2.3, -1.25)    # centre of the low pig feeder along the west fence
FEEDER_H = 0.24           # its rim height (pigs eat at about 0.18)
WALLOW = (0.55, -0.05)    # centre of the mud wallow

# Splitting the animation into static image + patches.
THRESH = 4 / 255          # a pixel 'moves' if any channel changes by more than this
GROUP = 3                 # moving areas closer than this (px) may share a patch
SPECK_PX = 8              # isolated flecks smaller than this ...
SPECK_MAX = 24 / 255      # ... that never change more than this are lighting noise, not motion

# Animal positions, in the order they appear as the barn fills up:
# (x, y, yaw degrees, pose, coat); scene x = east, y = north.
COWS = [
    (0.2, 1.12, 90, "eating", "cow_holstein"),     # muzzle in the hay trough
    (1.02, -0.8, 0, "drinking", "cow_holstein"),   # muzzle in the water trough
    (1.55, 1.12, 90, "eating", "cow_brown"),
    (-0.9, -1.5, 220, "grazing", "cow_holstein"),
    (-1.95, 0.25, -15, "looking", "cow_holstein"),  # in front of the shed
    (1.5, -2.15, 165, "standing", "cow_brown"),     # south-east corner
]
PIGS = [
    (-1.69, -1.55, 180, "eating", "pig_pink"),     # snout in the low feeder
    (-1.69, -0.95, 180, "eating", "pig_spotted"),
    (0.35, -1.75, -35, "rooting", "pig_pink"),
    (1.45, 0.8, 145, "standing", "pig_black"),
    (0.55, -0.05, 195, "lying", "pig_pink"),       # asleep in the wallow
    (-1.75, 0.35, 170, "lying", "pig_spotted"),    # asleep in front of the shed
]
SHEEP = [
    (0.28, 1.68, 90, "eating", "suffolk"),        # muzzle in the hay trough
    (1.38, 1.68, 90, "eating", "whiteface"),
    (1.5, -0.88, 0, "drinking", "suffolk"),       # muzzle in the water trough
    (-0.8, -1.55, 210, "grazing", "suffolk"),
    (-1.85, 0.38, 165, "lying", "black"),         # chewing the cud in front of the shed
    (0.5, -2.05, -25, "standing", "whiteface"),
]
HENS = [
    (-1.9, -1.6, 180, "feeding", "hen_brown"),     # breast at the low feeder, beak in the feed
    (-1.9, -0.9, 180, "feeding", "hen_white"),
    (0.45, -1.3, 25, "flapping", "hen_white"),     # in the open, where the wings have room
    (1.4, 0.9, 210, "scratching", "hen_black"),
    (-1.6, 0.6, 160, "sitting", "hen_brown"),      # settled in front of the shed
    (1.0, -2.2, -60, "pecking", "hen_brown"),
]
# Hatching eggs: a hen brooding on a straw nest for each egg.
NESTS = [
    (-2.25, 0.5, 150, "brooding", "hen_brown"),    # in front of the shed
    (-1.7, 0.55, 200, "brooding", "hen_white"),
    (-1.15, 0.5, 170, "brooding", "hen_brown"),
    (2.35, 0.45, 190, "brooding", "hen_white"),     # along the east fence
    (2.35, 1.05, 160, "brooding", "hen_brown"),
    (1.8, 0.75, 120, "brooding", "hen_black"),
]
# Sheep coats: (fleece, face and legs).
SHEEP_COATS = {
    "suffolk": ("wool_white", "sheep_skin_black"),
    "whiteface": ("wool_white", "sheep_skin_white"),
    "black": ("wool_black", "sheep_skin_black"),
}
def pig_parts(coat):
    """Snout, ear and trotter materials for a pig's coat."""
    if coat == "pig_black":
        return {"snout_mat": "snout_black", "ear_mat": "ear_black", "trotter_mat": "trotter"}
    return {"snout_mat": "snout_pink", "ear_mat": "ear_pink", "trotter_mat": "trotter_pale"}


ANIMALS = {
    "cow": {
        "positions": COWS,
        "build": lambda name, pose, coat, mats: cow_model.build_cow_mesh(
            name, pose, coat, mats, horns=coat == "cow_brown"),
        "rig": cow_rig.rig_cow,
        "pose": cow_rig.pose_cow,
    },
    # Males take the same places and poses as the females (the barn script swaps them in).
    "bull": {
        "positions": COWS,
        "scale": ANIMAL_SCALE * MALE_SCALE,
        "build": lambda name, pose, coat, mats: cow_model.build_cow_mesh(name, pose, coat, mats, bull=True),
        "rig": cow_rig.rig_cow,
        "pose": cow_rig.pose_cow,
    },
    "pig": {
        "positions": PIGS,
        "build": lambda name, pose, coat, mats: pig_model.build_pig_mesh(name, pose, coat, mats, **pig_parts(coat)),
        "rig": pig_rig.rig_pig,
        "pose": pig_rig.pose_pig,
        "roll_lying": True,
    },
    "boar": {
        "positions": PIGS,
        "scale": ANIMAL_SCALE * MALE_SCALE,
        "roll_lying": True,
        "build": lambda name, pose, coat, mats: pig_model.build_pig_mesh(name, pose, coat, mats, boar=True,
                                                                         **pig_parts(coat)),
        "rig": pig_rig.rig_pig,
        "pose": pig_rig.pose_pig,
    },
    "sheep": {
        "positions": SHEEP,
        "build": lambda name, pose, coat, mats: sheep_model.build_sheep_mesh(
            name, pose, SHEEP_COATS[coat][0], mats, SHEEP_COATS[coat][1]),
        "rig": sheep_rig.rig_sheep,
        "pose": sheep_rig.pose_sheep,
    },
    "ram": {
        "positions": SHEEP,
        "scale": ANIMAL_SCALE * MALE_SCALE,
        "build": lambda name, pose, coat, mats: sheep_model.build_sheep_mesh(
            name, pose, SHEEP_COATS[coat][0], mats, SHEEP_COATS[coat][1], ram=True),
        "rig": sheep_rig.rig_sheep,
        "pose": sheep_rig.pose_sheep,
    },
    "chicken": {
        "positions": HENS,
        "scale": HEN_SCALE,
        "build": lambda name, pose, coat, mats: chicken_model.build_chicken_mesh(name, pose, coat, mats),
        "rig": chicken_rig.rig_chicken,
        "pose": chicken_rig.pose_chicken,
    },
    "nest": {
        "positions": NESTS,
        "scale": HEN_SCALE,
        "build": lambda name, pose, coat, mats: chicken_model.build_chicken_mesh(name, pose, coat, mats),
        "rig": chicken_rig.rig_chicken,
        "pose": chicken_rig.pose_chicken,
    },
}
ONLY = ARGS[ARGS.index("only") + 1].split(",") if "only" in ARGS else list(ANIMALS)


# Materials -------------------------------------------------------------------------

def pasture_materials(mats):
    mats.update({
        "pasture": material("pasture", (0.1, 0.12, 0.05), rough=0.95, noise=((0.17, 0.16, 0.08), 4, 8)),
        "grass_tuft": material("grass_tuft", (0.2, 0.26, 0.1), rough=0.8),
        "grass_dry": material("grass_dry", (0.36, 0.32, 0.18), rough=0.85),
        "mud": material("mud", (0.12, 0.08, 0.05), rough=0.4, noise=((0.19, 0.13, 0.08), 9, 6)),
        "wet_mud": material("wet_mud", (0.08, 0.06, 0.04), rough=0.15, noise=((0.13, 0.1, 0.07), 9, 6)),
        "fence": material("fence", (0.37, 0.33, 0.28), rough=0.9, noise=((0.27, 0.24, 0.2), 14, 6)),
        "plank": material("plank", (0.3, 0.22, 0.15), rough=0.85, noise=((0.22, 0.16, 0.1), 10, 6)),
        "corrugated": material("corrugated", (0.4, 0.29, 0.21), rough=0.6, metallic=0.5,
                               noise=((0.28, 0.18, 0.12), 6, 8)),
        "galvanised": material("galvanised", (0.5, 0.52, 0.53), rough=0.45, metallic=0.85,
                               noise=((0.38, 0.38, 0.37), 8)),
        "water": material("water", (0.1, 0.2, 0.22), rough=0.02, specular=1.0),
        "cow_holstein": material("cow_holstein", (0.8, 0.79, 0.74), rough=0.7,
                                 noise=((0.04, 0.04, 0.04), 2.4, 2, (0.5, 0.53))),
        "cow_brown": material("cow_brown", (0.42, 0.24, 0.13), rough=0.75, noise=((0.32, 0.17, 0.09), 4)),
        "muzzle": material("muzzle", (0.62, 0.42, 0.38), rough=0.5),
        "brass": material("brass", (0.6, 0.42, 0.15), rough=0.3, metallic=1.0),
        "tusk": material("tusk", (0.78, 0.72, 0.6), rough=0.35),
        "egg_shell": material("egg_shell", (0.72, 0.5, 0.33), rough=0.45),
        "ram_horn": material("ram_horn", (0.5, 0.42, 0.32), rough=0.55, noise=((0.36, 0.3, 0.22), 40)),
        "udder": material("udder", (0.78, 0.55, 0.5), rough=0.6),
        "hoof": material("hoof", (0.09, 0.08, 0.07), rough=0.6),
        "horn": material("horn", (0.8, 0.76, 0.64), rough=0.5),
        "tuft": material("tuft", (0.06, 0.05, 0.05), rough=0.9),
        "pig_pink": material("pig_pink", (0.83, 0.61, 0.59), rough=0.6, noise=((0.77, 0.53, 0.51), 6)),
        "pig_spotted": material("pig_spotted", (0.83, 0.61, 0.59), rough=0.6,
                                noise=((0.1, 0.09, 0.09), 2.6, 2, (0.58, 0.61))),
        "pig_black": material("pig_black", (0.035, 0.03, 0.03), rough=0.65, specular=0.3,
                              noise=((0.065, 0.055, 0.055), 5)),
        "snout_pink": material("snout_pink", (0.72, 0.32, 0.31), rough=0.45),
        "ear_pink": material("ear_pink", (0.83, 0.35, 0.31), rough=0.55),     # thin, pink skin
        "wool_white": material("wool_white", (0.7, 0.67, 0.6), rough=0.9, noise=((0.55, 0.51, 0.44), 9)),
        "wool_black": material("wool_black", (0.045, 0.038, 0.034), rough=0.9, noise=((0.085, 0.07, 0.058), 9)),
        "sheep_skin_black": material("sheep_skin_black", (0.03, 0.026, 0.026), rough=0.5),
        "sheep_skin_white": material("sheep_skin_white", (0.66, 0.57, 0.5), rough=0.55),
        "comb_red": material("comb_red", (0.55, 0.03, 0.025), rough=0.4),
        "beak_yellow": material("beak_yellow", (0.75, 0.52, 0.16), rough=0.4),
        "eye_black": material("eye_black", (0.02, 0.015, 0.01), rough=0.1),
        "chicken_leg": material("chicken_leg", (0.72, 0.55, 0.16), rough=0.5),
        "snout_black": material("snout_black", (0.46, 0.37, 0.35), rough=0.45),
        "trotter": material("trotter", (0.28, 0.24, 0.21), rough=0.6),
        "trotter_pale": material("trotter_pale", (0.56, 0.45, 0.39), rough=0.5),
        "ear_black": material("ear_black", (0.045, 0.038, 0.036), rough=0.6),
        "nostril": material("nostril", (0.06, 0.03, 0.03), rough=0.5),
        "puddle": material("puddle", (0.07, 0.055, 0.035), rough=0.05, specular=0.6),
        "pig_feed": material("pig_feed", (0.5, 0.4, 0.24), rough=0.9, noise=((0.36, 0.28, 0.15), 30, 4)),
    })
    # Coat patterns follow the skin: read from the rest positions stored on the animal mesh.
    for key in ("cow_holstein", "cow_brown", "pig_pink", "pig_spotted", "pig_black", "wool_white", "wool_black"):
        nt = mats[key].node_tree
        attr = nt.nodes.new("ShaderNodeAttribute")
        attr.attribute_name = "rest"
        for node in nt.nodes:
            if node.type == "TEX_NOISE":
                nt.links.new(attr.outputs["Vector"], node.inputs["Vector"])
    # Pigs get mud on their legs and the underside of the belly.
    for key in ("pig_pink", "pig_spotted", "pig_black"):
        add_mud(mats[key], top=0.13, amount=0.55)
    # Wool: soft sheen and a fine fibrous bump on top of the lumps in the geometry.
    for key in ("wool_white", "wool_black"):
        add_wool_bump(mats[key])
    for coat, colours in HEN_COATS.items():
        mats[coat] = hen_body_material(coat, colours)
        mats[coat + "_wing"] = feather_material(coat + "_wing", colours["wing"], colours["fringe"], colours["shaft"],
                                                colours["film"])
        mats[coat + "_tail"] = feather_material(coat + "_tail", colours["tail"], colours["tail_fringe"],
                                                colours["shaft"], colours["tail_film"])
    gh.VELVET.update({"wool_white", "wool_black"})
    gh.GRIMY.update({"fence", "plank", "corrugated", "galvanised"})
    gh.BEVELED.update({"fence", "plank"})
    gh.MATTE.update({"pasture", "mud", "fence", "plank"})


# Hen plumage per breed: body, hackle (neck and cape), saddle (in front of the tail), breast,
# lacing (feather edges on the body); wing feathers with their fringe and shaft; tail
# feathers; thin-film thickness in nm for an iridescent sheen (0 = none).
HEN_COATS = {
    "hen_brown": {   # red-brown laying hen: golden hackle, darker wings, black tail with a green sheen
        "body": (0.3, 0.11, 0.038), "hackle": (0.56, 0.27, 0.07), "saddle": (0.44, 0.19, 0.05),
        "breast": (0.36, 0.14, 0.045), "lace": (0.14, 0.05, 0.018), "lace_amount": 0.4,
        "wing": (0.25, 0.09, 0.03), "fringe": (0.48, 0.22, 0.07), "shaft": (0.55, 0.36, 0.17), "film": 0,
        "tail": (0.025, 0.022, 0.02), "tail_fringe": (0.05, 0.04, 0.03), "tail_film": 255},
    "hen_white": {   # white leghorn: cream hackle
        "body": (0.8, 0.78, 0.72), "hackle": (0.84, 0.79, 0.64), "saddle": (0.82, 0.79, 0.7),
        "breast": (0.79, 0.77, 0.72), "lace": (0.64, 0.62, 0.58), "lace_amount": 0.2,
        "wing": (0.79, 0.77, 0.71), "fringe": (0.86, 0.84, 0.79), "shaft": (0.68, 0.66, 0.6), "film": 0,
        "tail": (0.8, 0.78, 0.72), "tail_fringe": (0.86, 0.84, 0.79), "tail_film": 0},
    "hen_black": {   # black hen with a beetle-green sheen (thin film: green first-order reflection)
        "body": (0.022, 0.023, 0.026), "hackle": (0.03, 0.032, 0.034), "saddle": (0.024, 0.026, 0.028),
        "breast": (0.02, 0.021, 0.024), "lace": (0.01, 0.01, 0.012), "lace_amount": 0.3,
        "wing": (0.02, 0.022, 0.025), "fringe": (0.03, 0.032, 0.036), "shaft": (0.05, 0.05, 0.055), "film": 255,
        "tail": (0.018, 0.02, 0.022), "tail_fringe": (0.03, 0.032, 0.036), "tail_film": 255},
}


class _Nodes:
    """Small helper for building shader node trees."""

    def __init__(self, mat):
        self.nt = mat.node_tree
        self.bsdf = gh.principled(mat)

    def new(self, kind, **inputs):
        node = self.nt.nodes.new(kind)
        for k, v in inputs.items():
            if isinstance(v, bpy.types.NodeSocket):
                self.nt.links.new(v, node.inputs[k])
            else:
                node.inputs[k].default_value = v
        return node

    def rgb(self, c):
        node = self.nt.nodes.new("ShaderNodeRGB")
        node.outputs[0].default_value = (*c, 1)
        return node.outputs[0]

    def mix(self, fac, a, b, blend="MIX"):
        node = self.nt.nodes.new("ShaderNodeMix")
        node.data_type = "RGBA"
        node.blend_type = blend
        sockets = [i for i in node.inputs if i.type == "RGBA"]
        for sock, val in ((node.inputs[0], fac), (sockets[0], a), (sockets[1], b)):
            if isinstance(val, bpy.types.NodeSocket):
                self.nt.links.new(val, sock)
            else:
                sock.default_value = val if sock.type != "RGBA" else (*val, 1)
        return next(o for o in node.outputs if o.type == "RGBA")

    def ramp(self, value, lo, hi, to_lo, to_hi):
        node = self.new("ShaderNodeMapRange", Value=value)
        node.inputs["From Min"].default_value, node.inputs["From Max"].default_value = lo, hi
        node.inputs["To Min"].default_value, node.inputs["To Max"].default_value = to_lo, to_hi
        return node.outputs["Result"]

    def gray(self, value):
        node = self.new("ShaderNodeCombineColor", Red=value, Green=value, Blue=value)
        return node.outputs[0]

    def math(self, op, a, b=0.0):
        node = self.nt.nodes.new("ShaderNodeMath")
        node.operation = op
        for sock, val in zip(node.inputs, (a, b)):
            if isinstance(val, bpy.types.NodeSocket):
                self.nt.links.new(val, sock)
            else:
                sock.default_value = val
        return node.outputs[0]

    def finish(self, colour, height, bump=0.3, rough=0.62, film=0):
        self.nt.links.new(colour, self.bsdf.inputs["Base Color"])
        self.bsdf.inputs["Roughness"].default_value = rough
        self.bsdf.inputs["Sheen Weight"].default_value = 0.35            # the soft gloss of feathers
        self.bsdf.inputs["Sheen Roughness"].default_value = 0.45
        if film:                                  # iridescent, but still soft feathers
            self.bsdf.inputs["Thin Film Thickness"].default_value = film
            self.bsdf.inputs["Thin Film IOR"].default_value = 1.55
            self.bsdf.inputs["Roughness"].default_value = 0.52
            self.bsdf.inputs["Specular IOR Level"].default_value = 0.35
        b = self.new("ShaderNodeBump", Height=height, Strength=bump, Distance=0.004)
        self.nt.links.new(b.outputs["Normal"], self.bsdf.inputs["Normal"])


def hen_body_material(name, c):
    """Body plumage: overlapping contour feathers (cells of a Voronoi pattern on the rest
    positions, longer than wide, each lighter towards its tip at the back and bottom where it
    lies over the next and darker at its root, laced at the rim, varying slightly), the
    breed's colours by zone (hackle, saddle, breast from the "zones" attribute) and darker
    streaks in the hackle."""
    mat = material(name, c["body"], rough=0.62)
    n = _Nodes(mat)
    rest = n.new("ShaderNodeAttribute").outputs["Vector"]
    rest.node.attribute_name = "rest"
    zones = n.new("ShaderNodeAttribute")
    zones.attribute_name = "zones"
    split = n.new("ShaderNodeSeparateColor", Color=zones.outputs["Color"])
    scaled = n.new("ShaderNodeVectorMath", Vector=rest)
    scaled.operation = "MULTIPLY"
    scaled.inputs[1].default_value = (27.0, 36.0, 36.0)
    cells = n.new("ShaderNodeTexVoronoi", Vector=scaled.outputs[0], Scale=1.0)
    offset = n.new("ShaderNodeVectorMath")
    offset.operation = "SUBTRACT"
    n.nt.links.new(scaled.outputs[0], offset.inputs[0])
    n.nt.links.new(cells.outputs["Position"], offset.inputs[1])
    off = n.new("ShaderNodeSeparateXYZ", Vector=offset.outputs[0])
    tipward = n.math("MULTIPLY", n.math("ADD", off.outputs["X"], off.outputs["Z"]), -0.7)
    colour = n.rgb(c["body"])
    colour = n.mix(split.outputs["Blue"], colour, n.rgb(c["breast"]))
    colour = n.mix(split.outputs["Green"], colour, n.rgb(c["saddle"]))
    colour = n.mix(split.outputs["Red"], colour, n.rgb(c["hackle"]))
    # Hackle: long narrow feathers hanging down the neck, light at their tips, dark at the roots.
    hscaled = n.new("ShaderNodeVectorMath", Vector=rest)
    hscaled.operation = "MULTIPLY"
    hscaled.inputs[1].default_value = (50.0, 50.0, 14.0)
    hcells = n.new("ShaderNodeTexVoronoi", Vector=hscaled.outputs[0], Scale=1.0)
    hoffset = n.new("ShaderNodeVectorMath")
    hoffset.operation = "SUBTRACT"
    n.nt.links.new(hscaled.outputs[0], hoffset.inputs[0])
    n.nt.links.new(hcells.outputs["Position"], hoffset.inputs[1])
    hshade = n.ramp(n.new("ShaderNodeSeparateXYZ", Vector=hoffset.outputs[0]).outputs["Z"], -0.5, 0.5, 1.12, 0.68)
    colour = n.mix(1.0, colour, n.mix(split.outputs["Red"], (1.0, 1.0, 1.0), n.gray(hshade)), "MULTIPLY")
    shade = n.ramp(tipward, -0.6, 0.6, 0.8, 1.12)                            # tip light, root dark
    colour = n.mix(1.0, colour, n.gray(shade), "MULTIPLY")
    vary = n.ramp(n.new("ShaderNodeSeparateColor", Color=cells.outputs["Color"]).outputs["Red"], 0, 1, 0.9, 1.1)
    colour = n.mix(1.0, colour, n.gray(vary), "MULTIPLY")
    lace = n.ramp(cells.outputs["Distance"], 0.4, 0.62, 0.0, c["lace_amount"])
    colour = n.mix(lace, colour, n.rgb(c["lace"]))
    n.finish(colour, tipward, bump=0.2, film=c["film"])
    return mat


def feather_material(name, base, fringe, shaft, film=0):
    """Wing and tail feathers, from their UVs (u along the shaft, v across): a fine pale
    shaft, the vane darker towards the covered root, a lighter fringe and fine barbs."""
    mat = material(name, base, rough=0.6)
    n = _Nodes(mat)
    uv = n.new("ShaderNodeUVMap")
    xyz = n.new("ShaderNodeSeparateXYZ", Vector=uv.outputs["UV"])
    u = xyz.outputs["X"]
    across = n.math("ABSOLUTE", n.math("SUBTRACT", xyz.outputs["Y"], 0.5))
    across = n.math("MULTIPLY", across, 2.0)                               # 0 at the shaft, 1 at the edge
    colour = n.rgb(base)
    colour = n.mix(n.ramp(across, 0.62, 0.95, 0.0, 1.0), colour, n.rgb(fringe))
    colour = n.mix(1.0, colour, n.gray(n.ramp(u, 0.0, 0.45, 0.7, 1.0)), "MULTIPLY")
    barb_coord = n.new("ShaderNodeCombineXYZ", X=n.math("ADD", u, n.math("MULTIPLY", across, 0.6)))
    barbs = n.new("ShaderNodeTexWave", Vector=barb_coord.outputs[0], Scale=24.0)
    colour = n.mix(1.0, colour, n.gray(n.ramp(barbs.outputs["Fac"], 0, 1, 0.9, 1.06)), "MULTIPLY")
    colour = n.mix(n.ramp(across, 0.03, 0.09, 1.0, 0.0), colour, n.rgb(shaft))
    n.finish(colour, barbs.outputs["Fac"], bump=0.12, film=film)
    return mat


def add_wool_bump(mat, scale=70, strength=0.5):
    """Fine texture for wool or feathers: a small-scale noise (following the body's rest
    positions) as a bump on the surface normal."""
    nt = mat.node_tree
    bsdf = gh.principled(mat)
    attr = nt.nodes.new("ShaderNodeAttribute")
    attr.attribute_name = "rest"
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = scale
    noise.inputs["Detail"].default_value = 3
    nt.links.new(attr.outputs["Vector"], noise.inputs["Vector"])
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = strength
    bump.inputs["Distance"].default_value = 0.01
    nt.links.new(noise.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])


def add_mud(mat, top, amount=0.85):
    """Mixes a dried-mud colour into the coat, strongest at the ground and fading out at
    height `top` (pig-local, from the rest positions, so it stays put when posed)."""
    nt = mat.node_tree
    bsdf = gh.principled(mat)
    base_in = bsdf.inputs["Base Color"]
    if base_in.is_linked:
        coat = base_in.links[0].from_socket
    else:
        rgb = nt.nodes.new("ShaderNodeRGB")
        rgb.outputs[0].default_value = base_in.default_value
        coat = rgb.outputs[0]
    attr = nt.nodes.new("ShaderNodeAttribute")
    attr.attribute_name = "rest"
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(attr.outputs["Vector"], sep.inputs[0])
    height = nt.nodes.new("ShaderNodeMapRange")
    height.inputs["From Min"].default_value = top
    height.inputs["From Max"].default_value = 0.02
    height.inputs["To Max"].default_value = amount
    nt.links.new(sep.outputs["Z"], height.inputs["Value"])
    mix = nt.nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    nt.links.new(height.outputs["Result"], mix.inputs[0])
    nt.links.new(coat, next(i for i in mix.inputs if i.name == "A" and i.type == "RGBA"))
    next(i for i in mix.inputs if i.name == "B" and i.type == "RGBA").default_value = (0.2, 0.15, 0.1, 1)
    nt.links.new(next(o for o in mix.outputs if o.type == "RGBA"), base_in)


# Scene -----------------------------------------------------------------------------

def pasture_parts(rng):
    """Ground, fence, shed and props of the paddock (scene units = tiles)."""
    hw = hl = SIZE / 2 - 0.15
    ground, fence, shed, props = [], [], [], []
    ground.append(("cube", M((0, 0, 0.01), scale=(SIZE - 0.3, SIZE - 0.3, 0.02)), "pasture"))
    # Trampled mud where cows gather: hay trough, water trough, gate, shed front.
    for cx, cy, sx, sy in ((1.0, 2.0, 2.6, 0.9), (1.8, -0.7, 1.2, 1.2), (-1.5, -2.4, 1.6, 0.8),
                           (-1.7, 0.65, 1.4, 0.6), (-1.2, -1.4, 1.0, 1.0)):
        for _ in range(3):
            ground.append(("ico", M((cx + rng.uniform(-0.3, 0.3), cy + rng.uniform(-0.2, 0.2), 0.012),
                                    rot=(0, 0, rng.uniform(0, 3.14)),
                                    scale=(sx * rng.uniform(0.6, 1.0), sy * rng.uniform(0.6, 1.0), 0.03)), "mud"))
    for _ in range(95):
        x, y = rng.uniform(-hw + 0.3, hw - 0.3), rng.uniform(-hl + 0.3, hl - 0.3)
        mat = "grass_dry" if rng.random() < 0.3 else "grass_tuft"
        # Nothing grows in the wallow (the random draws still happen, so the rest stays put).
        in_wallow = ((x - WALLOW[0]) / 1.2) ** 2 + ((y - WALLOW[1]) / 0.8) ** 2 < 1
        for _ in range(7):
            a = rng.uniform(0, 2 * math.pi)
            lean = rng.uniform(0.2, 0.7)
            h = rng.uniform(0.12, 0.25)
            base = V((x + rng.uniform(-0.05, 0.05), y + rng.uniform(-0.05, 0.05), 0.02))
            tip = base + V((math.cos(a) * math.sin(lean), math.sin(a) * math.sin(lean), math.cos(lean))) * h
            if not in_wallow:
                ground.append(("cone", between(base, tip, 0.035, 0.01), mat))

    # Post-and-rail fence, with the gate off-centre so it does not meet the south pipe.
    def fence_line(a, b):
        n = max(1, round((b - a).length / 1.2))
        pts = [a.lerp(b, i / n) for i in range(n + 1)]
        for p in pts:
            fence.append(("cube", M((p.x, p.y, 0.55), rot=(0, 0, rng.uniform(-0.05, 0.05)),
                                    scale=(0.14, 0.14, 1.1)), "fence"))
        for z in (0.42, 0.78, 1.02):
            for p, q in zip(pts, pts[1:]):
                fence.append(("cube", between(V((p.x, p.y, z)), V((q.x, q.y, z + rng.uniform(-0.03, 0.03))),
                                              0.06, 0.12), "fence"))

    g0, g1 = V((-2.2, -hl, 0)), V((-0.8, -hl, 0))
    fence_line(V((-hw, -hl, 0)), g0)
    fence_line(g1, V((hw, -hl, 0)))
    fence_line(V((hw, -hl, 0)), V((hw, hl, 0)))
    fence_line(V((hw, hl, 0)), V((-hw, hl, 0)))
    fence_line(V((-hw, hl, 0)), V((-hw, -hl, 0)))
    for z in (0.25, 0.6, 0.95):                                   # galvanised tube gate, closed
        fence.append(("cyl", between(g0 + V((0, 0, z)), g1 + V((0, 0, z)), 0.05), "galvanised"))
    for x in (g0.x + 0.05, (g0.x + g1.x) / 2, g1.x - 0.05):
        fence.append(("cyl", between(V((x, -hl, 0.2)), V((x, -hl, 1.0)), 0.05), "galvanised"))
    fence.append(("cyl", between(g0 + V((0, 0, 0.25)), g1 + V((0, 0, 0.95)), 0.04), "galvanised"))

    # Open-fronted shed in the north-west corner.
    x0, x1, y0, y1 = -hw + 0.3, -0.95, 0.95, hl - 0.3
    back_h, front_h = 2.0, 1.7
    for x, y, h in ((x0, y0, front_h), (x1, y0, front_h), (x0, y1, back_h), (x1, y1, back_h)):
        shed.append(("cube", M((x, y, h / 2), scale=(0.15, 0.15, h)), "plank"))
    for i in range(7):                                            # back and west walls
        z = 0.15 + i * 0.26
        shed.append(("cube", M(((x0 + x1) / 2, y1, z), scale=(x1 - x0, 0.06, 0.24)), "plank"))
        shed.append(("cube", M((x0, (y0 + y1) / 2, z), scale=(0.06, y1 - y0, 0.24)), "plank"))
    pitch = math.atan2(back_h - front_h, y1 - y0)
    roof = Matrix.LocRotScale(V(((x0 + x1) / 2, (y0 + y1) / 2 - 0.12, (back_h + front_h) / 2 + 0.08)),
                              Matrix.Rotation(pitch, 4, "X").to_quaternion(), V((1, 1, 1)))
    rw, rl = (x1 - x0) + 0.35, (y1 - y0) + 0.45
    shed.append(("cube", roof @ M(scale=(rw, rl, 0.03)), "corrugated"))
    n_ribs = int(rw / 0.12)
    for i in range(n_ribs):                                       # corrugation ribs
        x = -rw / 2 + (i + 0.5) * rw / n_ribs
        shed.append(("cyl", roof @ between(V((x, -rl / 2, 0.02)), V((x, rl / 2, 0.02)), 0.05, 0.035),
                     "corrugated"))
    # Ground cover (straw bedding, mud, puddles) is part of the ground, so animal shadows
    # falling on it are caught like on the grass.
    ground.append(("cube", M(((x0 + x1) / 2, (y0 + y1) / 2, 0.025), scale=(x1 - x0 - 0.1, y1 - y0 - 0.1, 0.03)),
                   "straw"))
    for bx, by, bz in ((x0 + 0.45, y1 - 0.45, 0.33), (x0 + 1.05, y1 - 0.45, 0.33), (x0 + 0.75, y1 - 0.45, 0.9)):
        props.append(("cyl", M((bx, by, bz), rot=(0, math.pi / 2, 0), scale=(0.62, 0.62, 0.66)), "straw"))
        props.append(("cyl", M((bx, by, bz), rot=(0, math.pi / 2, 0), scale=(0.44, 0.44, 0.67)), "straw_old"))

    # Hollow galvanised water trough on the east fence. Its feed pipe runs through the fence
    # where the barn's east pipe connection is (scene y = -0.5).
    tx, ty = hw - 0.5, -0.7
    tw, tl, th, t = 0.62, 1.2, WATER_H, 0.04
    props.append(("cube", M((tx, ty, 0.04), scale=(tw, tl, 0.08)), "galvanised"))
    for sx in (-1, 1):
        props.append(("cube", M((tx + sx * (tw - t) / 2, ty, th / 2), scale=(t, tl, th)), "galvanised"))
        props.append(("cyl", between(V((tx + sx * (tw - t) / 2, ty - tl / 2, th)),
                                     V((tx + sx * (tw - t) / 2, ty + tl / 2, th)), 0.06), "galvanised"))
    for sy in (-1, 1):
        props.append(("cube", M((tx, ty + sy * (tl - t) / 2, th / 2), scale=(tw, t, th)), "galvanised"))
        props.append(("cyl", between(V((tx - tw / 2, ty + sy * (tl - t) / 2, th)),
                                     V((tx + tw / 2, ty + sy * (tl - t) / 2, th)), 0.06), "galvanised"))
    props.append(("cube", M((tx, ty, th - 0.07), scale=(tw - 2 * t, tl - 2 * t, 0.02)), "water"))
    props.append(("cyl", between(V((tx + 0.25, -0.5, th + 0.06)), V((hw + 0.35, -0.5, th + 0.06)), 0.12), "galvanised"))
    props.append(("cyl", between(V((tx + 0.2, -0.5, th + 0.06)), V((tx + 0.2, -0.5, th - 0.07)), 0.08), "galvanised"))

    # Hay trough along the north fence.
    fx0, fx1, fy = -0.65, hw - 0.3, hl - 0.4
    props.append(("cube", M(((fx0 + fx1) / 2, fy, HAY_H / 2), scale=(fx1 - fx0, 0.55, HAY_H)), "plank"))
    props.append(("cube", M(((fx0 + fx1) / 2, fy, HAY_H - 0.02), scale=(fx1 - fx0 - 0.1, 0.44, 0.06)), "straw_mat"))
    for _ in range(500):
        x = rng.uniform(fx0 + 0.08, fx1 - 0.08)
        y = fy + rng.uniform(-0.2, 0.2)
        z = HAY_H + 0.01 + 0.08 * (1 - (2 * (y - fy) / 0.44) ** 2) + rng.uniform(0, 0.03)
        a = rng.uniform(0, math.pi)
        d = V((math.cos(a), math.sin(a), rng.uniform(-0.3, 0.3))) * rng.uniform(0.04, 0.08)
        c = V((x, y, z))
        props.append(("cube", between(c - d, c + d, 0.014, 0.004), rng.choice(["straw", "straw_pale", "straw_old"])))

    # Low galvanised pig feeder along the west fence, full of meal, with one divider.
    px, py = FEEDER
    pw, pl, ph, t = 0.42, 1.3, FEEDER_H, 0.035
    props.append(("cube", M((px, py, 0.03), scale=(pw, pl, 0.06)), "galvanised"))
    for sx in (-1, 1):
        props.append(("cube", M((px + sx * (pw - t) / 2, py, ph / 2), scale=(t, pl, ph)), "galvanised"))
        props.append(("cyl", between(V((px + sx * (pw - t) / 2, py - pl / 2, ph)),
                                     V((px + sx * (pw - t) / 2, py + pl / 2, ph)), 0.05), "galvanised"))
    for sy in (-1, 1):
        props.append(("cube", M((px, py + sy * (pl - t) / 2, ph / 2), scale=(pw, t, ph)), "galvanised"))
        props.append(("cyl", between(V((px - pw / 2, py + sy * (pl - t) / 2, ph)),
                                     V((px + pw / 2, py + sy * (pl - t) / 2, ph)), 0.05), "galvanised"))
    props.append(("cube", M((px, py, ph - 0.07), scale=(pw - 2 * t, pl - 2 * t, 0.04)), "pig_feed"))
    props.append(("cyl", between(V((px - pw / 2, py, ph + 0.02)), V((px + pw / 2, py, ph + 0.02)), 0.04),
                  "galvanised"))
    for _ in range(5):   # trampled, spilled-on ground in front of the feeder
        ground.append(("ico", M((px + 0.55 + rng.uniform(-0.2, 0.2), py + rng.uniform(-0.6, 0.6), 0.016),
                                rot=(0, 0, rng.uniform(0, 3.14)),
                                scale=(rng.uniform(0.5, 0.9), rng.uniform(0.4, 0.7), 0.02)), "mud"))

    # Mud wallow: wet, glossy mud with a few puddles.
    wx, wy = WALLOW
    for _ in range(7):
        ground.append(("ico", M((wx + rng.uniform(-0.45, 0.45), wy + rng.uniform(-0.25, 0.25), 0.014),
                                rot=(0, 0, rng.uniform(-0.4, 0.4)),
                                scale=(rng.uniform(1.2, 1.8), rng.uniform(0.7, 1.1), 0.03)), "wet_mud"))
    # Puddles sink into the mud, so only their middle shows, with an uneven shore line.
    for _ in range(4):
        ground.append(("ico", M((wx + rng.uniform(-0.7, 0.7), wy + rng.uniform(-0.35, 0.35), 0.026),
                                rot=(0, 0, rng.uniform(0, 3.14)),
                                scale=(rng.uniform(0.25, 0.5), rng.uniform(0.15, 0.3), 0.008)), "puddle"))
    # Soil turned over by rooting.
    for _ in range(3):
        ground.append(("ico", M((0.85 + rng.uniform(-0.15, 0.15), -2.1 + rng.uniform(-0.12, 0.12), 0.016),
                                rot=(0, 0, rng.uniform(0, 3.14)),
                                scale=(rng.uniform(0.4, 0.6), rng.uniform(0.3, 0.45), 0.04)), "mud"))
    return ground, fence, shed, props


# Image helpers ---------------------------------------------------------------------

def load(path):
    img = bpy.data.images.load(path)
    a = gh.read_pixels(img)[::-1].copy()
    bpy.data.images.remove(img)
    return a                                                       # top row first


def save(a, path):
    img = bpy.data.images.new("out", a.shape[1], a.shape[0], alpha=True)
    gh.write_pixels(img, a[::-1])
    img.filepath_raw = path
    img.file_format = "PNG"
    img.save()
    bpy.data.images.remove(img)


def over(top, bottom):
    """Straight-alpha 'over' compositing."""
    ta, ba = top[..., 3:4], bottom[..., 3:4]
    oa = ta + ba * (1 - ta)
    rgb = (top[..., :3] * ta + bottom[..., :3] * ba * (1 - ta)) / np.maximum(oa, 1e-6)
    return np.concatenate([rgb, oa], -1)


def components(mask):
    """4-connected components of a boolean mask: list of (ys, xs) index arrays."""
    H, W = mask.shape
    seen = np.zeros(mask.shape, bool)
    out = []
    for y0, x0 in zip(*np.nonzero(mask)):
        if seen[y0, x0]:
            continue
        stack, pts = [(y0, x0)], []
        seen[y0, x0] = True
        while stack:
            y, x = stack.pop()
            pts.append((y, x))
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx < W and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        pts = np.array(pts)
        out.append((pts[:, 0], pts[:, 1]))
    return out


def area(b):
    return (b[2] - b[0]) * (b[3] - b[1])


def union(a, b):
    return [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]


def tight(mask, b):
    x0, y0, x1, y1 = b
    ys, xs = np.nonzero(mask[y0:y1, x0:x1])
    if not len(ys):
        return None
    return [x0 + int(xs.min()), y0 + int(ys.min()), x0 + int(xs.max()) + 1, y0 + int(ys.max()) + 1]


def split(mask, b, depth=0):
    """Splits a mostly empty box in two where that saves enough area; recursive."""
    x0, y0, x1, y1 = b
    best, parts = area(b) * 0.85, None
    for axis in (0, 1):
        lo, hi = (x0, x1) if axis == 0 else (y0, y1)
        for cut in range(lo + 4, hi - 3, 2):
            a = [x0, y0, cut, y1] if axis == 0 else [x0, y0, x1, cut]
            c = [cut, y0, x1, y1] if axis == 0 else [x0, cut, x1, y1]
            ta, tc = tight(mask, a), tight(mask, c)
            total = (area(ta) if ta else 0) + (area(tc) if tc else 0)
            if total < best:
                best, parts = total, [t for t in (ta, tc) if t]
    if parts is None or depth >= 4:
        return [b]
    return [q for p in parts for q in split(mask, p, depth + 1)]


def merge_to(boxes, n):
    """Merges the pair whose union adds the least area until n boxes remain."""
    boxes = [list(b) for b in boxes]
    while len(boxes) > n:
        cost, i, j = min((area(union(boxes[i], boxes[j])) - area(boxes[i]) - area(boxes[j]), i, j)
                         for i in range(len(boxes)) for j in range(i + 1, len(boxes)))
        boxes[i] = union(boxes[i], boxes[j])
        del boxes[j]
    return boxes


# Main ------------------------------------------------------------------------------

def place_animal(rig, me, x, y, yaw, pose, scale=ANIMAL_SCALE, roll=False):
    """Positions an animal's skeleton. Lying pigs are rolled onto their side, shifted so
    their body is centred on (x, y), and lifted so their lowest point is on the ground."""
    a = math.radians(yaw)
    rig.scale = (scale, scale, scale)
    if pose == "lying" and roll:
        shift, lift = pig_model.lying_transform(me)
        rig.rotation_euler = (pig_model.LYING_ROLL, 0, a)
        rig.location = (x - math.sin(a) * shift * scale, y + math.cos(a) * shift * scale, lift * scale + 0.02)
    else:
        rig.rotation_euler = (0, 0, a)
        rig.location = (x, y, 0.02)

def render_region(scene, cam, obj, set_pose=None, samples=16):
    """Limits rendering to the part of the frame where obj and its shadow can appear,
    which makes the per-cow renders much faster (the image size stays the same). With
    set_pose(t), the region covers the whole loop (spread wings, flying feathers)."""
    from bpy_extras.object_utils import world_to_camera_view
    sun = scene.objects["Sun"].matrix_world.to_quaternion() @ V((0, 0, -1))
    pts = []
    for k in range(samples if set_pose else 1):
        if set_pose:
            set_pose(k / samples)
            bpy.context.view_layer.update()
        deps = bpy.context.evaluated_depsgraph_get()
        for c in obj.evaluated_get(deps).bound_box:
            w = obj.matrix_world @ V(c)
            pts.append(w)
            pts.append(w - sun * (max(w.z, 0.0) / sun.z))         # where its shadow lands
    view = [world_to_camera_view(scene, cam, p) for p in pts]
    pad = 0.04
    scene.render.use_border = True
    scene.render.use_crop_to_border = False
    scene.render.border_min_x = max(0.0, min(v.x for v in view) - pad)
    scene.render.border_max_x = min(1.0, max(v.x for v in view) + pad)
    scene.render.border_min_y = max(0.0, min(v.y for v in view) - pad)
    scene.render.border_max_y = min(1.0, max(v.y for v in view) + pad)


def split_layer(frames, budget, name):
    """Splits one cow's frames into a still image (cropped to the cow) and at most `budget`
    animated patches covering the pixels that change. Returns (still, patches)."""
    H, W = frames[0].shape[:2]
    maxdiff = np.zeros((H, W), np.float32)
    for c in frames[1:]:
        maxdiff = np.maximum(maxdiff, np.abs(c - frames[0]).max(-1))
    moving = maxdiff > THRESH
    for ys, xs in components(moving):
        if len(ys) < SPECK_PX and maxdiff[ys, xs].max() < SPECK_MAX:
            moving[ys, xs] = False

    still_img = frames[0].copy()
    still_img[moving, 3] = 0
    present = still_img[..., 3] > 0.002
    still = None
    if present.any():
        x0, y0, x1, y1 = tight(present, [0, 0, W, H])
        save(still_img[y0:y1, x0:x1], os.path.join(GFX, f"{name}-still.png"))
        still = {"file": f"{name}-still", "box": (x0, y0, x1, y1)}

    grown = moving.copy()
    for _ in range(GROUP):
        m = grown.copy()
        m[1:] |= grown[:-1]; m[:-1] |= grown[1:]; m[:, 1:] |= grown[:, :-1]; m[:, :-1] |= grown[:, 1:]
        grown = m
    boxes = []
    for ys, xs in components(grown):
        keep = moving[ys, xs]
        if keep.any():
            ys, xs = ys[keep], xs[keep]
            boxes.append([int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1])
    boxes = [q for b in boxes for q in split(moving, b)]
    boxes = sorted(merge_to(boxes, budget), key=area, reverse=True)

    owned = np.zeros((H, W), bool)
    patches = []
    for k, (x0, y0, x1, y1) in enumerate(boxes, start=1):
        own = np.zeros((H, W), bool)
        own[y0:y1, x0:x1] = moving[y0:y1, x0:x1] & ~owned[y0:y1, x0:x1]
        owned |= own
        w, h = x1 - x0, y1 - y0
        per_row = max(1, min(FRAMES, 4096 // w))
        rows = -(-FRAMES // per_row)
        sheet = np.zeros((rows * h, per_row * w, 4), np.float32)
        keep = np.concatenate([np.ones((h, w, 3)), own[y0:y1, x0:x1, None]], -1)
        for f, c in enumerate(frames):
            r, col = divmod(f, per_row)
            sheet[r * h:(r + 1) * h, col * w:(col + 1) * w] = c[y0:y1, x0:x1] * keep
        save(sheet, os.path.join(GFX, f"{name}-anim-{k}.png"))
        patches.append({"file": f"{name}-anim-{k}", "box": (x0, y0, x1, y1), "line_length": per_row,
                        "own": own})
    return still, patches


def rebuild(still, patches, f, frames, shape):
    """One frame of one cow, rebuilt from its still image and patches as the game layers them."""
    out = np.zeros(shape, np.float32)
    if still:
        x0, y0, x1, y1 = still["box"]
        out[y0:y1, x0:x1] = load(os.path.join(GFX, still["file"] + ".png"))
    for p in patches:
        x0, y0, x1, y1 = p["box"]
        layer = frames[f][y0:y1, x0:x1] * np.concatenate(
            [np.ones((y1 - y0, x1 - x0, 3)), p["own"][y0:y1, x0:x1, None]], -1)
        out[y0:y1, x0:x1] = over(layer, out[y0:y1, x0:x1])
    return out


def main():
    scene = gh.reset_scene()
    mats = gh.make_materials()
    pasture_materials(mats)
    if gh.GAME_LOOK:
        gh.apply_material_finishes(mats)
        gh.apply_game_look(mats)
    rng = random.Random("barn")
    coll = scene.collection
    root = bpy.data.objects.new("ROOT", None)
    coll.objects.link(root)
    model = bpy.data.objects.new("MODEL", None)
    coll.objects.link(model)
    model.parent = root

    ground_parts, fence, shed, props = pasture_parts(rng)
    ground = place(compose("barn-ground", ground_parts, mats), coll, model)
    statics = [place(compose(n, p, mats), coll, model) for n, p in (("fence", fence), ("shed", shed), ("props", props))]
    catcher = place(compose("catcher", [("cube", M((0, 0, -0.01), scale=(60, 60, 0.02)), "floor")], mats), coll, root)
    catcher.is_shadow_catcher = True
    catcher.hide_render = True

    tmp = tempfile.mkdtemp(prefix="barn-")
    root.scale = (1, gh.STRETCH, 1)
    cam = scene.camera
    top = TOP_H * math.sin(gh.THETA) + 0.25
    rect = (-SIZE / 2 - 0.4, SIZE / 2 + 0.4, -SIZE / 2 - 0.4, SIZE / 2 + top + 0.4)

    # 1. The paddock, empty: the barn entity's own picture.
    main_info = gh.frame_camera(cam, scene, *rect)
    gh.render(scene, os.path.join(GFX, "barn-base.png"))
    # 2. Shadow of the fence and shed.
    shadow_len = TOP_H / math.tan(gh.SUN_ELEVATION) + 0.3
    shadow_info = gh.frame_camera(cam, scene, -SIZE / 2 - 0.4, SIZE / 2 + 0.4 + shadow_len,
                                  -SIZE / 2 - 1.2, SIZE / 2 + 0.4)
    for o in statics + [ground]:
        o.visible_camera = False
    ground.visible_shadow = False
    catcher.hide_render = False
    gh.render(scene, os.path.join(GFX, "barn-shadow.png"))
    catcher.hide_render = True
    ground.visible_shadow = True
    for o in statics + [ground]:
        o.visible_camera = True

    # 3. Each animal alone over all frames, in two passes: its body (the ground not drawn),
    #    and its shadow (the animal not drawn, the ground only catching its shadow). Statics
    #    cut both out where they are in front and are not drawn, but still cast their shadows:
    #    an animal in the shade of the fence or shed is shaded, and its own shadow is only the
    #    extra darkening it adds to the ground (see static_light below).
    gh.frame_camera(cam, scene, *rect)
    for o in statics:
        o.is_holdout = True
    # Every pass samples each pixel the same way, so the ground away from the animal matches
    # exactly between renders with and without it.
    scene.cycles.use_adaptive_sampling = False
    # Keep the scene loaded between the hundreds of frames (only the animals change), which
    # saves re-syncing the whole paddock for every frame.
    scene.render.use_persistent_data = True
    placed = {}
    for kind in ONLY:
        spec = ANIMALS[kind]
        rigs, bodies = [], []
        for i, (x, y, yaw, pose, coat) in enumerate(spec["positions"]):
            me, joints = spec["build"](f"{kind}{i}", pose, coat, mats)
            body = place(me, coll, model)
            rig = spec["rig"](body, joints, f"{kind}{i}")
            rig.parent = model
            place_animal(rig, me, x, y, yaw, pose, spec.get("scale", ANIMAL_SCALE), spec.get("roll_lying", False))
            rigs.append(rig)
            bodies.append(body)
        placed[kind] = (rigs, bodies)
        for b in bodies:
            b.hide_render = True
    # How much light reaches the ground with only the statics in the way (shadow catcher:
    # alpha = 1 - light / unshadowed light). An animal's shadow is its frame's light relative
    # to this, so it never darkens ground the statics already shade a second time, and it
    # covers the lit gaps between their shadows (e.g. between fence rails) as it should.
    ground.visible_camera = True
    ground.is_shadow_catcher = True
    scene.render.use_border = False
    gh.render(scene, os.path.join(tmp, "_static-light.png"))
    static_light = np.maximum(1 - load(os.path.join(tmp, "_static-light.png"))[..., 3], 0.02)

    def own_shadow(paths):
        """Turns one animal's shadow catcher frames into the darkening it adds, without the
        sampling noise: a pixel that only flickers between frames gets its median over the
        loop, so only the shadows of the moving head, ears and tail change."""
        frames = [load(p) for p in paths]
        alpha = np.stack([np.clip(1 - (1 - a[..., 3]) / static_light, 0, 1) for a in frames])
        median = np.median(alpha, axis=0)
        alpha = np.where(np.abs(alpha - median) < 0.08, median, alpha)
        alpha[alpha < 0.02] = 0
        for a, al, p in zip(frames, alpha, paths):
            a[..., :3] = 0
            a[..., 3] = al
            save(a, p)

    for kind in ONLY:
        spec = ANIMALS[kind]
        rigs, bodies = placed[kind]
        for i, (rig, pos) in enumerate(zip(rigs, spec["positions"])):
            bodies[i].hide_render = False
            spec["pose"](rig, pos[3], 0.0, i)
            bpy.context.view_layer.update()
            render_region(scene, cam, bodies[i], lambda t: spec["pose"](rig, pos[3], t, i))
            for layer_kind in ("body", "shadow"):
                shadow_pass = layer_kind == "shadow"
                ground.visible_camera = shadow_pass
                ground.is_shadow_catcher = shadow_pass
                bodies[i].visible_camera = not shadow_pass
                paths = []
                for f in range(FRAMES):
                    spec["pose"](rig, pos[3], f / FRAMES, i)
                    bpy.context.view_layer.update()
                    paths.append(os.path.join(tmp, f"{kind}{i}-{layer_kind}-{f:02d}.png"))
                    gh.render(scene, paths[-1])
                if shadow_pass:
                    own_shadow(paths)
            bodies[i].visible_camera = True
            bodies[i].hide_render = True
            print(f"{kind.upper()} {i + 1}/{len(rigs)} rendered", flush=True)
    scene.render.use_border = False
    ground.visible_camera = True
    ground.is_shadow_catcher = False

    # 4. Icon: the paddock with all cows at rest, from a 3/4 view.
    if "cow" in placed:
        for o in statics:
            o.is_holdout = False
        rigs, bodies = placed["cow"]
        for i, (rig, pos) in enumerate(zip(rigs, COWS)):
            bodies[i].hide_render = False
            ANIMALS["cow"]["pose"](rig, pos[3], 0.0, i)
        root.scale = (1, 1, 1)
        cam.rotation_euler = (math.radians(55), 0, math.radians(-30))
        cam.data.ortho_scale = SIZE * 1.25
        d = cam.rotation_euler.to_matrix() @ V((0, 0, 1))
        cam.location = V((0, 0, 0.6)) + d * 60
        scene.render.resolution_x = scene.render.resolution_y = 64 * gh.SUPERSAMPLE
        gh.render(scene, os.path.join(GFX, "barn-icon.png"))

    # 5. Split every animal into a still image + patches, and check the result.
    H, W = main_info["height"], main_info["width"]
    if PREVIEW_DIR:
        os.makedirs(PREVIEW_DIR, exist_ok=True)
        base = load(os.path.join(GFX, "barn-base.png"))
        shadow = load(os.path.join(GFX, "barn-shadow.png"))
    for kind in ONLY:
        positions = ANIMALS[kind]["positions"]
        for old in os.listdir(GFX):
            if old.startswith(f"barn-{kind}-") or old.startswith("barn-anim-"):
                os.remove(os.path.join(GFX, old))
        out_all, all_frames = [], []
        worst, total, n_patches = 0.0, 0, 0
        for i in range(len(positions)):
            out = {"depth": positions[i][1]}
            frames_of = {}
            for layer_kind, suffix in (("body", ""), ("shadow", "-shadow")):
                frames = [load(os.path.join(tmp, f"{kind}{i}-{layer_kind}-{f:02d}.png")) for f in range(FRAMES)]
                frames_of[layer_kind] = frames
                still, patches = split_layer(frames, PATCHES_PER_ANIMAL, f"barn-{kind}-{i + 1}{suffix}")
                for f in range(FRAMES):
                    err = np.abs(rebuild(still, patches, f, frames, frames[f].shape) - frames[f])
                    err[..., :3] *= frames[f][..., 3:4]
                    worst = max(worst, float(err.max()))
                total += sum(area(p["box"]) for p in patches) * FRAMES * 4 + (area(still["box"]) * 4 if still else 0)
                n_patches += len(patches)
                out[layer_kind] = {"still": still, "patches": patches}
            all_frames.append(frames_of)
            out_all.append(out)
        print(f"RESULT {kind}: {len(positions)} positions, {n_patches} animated patches, {total / 2**20:.2f} MB "
              f"with shadows; rebuilt vs rendered max difference {worst * 255:.1f}/255")

        # Bodies are drawn north first; report where they overlap (the south one must be in front).
        alpha = [np.max([fr[..., 3] for fr in fo["body"]], axis=0) > 0.05 for fo in all_frames]
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                n = int((alpha[i] & alpha[j]).sum())
                if n:
                    front = i if positions[i][1] < positions[j][1] else j
                    print(f"RESULT {kind} {i + 1} and {kind} {j + 1} overlap by {n} px; {front + 1} (further south) in front")

        if PREVIEW_DIR:
            order = sorted(range(len(positions)), key=lambda k: -positions[k][1])
            for n in range(len(positions) + 1):
                shown = [k for k in order if k < n]
                for f in range(FRAMES) if n == len(positions) else (0,):
                    img = base.copy()
                    for layer_kind in ("shadow", "body"):            # all shadows, then all bodies
                        for k in shown:
                            c = out_all[k][layer_kind]
                            img = over(rebuild(c["still"], c["patches"], f, all_frames[k][layer_kind], img.shape), img)
                    write_preview(img, shadow, main_info, shadow_info,
                                  os.path.join(PREVIEW_DIR, f"barn-{kind}-{n}-{f:02d}.png"))
        write_animal_lua(kind, out_all, main_info, (W, H))

    write_paddock_lua(main_info, shadow_info)
    shutil.rmtree(tmp, ignore_errors=True)


def write_preview(sprite, shadow, info, shadow_info, path):
    """Sprite and its shadow over grass, laid out like the game does."""
    cw, ch = (SIZE + 8) * TILE_PX, (SIZE + 6) * TILE_PX
    canvas = np.zeros((ch, cw, 4), np.float32)
    canvas[...] = (0.28, 0.36, 0.17, 1)
    canvas[..., :3] *= np.random.default_rng(1).uniform(0.9, 1.1, size=(ch, cw))[..., None]
    cx, cy = cw / 2 - TILE_PX, ch / 2 + TILE_PX
    for img, inf, is_shadow in ((shadow, shadow_info, True), (sprite, info, False)):
        h, w = img.shape[:2]
        x0 = int(round(cx + inf["shift"][0] * TILE_PX - w / 2))
        y0 = int(round(cy + inf["shift"][1] * TILE_PX - h / 2))
        region = canvas[y0:y0 + h, x0:x0 + w]
        a = img[..., 3:4]
        if is_shadow:
            region[..., :3] *= 1 - a * 0.55
        else:
            region[..., :3] = region[..., :3] * (1 - a) + img[..., :3] * a
    save(canvas, path)


def write_paddock_lua(main_info, shadow_info):
    def entry(name, info):
        return '    %s = { width = %d, height = %d, shift = { %.4f, %.4f } },' % (
            name, info["width"], info["height"], info["shift"][0], info["shift"][1])
    lines = [
        "-- Generated by blender/render_barn.py. Do not edit by hand; re-run the script instead.",
        "return {",
        "    size = %d," % SIZE,
        "    frames = %d," % FRAMES,
        entry("base", main_info),
        entry("shadow", shadow_info),
        "    -- Animal kinds, each with its positions in prototypes/barn-sprites-<kind>.lua",
        "    animals = { %s }," % ", ".join('"%s"' % k for k in ANIMALS),
        "}",
    ]
    path = os.path.join(LUA_DIR, "barn-sprites.lua")
    with open(path, "w", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")
    print("Wrote", path)


def write_animal_lua(kind, animals, main_info, size_px):
    W, H = size_px

    def sprite(item, extra=""):
        x0, y0, x1, y1 = item["box"]
        sx = main_info["shift"][0] + ((x0 + x1) / 2 - W / 2) / TILE_PX
        sy = main_info["shift"][1] + ((y0 + y1) / 2 - H / 2) / TILE_PX
        return '{ file = "%s", width = %d, height = %d, shift = { %.4f, %.4f }%s }' % (
            item["file"], x1 - x0, y1 - y0, sx, sy, extra)

    lines = [
        "-- Generated by blender/render_barn.py. Do not edit by hand; re-run the script instead.",
        "-- %s positions in the order they appear as the barn fills up. depth: draw animals with a" % kind,
        "-- larger depth (further north) first, so animals further south are drawn in front.",
        "return {",
    ]
    for c in animals:
        lines.append("    {")
        lines.append("        depth = %.2f," % c["depth"])
        for layer_kind in ("body", "shadow"):
            part = c[layer_kind]
            lines.append("        %s = {" % layer_kind)
            if part["still"]:
                lines.append("            still = %s," % sprite(part["still"]))
            lines.append("            patches = {")
            for p in part["patches"]:
                lines.append("                %s," % sprite(p, ", line_length = %d" % p["line_length"]))
            lines.append("            },")
            lines.append("        },")
        lines.append("    },")
    lines.append("}")
    path = os.path.join(LUA_DIR, f"barn-sprites-{kind}.lua")
    with open(path, "w", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")
    print("Wrote", path)


if __name__ == "__main__":
    main()
