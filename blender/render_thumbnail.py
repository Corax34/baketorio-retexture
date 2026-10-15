"""Renders thumbnail.png (144x144, for the mod portal and the in-game mod list): the barn
paddock from a 3/4 view with a mix of its animals, a hen in mid-flap among them.

    blender -b --python blender/render_thumbnail.py
"""

import math
import os
import random
import sys

import bpy
from mathutils import Vector

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.argv = [sys.argv[0]]                 # render_barn reads its own arguments
import render_barn as rb  # noqa: E402

gh = rb.gh
OUT = os.path.join(os.path.dirname(HERE), "thumbnail.png")
SIZE_PX = 144

# (kind, (x, y, yaw, pose, coat), loop phase): spots chosen so that no two animals overlap.
CAST = [
    ("cow", rb.COWS[0], 0.0),                                   # at the hay rack
    ("cow", rb.COWS[1], 0.0),                                   # at the water trough
    ("pig", rb.PIGS[0], 0.0),                                   # at the feeder
    ("pig", rb.PIGS[3], 0.0),                                   # the black pig
    ("sheep", rb.SHEEP[5], 0.0),
    ("chicken", (-0.6, 0.0, 25, "flapping", "hen_white"), 0.2),  # wings out
    ("chicken", (-1.6, 0.6, 160, "scratching", "hen_brown"), 0.0),
]


def main():
    scene = gh.reset_scene()
    mats = gh.make_materials()
    rb.pasture_materials(mats)
    gh.apply_material_finishes(mats)
    gh.apply_game_look(mats)
    coll = scene.collection
    model = bpy.data.objects.new("MODEL", None)
    coll.objects.link(model)
    ground, fence, shed, props = rb.pasture_parts(random.Random("barn"))
    for name, parts in (("ground", ground), ("fence", fence), ("shed", shed), ("props", props)):
        gh.place(gh.compose(name, parts, mats), coll, model)
    for i, (kind, (x, y, yaw, pose, coat), t) in enumerate(CAST):
        spec = rb.ANIMALS[kind]
        me, info = spec["build"](f"{kind}{i}", pose, coat, mats)
        body = gh.place(me, coll, model)
        rig = spec["rig"](body, info, f"{kind}{i}")
        rig.parent = model
        rb.place_animal(rig, me, x, y, yaw, pose, spec.get("scale", rb.ANIMAL_SCALE), spec.get("roll_lying", False))
        spec["pose"](rig, pose, t, 0)
    cam = scene.camera
    cam.rotation_euler = (math.radians(55), 0, math.radians(-30))
    cam.data.ortho_scale = rb.SIZE * 1.18
    d = cam.rotation_euler.to_matrix() @ Vector((0, 0, 1))
    cam.location = Vector((0, 0, 0.55)) + d * 60
    scene.render.resolution_x = scene.render.resolution_y = SIZE_PX * gh.SUPERSAMPLE
    bpy.context.view_layer.update()
    gh.render(scene, OUT)


if __name__ == "__main__":
    main()
