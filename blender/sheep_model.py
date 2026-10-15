"""Sheep built like the pig (pig_model.py): a woolly fleece of lofted cross-sections fused
with a voxel remesh, its surface displaced into lumpy wool, plus the bare head and thin
legs as smooth lofts that start inside the fleece.

Ground at z=0, sheep faces +x. Units: metres (1 tile = 1 m). Top of the fleece ~0.83 m,
muzzle to tail ~1.4 m. Poses: standing (head up), grazing (muzzle in the grass),
eating (muzzle in the hay trough), drinking (muzzle in the water trough) and lying
(on the belly with the legs folded, the way sheep rest; on its side a sheep is stuck).
"""

import math

import bmesh
import bpy
from mathutils import Vector

from cow_model import _apply, _catmull, _lerp_stations, _to_object, loft
from pig_model import _at_back, _dense

V = Vector

LYING_DROP = 0.32      # how far the body sinks when the sheep lies down on its folded legs


def head_path(pose):
    """Control points from the neck (inside the shoulder wool) to the tip of the muzzle.
    The face keeps its length in every pose (HEAD_PROFILE is measured from the tip), so
    lowering the head only changes how much neck shows."""
    if pose == "grazing":
        return [V((0.3, 0, 0.6)), V((0.47, 0, 0.42)), V((0.59, 0, 0.23)), V((0.66, 0, 0.04))]
    if pose == "eating":                  # muzzle on top of the hay, so the face still shows
        return [V((0.3, 0, 0.62)), V((0.52, 0, 0.78)), V((0.68, 0, 0.74)), V((0.8, 0, 0.66))]
    if pose == "drinking":                # over the water trough's rim (0.48) and down into it
        return [V((0.3, 0, 0.6)), V((0.5, 0, 0.64)), V((0.66, 0, 0.58)), V((0.75, 0, 0.44))]
    if pose == "lying":                                    # head up, chewing the cud
        d = LYING_DROP
        return [V((0.3, 0, 0.62 - d)), V((0.5, 0, 0.8 - d)), V((0.62, 0, 0.74 - d)), V((0.72, 0, 0.64 - d))]
    return [V((0.3, 0, 0.62)), V((0.52, 0, 0.79)), V((0.64, 0, 0.71)), V((0.74, 0, 0.6))]


# Bare head (distance back from the muzzle tip in m, half_width, up, down, exponent): a
# long face with a gently arched (Roman) nose, widest at the eyes, narrowing to the muzzle.
NECK_SECTION = (0.09, 0.09, 0.09, 2.0)           # inside the neck wool
HEAD_PROFILE = [
    (0.27, 0.058, 0.062, 0.07, 2.1),             # poll, under the wool
    (0.2, 0.066, 0.066, 0.088, 2.1),             # cheeks and eyes, the jaw deep below
    (0.13, 0.05, 0.058, 0.062, 2.1),             # nose bridge
    (0.06, 0.037, 0.046, 0.042, 2.1),
    (0.0, 0.028, 0.03, 0.033, 2.1),              # narrow muzzle
]
EAR_BACK = 0.23          # distance of the ears back from the muzzle tip
WOOL_TO = 0.24           # the neck wool reaches up to here (back from the tip): behind the ears


def body_parts(fleece, bare, tail_bm, pose, ram=False):
    """Lofts the fleece volume (into bm fleece), the woolly tail (into bm tail_bm) and the
    bare head and legs (into bm bare) for a pose; returns the joint positions the skeleton
    (sheep_rig.py) is built from."""
    lying = pose == "lying"
    drop = LYING_DROP if lying else 0.0
    # Fleece: (x, z_top, z_bottom, half_width). Broad and deep, the back nearly level, the
    # wool hanging lowest under the belly and rounding off over the rump and chest.
    trunk = [
        (-0.62, 0.64, 0.46, 0.1),     # rump end
        (-0.55, 0.77, 0.37, 0.22),
        (-0.4, 0.82, 0.34, 0.26),
        (-0.15, 0.83, 0.33, 0.27),    # belly wool hangs lowest
        (0.1, 0.82, 0.34, 0.26),
        (0.3, 0.79, 0.37, 0.24),      # shoulders
        (0.43, 0.71, 0.43, 0.17),     # chest, where the neck wool begins
    ]
    stations = []
    for x, top, bot, hw in _catmull([V(k) for k in trunk], 40):
        h = (top - bot) / 2
        stations.append((V((x, 0, (top + bot) / 2 - drop)), hw, h, h, 2.4))
    loft(fleece, stations, up_hint=V((0, 0, 1)))
    # Wool down the hind legs (britches), as a teardrop on each side.
    britch = [(-0.38, 0.62, 0.06, 0.07, 0.07), (-0.43, 0.5, 0.1, 0.1, 0.11), (-0.45, 0.36, 0.08, 0.07, 0.09),
              (-0.46, 0.28, 0.05, 0.05, 0.05)]
    for s in (-1, 1):
        dense = _catmull([V(k) for k in britch], 20)
        loft(fleece, [(V((x, s * 0.11, z - drop)), hw, front, back, 2.1) for x, z, hw, front, back in dense],
             up_hint=V((1, 0, 0)))
    # Woolly tail hanging down the rump, a hand's width clear of it so it can swish freely
    # (kept apart from the fleece, whose remesh would fuse the two).
    # It ends above the visible legs, so seen from behind it can't pass for a third leg.
    tail = [V((-0.57, 0, 0.62 - drop)), V((-0.64, 0, 0.6 - drop)), V((-0.685, 0, 0.54 - drop)),
            V((-0.695, 0, 0.5 - drop)), V((-0.69, 0, 0.47 - drop))]
    loft(tail_bm, _lerp_stations(tail, [(0, 0.042, 0.042, 0.042, 2), (0.3, 0.043, 0.043, 0.043, 2),
                                        (0.7, 0.035, 0.035, 0.035, 2), (1.0, 0.025, 0.025, 0.025, 2)],
                                 samples=20), up_hint=V((1, 0, 0)))

    # Head: the neck wool from the shoulders up to behind the ears, then the bare face.
    path = head_path(pose)
    pts, back = _dense(path)
    length = back[0]
    loft(bare, _lerp_stations(path, [(0.0, *NECK_SECTION)] + [
        (1 - d / length, *section) for d, *section in HEAD_PROFILE], samples=28), up_hint=V((0, 0, 1)))
    wool_end, _ = _at_back(pts, back, WOOL_TO)
    neck = [p for p, b in zip(pts, back) if b >= WOOL_TO] + [wool_end]
    loft(fleece, _lerp_stations(neck, [(0.0, 0.17, 0.17, 0.17, 2.2), (0.6, 0.13, 0.13, 0.14, 2.2),
                                      (1.0, 0.085, 0.09, 0.09, 2.1)], samples=16), up_hint=V((0, 0, 1)))
    nose = path[-1]
    face = (path[-1] - path[-2]).normalized()

    # Ears: broad leaves sticking out sideways from just behind the eyes, drooping outwards.
    ear_point, ear_face = _at_back(pts, back, EAR_BACK)
    side = ear_face.cross(V((0, 0, 1))).normalized()
    head_up = side.cross(ear_face).normalized()
    ears = []
    for s in (-1, 1):
        base = ear_point + head_up * 0.04 + side * s * 0.05
        tip = base + side * s * 0.085 - ear_face * 0.025 - head_up * 0.05
        ears.append((base, tip))

    # Thin legs, starting inside the fleece. Standing: front ones straight, hind ones with the
    # hock bending backwards. Lying: folded under the body, the knees and hooves showing.
    legs = []
    for s in (-1, 1):
        y = s * 0.1
        if lying:
            front = [V((0.27, y, 0.5 - drop)), V((0.2, y * 1.3, 0.06)), V((0.33, y * 1.2, 0.035)),
                     V((0.45, y, 0.03))]
            hind = [V((-0.42, y, 0.5 - drop)), V((-0.3, y * 1.9, 0.08)), V((-0.2, y * 2.1, 0.04)),
                    V((-0.08, y * 2.0, 0.03))]
        else:
            # Knee and hock about a third of the way up, the pasterns angled forward.
            front = [V((0.27, y, 0.55)), V((0.27, y, 0.32)), V((0.275, y, 0.09)), V((0.3, y, 0.03))]
            hind = [V((-0.4, y, 0.55)), V((-0.47, y, 0.3)), V((-0.455, y, 0.09)), V((-0.43, y, 0.03))]
        # Slender, thickening a little at the knee/hock and the fetlock.
        profile = [(0.0, 0.05, 0.05, 0.05, 2), (0.3, 0.032, 0.036, 0.036, 2), (0.42, 0.034, 0.04, 0.04, 2),
                   (0.55, 0.026, 0.029, 0.029, 2), (0.8, 0.023, 0.025, 0.025, 2), (0.9, 0.026, 0.028, 0.028, 2),
                   (1.0, 0.025, 0.027, 0.027, 2)]
        loft(bare, _lerp_stations(front, profile, samples=24), up_hint=V((1, 0, 0)))
        loft(bare, _lerp_stations(hind, profile, samples=24), up_hint=V((1, 0, 0)))
        legs += [front, hind]

    return {"path": path, "nose": nose, "face": face, "ears": ears, "ear_face": ear_face, "head_up": head_up,
            "ear_point": ear_point, "side": side, "ram": ram, "legs": legs,
            "tail": tail, "drop": drop, "lying": lying}


def extras(bm_dict, info):
    """Small parts with their own material, kept out of the remesh: ears, hooves and a
    ram's horns."""
    if info["ram"]:
        # Heavy horns growing from the top of the head, curling back, down and forward round
        # the ears in about one turn, drifting outwards as they go and tapering to the tip.
        face, side, up, top = info["ear_face"], info["side"], info["head_up"], info["ear_point"]
        for sgn in (-1, 1):
            out = side * sgn
            centre = top + up * 0.0 - face * 0.03 + out * 0.075
            pts, n = [], 28
            for k in range(n + 1):
                f = k / n
                a = math.radians(100) + f * math.radians(330)       # start on top, sweep backwards
                r = 0.085 - 0.04 * f
                pts.append(centre + (up * math.sin(a) + face * math.cos(a)) * r + out * (0.035 * f))
            pts[0] = top + up * 0.05 + out * 0.035                   # root on the head
            prof = [(0, 0.04, 0.04, 0.04, 2), (0.4, 0.032, 0.032, 0.032, 2), (0.8, 0.019, 0.019, 0.019, 2),
                    (1, 0.007, 0.007, 0.007, 2)]
            loft(bm_dict["horn"], _lerp_stations(pts, prof, samples=40), up_hint=out)
    for base, tip in info["ears"]:
        root = base + (base - tip).normalized() * 0.025          # starts inside the head
        # Flat leaf: wide along the face, thin along the head's up axis.
        loft(bm_dict["bare"], _lerp_stations([root, base, tip], [
            (0.0, 0.016, 0.008, 0.008, 2), (0.3, 0.026, 0.007, 0.007, 2), (0.6, 0.03, 0.006, 0.006, 2.2),
            (0.85, 0.022, 0.005, 0.005, 2.2), (1.0, 0.006, 0.004, 0.004, 2)]), up_hint=info["head_up"])
    for path in info["legs"]:
        foot = path[-1]
        d = (path[-1] - path[-2]).normalized()
        if info["lying"]:              # the hoof lies along the ground, pointing where the leg ends
            stations = [(foot - d * 0.035, 0.026, 0.028, 0.028, 2.2), (foot, 0.03, 0.03, 0.03, 2.4),
                        (foot + d * 0.025, 0.028, 0.026, 0.026, 2.6)]
            loft(bm_dict["hoof"], stations, up_hint=V((0, 0, 1)))
        else:
            loft(bm_dict["hoof"], [(foot + V((0, 0, 0.035)), 0.026, 0.028, 0.028, 2.2),
                                   (foot + V((0.005, 0, 0)), 0.03, 0.032, 0.03, 2.4),
                                   (V((foot.x + 0.02, foot.y, 0.0)), 0.029, 0.03, 0.028, 2.6)],
                 up_hint=V((1, 0, 0)))


def _wool(obj, scale=0.04, strength=0.03):
    """Turns a smooth woolly surface into lumpy wool: cloud noise pushes the surface in and
    out along its normals, then a light smoothing rounds the lumps."""
    name = f"wool-lumps-{scale}"
    tex = bpy.data.textures.get(name) or bpy.data.textures.new(name, "CLOUDS")
    tex.noise_scale = scale
    tex.noise_depth = 2
    tex.noise_type = "HARD_NOISE"          # creases between the locks
    _apply(obj, "DISPLACE", texture=tex, texture_coords="LOCAL", strength=strength, mid_level=0.5)
    _apply(obj, "SMOOTH", factor=0.6, iterations=2)


def build_sheep_mesh(name, pose, wool_mat, mats, skin_mat, ram=False):
    """Returns (mesh, joints): the woolly fleece, bare head, ears, legs and hooves joined
    into one mesh, with each point's rest position stored as the "rest" attribute so the
    wool shading follows the body."""
    coll = bpy.context.scene.collection
    fleece_bm, bare_bm, tail_bm = bmesh.new(), bmesh.new(), bmesh.new()
    info = body_parts(fleece_bm, bare_bm, tail_bm, pose, ram)
    fleece = _to_object(fleece_bm, name + "-fleece", coll)
    _apply(fleece, "REMESH", mode="VOXEL", voxel_size=0.012)
    _apply(fleece, "SMOOTH", factor=0.8, iterations=3)
    _wool(fleece)
    _apply(fleece, "DECIMATE", ratio=0.4)
    fleece.data.materials.append(mats[wool_mat])
    tail = _to_object(tail_bm, name + "-tail", coll)
    _apply(tail, "REMESH", mode="VOXEL", voxel_size=0.008)
    _apply(tail, "SMOOTH", factor=0.8, iterations=2)
    _wool(tail, scale=0.025, strength=0.015)
    _apply(tail, "DECIMATE", ratio=0.5)
    tail.data.materials.append(mats[wool_mat])
    bms = {"bare": bare_bm, "hoof": bmesh.new(), "horn": bmesh.new()}
    extras(bms, info)
    pieces = [fleece, tail]
    for key, mat in (("bare", skin_mat), ("hoof", "hoof"), ("horn", "ram_horn")):
        if len(bms[key].verts) == 0:
            bms[key].free()
            continue
        o = _to_object(bms[key], f"{name}-{key}", coll)
        o.data.materials.append(mats[mat])
        pieces.append(o)
    for o in pieces:
        for p in o.data.polygons:
            p.use_smooth = True
    with bpy.context.temp_override(active_object=fleece, selected_editable_objects=pieces):
        bpy.ops.object.join()
    me = fleece.data
    me.name = name
    coll.objects.unlink(fleece)
    bpy.data.objects.remove(fleece)
    rest = me.attributes.new(name="rest", type="FLOAT_VECTOR", domain="POINT")
    co = [0.0] * (len(me.vertices) * 3)
    me.vertices.foreach_get("co", co)
    rest.data.foreach_set("vector", co)
    return me, info
