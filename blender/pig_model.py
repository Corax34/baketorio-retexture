"""Domestic pig built like the cow (cow_model.py): lofted cross-sections at real
proportions, fused into one surface with a voxel remesh and smoothed.

Ground at z=0, pig faces +x. Units: metres (1 tile = 1 m). Back ~0.86 m high,
snout to rump ~1.55 m. Poses: standing, eating (snout in a low feeder), rooting
(snout on the ground), lying (built standing-shaped; render_barn.py rolls it onto
its side with lying_transform()).
"""

import math

import bmesh
import bpy
from mathutils import Vector

from cow_model import _apply, _catmull, _lerp_stations, _to_object, loft

V = Vector

LYING_ROLL = math.radians(80)      # how far a lying pig is rolled onto its side


def head_path(pose):
    """Control points from the neck (inside the shoulders) to the tip of the snout. The face
    keeps its length in every pose (HEAD_PROFILE is measured from the tip), so lowering the
    head only changes how much neck shows."""
    if pose == "rooting":
        # Snout disc pushed into the soil: the short neck starts low in the shoulders and the
        # face points almost straight down, so no long neck shows.
        return [V((0.36, 0, 0.5)), V((0.5, 0, 0.42)), V((0.575, 0, 0.23)), V((0.65, 0, 0.04))]
    if pose == "eating":
        return [V((0.36, 0, 0.58)), V((0.54, 0, 0.5)), V((0.64, 0, 0.34)), V((0.72, 0, 0.21))]
    if pose == "lying":
        # Head stretched forward and turned slightly towards the ground side (-y).
        return [V((0.36, 0, 0.58)), V((0.57, -0.02, 0.58)), V((0.73, -0.06, 0.53)), V((0.84, -0.09, 0.48))]
    return [V((0.36, 0, 0.6)), V((0.56, 0, 0.62)), V((0.71, 0, 0.54)), V((0.82, 0, 0.48))]


# Head cross-sections (distance back from the snout tip in m, half_width, up, down,
# exponent): heavy jowls, then a straight wedge down to a narrow snout (the flat disc on
# its end is added in extras()). The neck section inside the shoulders comes first.
NECK_SECTION = (0.21, 0.21, 0.25, 2.3)
HEAD_PROFILE = [
    (0.35, 0.17, 0.15, 0.2, 2.3),      # jowls
    (0.25, 0.13, 0.115, 0.155, 2.2),   # cheeks and eyes
    (0.16, 0.1, 0.09, 0.115, 2.1),
    (0.07, 0.072, 0.066, 0.075, 2.1),
    (0.0, 0.05, 0.047, 0.044, 2.1),    # snout
]
EAR_BACK = 0.32      # distance of the ears back from the snout tip


def _dense(path):
    """The head's centre line (as lofted) with arc lengths measured back from the tip."""
    pts = _catmull(path, 80)
    back = [0.0]
    for a, b in zip(pts[::-1], pts[-2::-1]):
        back.append(back[-1] + (b - a).length)
    return pts, back[::-1]


def _at_back(pts, back, d):
    """(point, tangent) on the centre line at distance d back from the tip."""
    for i in range(len(pts) - 1):
        if back[i + 1] <= d:
            f = (back[i] - d) / max(back[i] - back[i + 1], 1e-6)
            return pts[i].lerp(pts[i + 1], f), (pts[i + 1] - pts[i]).normalized()
    return pts[-1], (pts[-1] - pts[-2]).normalized()


def body_parts(bm, pose):
    """Lofts the body, head, ears, legs and tail for a pose; returns the joint positions
    the skeleton (pig_rig.py) is built from."""
    # Trunk: (x, z_top, z_bottom, half_width). The back arches, highest over the loin, and
    # rounds off over the rump; the belly sags in the middle and tucks up at the flanks.
    trunk = [
        (-0.67, 0.66, 0.5, 0.07),    # around the tail root
        (-0.62, 0.76, 0.38, 0.17),
        (-0.5, 0.845, 0.33, 0.22),   # rump (the hams add their bulk below)
        (-0.32, 0.875, 0.3, 0.235),  # loin, flank tucked up in front of the hind legs
        (-0.1, 0.87, 0.235, 0.255),  # belly hanging lowest
        (0.12, 0.845, 0.25, 0.24),   # behind the shoulders
        (0.32, 0.83, 0.29, 0.235),   # shoulders
        (0.44, 0.7, 0.36, 0.19),     # where the neck begins: low enough that a lowered head
    ]                                # flows out of the shoulders without a fold
    # A smooth curve through the keys, so the outline has no corners between them.
    stations = []
    for x, top, bot, hw in _catmull([V(k) for k in trunk], 40):
        h = (top - bot) / 2
        stations.append((V((x, 0, (top + bot) / 2)), hw, h, h * 1.05, 2.2))
    loft(bm, stations, up_hint=V((0, 0, 1)))

    # Hams and shoulders: rounded masses on each side, fused with the trunk by the remesh.
    # Each ham is a teardrop from the top of the rump down into the hind leg (it is the thigh
    # muscle), bulging out sideways and back past the tail root, with a crease between the
    # two; the shoulders swell low over the front legs, below the line of the back.
    # Keys run downwards: (x, z, half_width, front, back).
    ham = [(-0.45, 0.76, 0.05, 0.08, 0.08), (-0.53, 0.63, 0.12, 0.17, 0.16), (-0.53, 0.48, 0.125, 0.15, 0.16),
           (-0.5, 0.34, 0.1, 0.1, 0.11), (-0.48, 0.22, 0.07, 0.065, 0.07)]
    shoulder = [(0.3, 0.7, 0.04, 0.08, 0.08), (0.3, 0.6, 0.1, 0.15, 0.15), (0.3, 0.47, 0.105, 0.14, 0.15),
                (0.31, 0.34, 0.08, 0.09, 0.1), (0.31, 0.24, 0.06, 0.07, 0.07)]
    for s in (-1, 1):
        for keys, y in ((ham, 0.12), (shoulder, 0.12)):
            dense = _catmull([V(k) for k in keys], 24)
            loft(bm, [(V((x, s * y, z)), hw, front, back, 2.1) for x, z, hw, front, back in dense],
                 up_hint=V((1, 0, 0)))

    # Head: hardly any neck, heavy jowls, then a straight wedge down to the snout.
    path = head_path(pose)
    pts, back = _dense(path)
    length = back[0]
    loft(bm, _lerp_stations(path, [(0.0, *NECK_SECTION)] + [
        (1 - d / length, *section) for d, *section in HEAD_PROFILE], samples=28), up_hint=V((0, 0, 1)))
    nose = path[-1]
    face = (path[-1] - path[-2]).normalized()

    # Ears: big, on top of the head just behind the eyes, pointing up and forward.
    ear_point, ear_face = _at_back(pts, back, EAR_BACK)
    side = ear_face.cross(V((0, 0, 1))).normalized()
    ear_centre = ear_point + side.cross(ear_face).normalized() * 0.12
    ears = []
    for s in (-1, 1):
        base = ear_centre + side * s * 0.075
        tip = base + side * s * 0.06 + V((0, 0, 0.1)) + ear_face * 0.11
        ears.append((base, tip))

    # Short legs: front ones straight, hind ones with the hock bending backwards.
    legs = []
    front_profile = [(0.0, 0.11, 0.11, 0.11, 2), (0.45, 0.07, 0.075, 0.075, 2),
                     (0.7, 0.055, 0.06, 0.06, 2), (1.0, 0.047, 0.052, 0.052, 2)]
    hind_profile = [(0.0, 0.1, 0.09, 0.12, 2), (0.45, 0.075, 0.08, 0.085, 2),   # thigh: in the ham
                    (0.7, 0.055, 0.06, 0.06, 2), (1.0, 0.047, 0.052, 0.052, 2)]
    for s in (-1, 1):
        front = [V((0.3, s * 0.15, 0.45)), V((0.31, s * 0.14, 0.24)), V((0.31, s * 0.14, 0.1)),
                 V((0.33, s * 0.14, 0.03))]
        hind = [V((-0.45, s * 0.13, 0.45)), V((-0.49, s * 0.14, 0.24)), V((-0.47, s * 0.14, 0.1)),
                V((-0.44, s * 0.14, 0.03))]
        loft(bm, _lerp_stations(front, front_profile), up_hint=V((1, 0, 0)))
        loft(bm, _lerp_stations(hind, hind_profile), up_hint=V((1, 0, 0)))
        legs += [front, hind]

    return {"path": path, "nose": nose, "face": face, "ears": ears, "ear_face": ear_face, "legs": legs,
            "tail": tail_path()}


TAIL_ROOT = V((-0.6, 0, 0.67))      # inside the rump, high up
TAIL_LEAD = 0.065                   # straight part from the root before it starts to curl


def tail_path():
    """Centre line of the curly tail: straight out of the rump, then a corkscrew whose coils
    open up over the first half turn and stay well apart, so it reads as a spiral and never
    touches the rump. Dense points, from the root inside the rump to the free tip."""
    axis = V((-1, 0, 0.12)).normalized()           # backwards, carried slightly up
    e1 = (V((0, 0, 1)) - axis * axis.z).normalized()
    e2 = axis.cross(e1)
    start = TAIL_ROOT + axis * TAIL_LEAD
    turns, length, radius = 1.75, 0.1, 0.024
    pts = [TAIL_ROOT, TAIL_ROOT.lerp(start, 0.5)]
    n = 48
    for i in range(n + 1):
        f = i / n
        a = 2 * math.pi * turns * f
        r = radius * min(1.0, a / math.pi) ** 0.7  # opens over the first half turn
        pts.append(start + axis * (length * f) + (e1 * math.cos(a) + e2 * math.sin(a)) * r)
    return pts


def extras(bm_dict, info):
    """Parts kept out of the body's remesh, which would blur or fuse them: the ears and the
    tail (too thin), the snout disc with its nostrils, and the trotters."""
    for base, tip in info["ears"]:
        # Starts inside the head so there is no gap where it joins.
        root = base + (base - tip).normalized() * 0.03
        loft(bm_dict["coat"], _lerp_stations([root, base, tip], [
            (0.0, 0.045, 0.016, 0.016, 2), (0.25, 0.055, 0.014, 0.014, 2),
            (0.55, 0.075, 0.012, 0.012, 2.2), (0.85, 0.05, 0.01, 0.01, 2.2), (1.0, 0.012, 0.006, 0.006, 2)]),
            up_hint=info["ear_face"])
    tail = info["tail"]
    total = sum((b - a).length for a, b in zip(tail, tail[1:]))
    stations, d = [], 0.0
    for i, c in enumerate(tail):
        if i:
            d += (c - tail[i - 1]).length
        r = 0.014 - 0.007 * d / total
        stations.append((c, r, r, r, 2))
    loft(bm_dict["coat"], stations, up_hint=V((0, 0, 1)), ring=12)
    # Snout: a flat oval disc, a little wider than the snout behind it, facing forward.
    nose, face = info["nose"], info["face"]
    loft(bm_dict["snout"], [(nose - face * 0.02, 0.049, 0.046, 0.042, 2.1),
                            (nose + face * 0.004, 0.055, 0.05, 0.047, 2.3),
                            (nose + face * 0.012, 0.052, 0.047, 0.044, 2.6)], up_hint=V((0, 0, 1)))
    side = face.cross(V((0, 0, 1))).normalized()
    up = side.cross(face).normalized()
    for s in (-1, 1):
        c = nose + face * 0.011 + side * s * 0.019 - up * 0.004
        loft(bm_dict["nostril"], [(c - face * 0.003, 0.008, 0.012, 0.012, 2),
                                  (c + face * 0.0018, 0.008, 0.012, 0.012, 2)], up_hint=up)
    for path in info["legs"]:
        foot = path[-1]
        loft(bm_dict["trotter"], [(foot + V((0.0, 0, 0.05)), 0.049, 0.054, 0.054, 2.2),
                                  (foot + V((0.01, 0, 0.0)), 0.054, 0.058, 0.056, 2.4),
                                  (V((foot.x + 0.03, foot.y, 0.0)), 0.056, 0.06, 0.056, 2.6)],
             up_hint=V((1, 0, 0)))


def build_pig_mesh(name, pose, coat_mat, mats, snout_mat):
    """Returns (mesh, joints): the fused, smoothed body plus the coloured extras, with each
    point's rest position stored as the "rest" attribute so coat patterns follow the skin."""
    coll = bpy.context.scene.collection
    bm = bmesh.new()
    info = body_parts(bm, pose)
    body = _to_object(bm, name + "-body", coll)
    _apply(body, "REMESH", mode="VOXEL", voxel_size=0.014)
    _apply(body, "SMOOTH", factor=0.8, iterations=4)
    _apply(body, "DECIMATE", ratio=0.35)
    bms = {k: bmesh.new() for k in ("coat", "snout", "nostril", "trotter")}
    extras(bms, info)
    pieces = [body]
    for key, mat in (("coat", coat_mat), ("snout", snout_mat), ("nostril", "nostril"), ("trotter", "trotter")):
        o = _to_object(bms[key], f"{name}-{key}", coll)
        o.data.materials.append(mats[mat])
        pieces.append(o)
    body.data.materials.append(mats[coat_mat])
    for o in pieces:
        for p in o.data.polygons:
            p.use_smooth = True
    with bpy.context.temp_override(active_object=body, selected_editable_objects=pieces):
        bpy.ops.object.join()
    me = body.data
    me.name = name
    coll.objects.unlink(body)
    bpy.data.objects.remove(body)
    rest = me.attributes.new(name="rest", type="FLOAT_VECTOR", domain="POINT")
    co = [0.0] * (len(me.vertices) * 3)
    me.vertices.foreach_get("co", co)
    rest.data.foreach_set("vector", co)
    return me, info


def lying_transform(me):
    """For a pig rolled LYING_ROLL onto its side: (sideways shift, lift) in pig units that
    put its body centre over the placement point and its lowest point on the ground."""
    import numpy as np
    co = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    s, c = math.sin(LYING_ROLL), math.cos(LYING_ROLL)
    y_rolled = co[:, 1] * c - co[:, 2] * s
    z_rolled = co[:, 1] * s + co[:, 2] * c
    return -float(y_rolled.mean()), -float(z_rolled.min())
