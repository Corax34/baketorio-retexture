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
    head only changes how much neck shows. Like a real pig the head is carried low: the
    topline runs on from the shoulders over the poll straight down to the snout."""
    if pose == "rooting":
        # Snout disc pushed into the soil, the face pointing steeply down.
        return [V((0.34, 0, 0.52)), V((0.48, 0, 0.44)), V((0.62, 0, 0.24)), V((0.71, 0, 0.05))]
    if pose == "eating":
        return [V((0.34, 0, 0.58)), V((0.5, 0, 0.52)), V((0.66, 0, 0.36)), V((0.77, 0, 0.19))]
    if pose == "lying":
        # Head stretched forward and turned slightly towards the ground side (-y).
        return [V((0.34, 0, 0.58)), V((0.54, -0.02, 0.56)), V((0.72, -0.06, 0.5)), V((0.88, -0.09, 0.44))]
    return [V((0.34, 0, 0.6)), V((0.52, 0, 0.58)), V((0.7, 0, 0.47)), V((0.87, 0, 0.36))]


# Head cross-sections (distance back from the snout tip in m, half_width, up, down,
# exponent): deep and broad behind the ears with a flat poll, narrowing in a long straight
# wedge past the small eyes to the snout (the flat disc on its end is added in extras()).
# The neck section inside the shoulders comes first. Jowls and brows are added on top.
NECK_SECTION = (0.22, 0.22, 0.26, 2.3)
HEAD_PROFILE = [
    (0.44, 0.2, 0.19, 0.22, 2.4),      # behind the ears, merging into neck and chest
    (0.34, 0.172, 0.15, 0.165, 2.7),   # poll between the ears: broad and flat on top
    (0.22, 0.132, 0.11, 0.118, 2.4),   # eyes
    (0.12, 0.09, 0.078, 0.082, 2.2),   # long straight face
    (0.04, 0.062, 0.058, 0.058, 2.1),
    (0.0, 0.058, 0.054, 0.054, 2.1),   # snout
]
EAR_BACK = 0.33      # distance of the ears back from the snout tip
EYE_BACK = 0.215     # distance of the eyes back from the snout tip
EAR_LEN = 0.22       # Large White ears: big, thin, pointed, standing up and leaning forward
EAR_HALF_WIDTH = 0.1


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


def body_parts(bm, pose, boar=False):
    """Lofts the body, head, ears, legs and tail for a pose; returns the joint positions
    the skeleton (pig_rig.py) is built from. A boar has a higher, broader shield of
    shoulders, heavier jowls and tusks."""
    # Trunk: (x, z_top, z_bottom, half_width). The back arches, highest over the loin, and
    # rounds off over the rump; the belly sags in the middle and tucks up at the flanks.
    trunk = [
        (-0.67, 0.66, 0.5, 0.07),    # around the tail root
        (-0.62, 0.76, 0.38, 0.17),
        (-0.5, 0.845, 0.33, 0.22),   # rump (the hams add their bulk below)
        (-0.32, 0.875, 0.3, 0.235),  # loin, flank tucked up in front of the hind legs
        (-0.1, 0.87, 0.235, 0.255),  # belly hanging lowest
        (0.12, 0.845, 0.235, 0.24),  # behind the shoulders
        (0.32, 0.83, 0.25, 0.235),   # shoulders, the chest deep between the front legs
        (0.44, 0.7, 0.36, 0.19),     # where the neck begins: low enough that a lowered head
    ]                                # flows out of the shoulders without a fold
    if boar:   # higher, broader shoulders and neck: the boar's shield (x: extra top, extra width)
        more = {0.12: (0.02, 0.01), 0.32: (0.045, 0.025), 0.44: (0.05, 0.02)}
        trunk = [(x, top + more.get(x, (0, 0))[0], bot, hw + more.get(x, (0, 0))[1]) for x, top, bot, hw in trunk]
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
    if boar:   # the shoulder shield of a boar
        shoulder = [(x, z, hw * 1.25, f * 1.15, b * 1.15) for x, z, hw, f, b in shoulder]
    for s in (-1, 1):
        for keys, y in ((ham, 0.12), (shoulder, 0.12)):
            dense = _catmull([V(k) for k in keys], 24)
            loft(bm, [(V((x, s * y, z)), hw, front, back, 2.1) for x, z, hw, front, back in dense],
                 up_hint=V((1, 0, 0)))

    # Head: hardly any neck, heavy jowls, then a straight wedge down to the snout.
    path = head_path(pose)
    pts, back = _dense(path)
    length = back[0]
    jowl = 1.15 if boar else 1.0
    loft(bm, _lerp_stations(path, [(0.0, *NECK_SECTION)] + [
        (1 - d / length, hw * (jowl if d > 0.2 else 1), up, down * (jowl if d > 0.2 else 1), ex)
        for d, hw, up, down, ex in HEAD_PROFILE], samples=28), up_hint=V((0, 0, 1)))
    nose = path[-1]
    face = (path[-1] - path[-2]).normalized()

    # Heavy jowls: rounded masses low on each cheek, so the head is broadest at the bottom
    # behind the eyes and runs smoothly into the throat and chest.
    jowl_point, jowl_face = _at_back(pts, back, 0.3)
    side = jowl_face.cross(V((0, 0, 1))).normalized()
    jowl_up = side.cross(jowl_face).normalized()
    big = 1.15 if boar else 1.0
    for s in (-1, 1):
        c = jowl_point + side * s * 0.085 * big - jowl_up * 0.075
        loft(bm, [(c - jowl_face * 0.09, 0.05, 0.05, 0.05, 2), (c - jowl_face * 0.02, 0.085 * big, 0.08, 0.1 * big, 2),
                  (c + jowl_face * 0.06, 0.06, 0.055, 0.07, 2), (c + jowl_face * 0.11, 0.03, 0.03, 0.03, 2)],
             up_hint=jowl_up)

    # Small eyes high on the sides of the face, each under a fleshy brow.
    eye_point, eye_face = _at_back(pts, back, EYE_BACK)
    eye_side = eye_face.cross(V((0, 0, 1))).normalized()
    eye_up = eye_side.cross(eye_face).normalized()
    eyes = []
    for s in (-1, 1):
        eye = eye_point + eye_side * s * 0.118 + eye_up * 0.045
        eyes.append((eye, eye_side * s))
        brow = eye + eye_up * 0.018 - eye_side * s * 0.012 - eye_face * 0.005
        loft(bm, [(brow - eye_face * 0.025, 0.012, 0.012, 0.012, 2), (brow, 0.022, 0.016, 0.014, 2),
                  (brow + eye_face * 0.025, 0.012, 0.01, 0.01, 2)], up_hint=eye_up)

    # Ears: big pointed sheets on top of the head behind the eyes, standing up, splayed out
    # and tipped forward, their cupped fronts facing forward (built in extras()).
    ear_point, ear_face = _at_back(pts, back, EAR_BACK)
    ear_side = ear_face.cross(V((0, 0, 1))).normalized()
    head_up = ear_side.cross(ear_face).normalized()
    ears, ear_frames = [], []
    for s in (-1, 1):
        base = ear_point + head_up * 0.115 + ear_side * s * 0.08
        # Up and leaning forward over the face (more so when the pig lowers its head), splayed
        # a little outwards, their broad hollow fronts facing forward and outward.
        e1 = (head_up * 0.9 - ear_face * 0.12 + ear_side * s * 0.33).normalized()
        front = (ear_face * 0.7 + ear_side * s * 0.7).normalized()
        n = (front - e1 * front.dot(e1)).normalized()
        e2 = e1.cross(n).normalized()
        ears.append((base, base + e1 * EAR_LEN))
        ear_frames.append((base, e1, e2, n))

    # Short legs: front ones straight, hind ones with the hock bending backwards.
    legs = []
    front_profile = [(0.0, 0.12, 0.12, 0.12, 2), (0.45, 0.08, 0.085, 0.085, 2),   # muscular forearm
                     (0.7, 0.06, 0.066, 0.066, 2), (1.0, 0.05, 0.055, 0.055, 2)]
    hind_profile = [(0.0, 0.1, 0.09, 0.12, 2), (0.45, 0.082, 0.088, 0.092, 2),   # thigh: in the ham
                    (0.7, 0.06, 0.066, 0.066, 2), (1.0, 0.05, 0.055, 0.055, 2)]
    for s in (-1, 1):
        front = [V((0.3, s * 0.15, 0.45)), V((0.31, s * 0.14, 0.24)), V((0.31, s * 0.14, 0.1)),
                 V((0.33, s * 0.14, 0.03))]
        hind = [V((-0.45, s * 0.13, 0.45)), V((-0.49, s * 0.14, 0.24)), V((-0.47, s * 0.14, 0.1)),
                V((-0.44, s * 0.14, 0.03))]
        loft(bm, _lerp_stations(front, front_profile), up_hint=V((1, 0, 0)))
        loft(bm, _lerp_stations(hind, hind_profile), up_hint=V((1, 0, 0)))
        legs += [front, hind]

    return {"path": path, "nose": nose, "face": face, "ears": ears, "ear_frames": ear_frames, "ear_face": ear_face,
            "eyes": eyes, "legs": legs, "tail": tail_path(), "boar": boar}


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


def _ear_sheet(bm, base, e1, e2, n):
    """A pig's ear as a curved surface (thickened later): pointed, broadest near the base,
    cupped so its front (n) is hollow, the tip bending forward. e1 runs along the ear, e2
    across it. It starts a little inside the head so there is no gap where it joins."""
    nu, nv = 14, 10
    rows = []
    for i in range(nu + 1):
        u = -0.15 + 1.15 * i / nu
        uc = max(u, 0.0)
        w = max(EAR_HALF_WIDTH * (1 - uc) ** 0.85 * (1 + 0.6 * uc * (1 - uc)), 0.002)
        centre = base + e1 * (EAR_LEN * u) + n * (0.22 * EAR_LEN * uc * uc)
        row = []
        for j in range(nv + 1):
            v = -1 + 2 * j / nv
            row.append(bm.verts.new(centre + e2 * (w * v) - n * (0.16 * w * (1 - v * v))))
        rows.append(row)
    for i in range(nu):
        for j in range(nv):
            bm.faces.new((rows[i][j], rows[i][j + 1], rows[i + 1][j + 1], rows[i + 1][j]))


def extras(bm_dict, info):
    """Parts kept out of the body's remesh, which would blur or fuse them: the ears and the
    tail (too thin), the snout disc with its nostrils, and the trotters."""
    for base, e1, e2, n in info["ear_frames"]:
        _ear_sheet(bm_dict["ears"], base, e1, e2, n)
    for eye, out in info["eyes"]:
        loft(bm_dict["eye"], [(eye - out * 0.012, 0.009, 0.009, 0.009, 2), (eye - out * 0.002, 0.012, 0.011, 0.011, 2),
                              (eye + out * 0.004, 0.007, 0.006, 0.006, 2)], up_hint=V((0, 0, 1)))
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
    loft(bm_dict["snout"], [(nose - face * 0.02, 0.057, 0.053, 0.053, 2.1),
                            (nose + face * 0.004, 0.063, 0.057, 0.057, 2.3),
                            (nose + face * 0.012, 0.06, 0.054, 0.054, 2.6)], up_hint=V((0, 0, 1)))
    side = face.cross(V((0, 0, 1))).normalized()
    up = side.cross(face).normalized()
    for s in (-1, 1):
        c = nose + face * 0.011 + side * s * 0.022 - up * 0.008
        loft(bm_dict["nostril"], [(c - face * 0.003, 0.009, 0.014, 0.014, 2),
                                  (c + face * 0.0018, 0.009, 0.014, 0.014, 2)], up_hint=up)
    if info["boar"]:
        # Tusks: curving up out of the lower jaw on each side of the snout.
        up = side.cross(face).normalized()
        for s in (-1, 1):
            # From the corner of the mouth, curving out sideways and up past the snout.
            base = nose - face * 0.1 - up * 0.05 + side * s * 0.055
            tusk = [base, base + side * s * 0.035 + up * 0.02 + face * 0.01,
                    base + side * s * 0.05 + up * 0.06 - face * 0.01, base + side * s * 0.045 + up * 0.09 - face * 0.04]
            loft(bm_dict["tusk"], _lerp_stations(tusk, [(0, 0.015, 0.015, 0.015, 2), (0.5, 0.012, 0.012, 0.012, 2),
                                                        (1, 0.003, 0.003, 0.003, 2)], samples=12), up_hint=face)
    for path in info["legs"]:
        # Cloven trotter: two toes with a gap between them, and two small dew claws behind.
        foot = V((path[-1].x, path[-1].y, 0.0))
        for s in (-1, 1):
            toe = foot + V((0.012, s * 0.023, 0))
            loft(bm_dict["trotter"], [(toe + V((-0.012, 0, 0.07)), 0.022, 0.026, 0.026, 2.2),
                                      (toe + V((0.0, 0, 0.024)), 0.024, 0.03, 0.024, 2.4),
                                      (toe + V((0.028, 0, 0.012)), 0.018, 0.02, 0.012, 2.2),
                                      (toe + V((0.04, 0, 0.008)), 0.008, 0.008, 0.008, 2)], up_hint=V((1, 0, 0)))
            claw = foot + V((-0.038, s * 0.03, 0.05))
            loft(bm_dict["trotter"], [(claw + V((0.012, 0, 0.01)), 0.008, 0.008, 0.008, 2),
                                      (claw, 0.009, 0.01, 0.01, 2), (claw + V((-0.01, 0, -0.012)), 0.004, 0.004, 0.004, 2)],
                 up_hint=V((1, 0, 0)))


def build_pig_mesh(name, pose, coat_mat, mats, snout_mat, ear_mat=None, boar=False, trotter_mat="trotter"):
    """Returns (mesh, joints): the fused, smoothed body plus the coloured extras, with each
    point's rest position stored as the "rest" attribute so coat patterns follow the skin."""
    coll = bpy.context.scene.collection
    bm = bmesh.new()
    info = body_parts(bm, pose, boar)
    body = _to_object(bm, name + "-body", coll)
    _apply(body, "REMESH", mode="VOXEL", voxel_size=0.014)
    _apply(body, "SMOOTH", factor=0.8, iterations=4)
    _apply(body, "DECIMATE", ratio=0.35)
    bms = {k: bmesh.new() for k in ("coat", "ears", "eye", "snout", "nostril", "trotter", "tusk")}
    extras(bms, info)
    pieces = [body]
    for key, mat in (("coat", coat_mat), ("ears", ear_mat or coat_mat), ("eye", "eye_black"), ("snout", snout_mat),
                     ("nostril", "nostril"), ("trotter", trotter_mat), ("tusk", "tusk")):
        if len(bms[key].verts) == 0:
            bms[key].free()
            continue
        o = _to_object(bms[key], f"{name}-{key}", coll)
        if key == "ears":                                   # thin, but with some substance
            _apply(o, "SOLIDIFY", thickness=0.008, offset=0.0)
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
