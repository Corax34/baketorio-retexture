"""Hen built like the other animals: the feathered body (breast, back, feathered thighs,
neck with its hackles and the head) lofted from cross-sections and fused with a voxel
remesh; wings, tail, comb, wattles, beak, eyes, legs and toes added as separate parts so
they stay crisp. Each wing is a fan of real feathers (flight feathers and coverts) on its
own bones, so it can lift and spread; a flapping hen also gets loose feathers that the
skeleton throws off and lets flutter down (chicken_rig.py).

Ground at z=0, hen faces +x. Units: metres (1 tile = 1 m). About 0.42 m to the top of the
comb, 0.42 m from tail to beak. Poses: standing (head up, alert), flapping (standing,
bursts of wing flapping), pecking (beak at the ground, the body tipped forward), feeding
(beak in the low feeder), scratching (looking down at the ground it scratches), sitting
(settled on the ground, legs hidden) and brooding (sitting on a straw nest of eggs).

Feather sheets carry UVs (u along the shaft, v across the vane) for the feather shader, the
body a "zones" colour attribute (hackle, saddle, breast) for the plumage colours, and every
point a "part" number telling the rig which bone a separate feather belongs to.
"""

import math
import random

import bmesh
import bpy
from mathutils import Matrix, Vector

from cow_model import _apply, _catmull, _lerp_stations, _to_object, loft

V = Vector

SIT_DROP = 0.12        # how far the body settles when the hen sits down
HIP = V((0.0, 0, 0.15))  # the body tips forward about this point when the hen pecks
LOW = 0.025              # the body sits this much lower than the keys below say

# Wing: flight feathers (0 = innermost secondary ... last = outer primary), folded along the
# side and fanned slightly. (folded angle below the back line in degrees, length, half width)
WING_FEATHERS = [(-1, 0.14, 0.022), (0.5, 0.145, 0.022), (2, 0.15, 0.022), (3.5, 0.155, 0.021),
                 (5, 0.17, 0.019), (6.5, 0.178, 0.018), (8, 0.185, 0.017)]
ROOT_STEP = 0.007      # spacing of the feather roots along the arm, folded
PARTICLES = 8          # loose feathers a flapping hen throws off per loop
PART_WING, PART_SHOULDER, PART_PARTICLE = 1000, 1050, 2000   # "part" numbers (see rig)


def head_pose(pose):
    """(neck base, neck bend, head centre, beak direction, body tilt in degrees) for a pose.
    The neck base is on the upright body (it tips and settles with it); the rest is where it
    ends up. A positive tilt tips the body forward, tail up."""
    if pose == "pecking":                 # body tipped well forward, the neck nearly straight
        return V((0.07, 0, 0.29)), V((0.165, 0, 0.15)), V((0.2, 0, 0.065)), V((0.3, 0, -1)), 28
    if pose == "feeding":                 # breast at the feeder: reaching over its rim, head down
        return V((0.07, 0, 0.29)), V((0.175, 0, 0.245)), V((0.235, 0, 0.175)), V((0.3, 0, -1)), 20
    if pose == "scratching":                               # looking down at where it scratches
        return V((0.07, 0, 0.29)), V((0.13, 0, 0.31)), V((0.17, 0, 0.28)), V((0.7, 0, -1)), 8
    if pose in ("sitting", "brooding"):
        return V((0.07, 0, 0.29)), V((0.11, 0, 0.34 - SIT_DROP)), V((0.13, 0, 0.37 - SIT_DROP)), V((1, 0, -0.1)), 0
    if pose == "flapping":                # standing tall, head up, the body leaning back a little
        return V((0.07, 0, 0.29)), V((0.1, 0, 0.36)), V((0.115, 0, 0.4)), V((1, 0, 0.05)), -6
    return V((0.07, 0, 0.29)), V((0.11, 0, 0.35)), V((0.13, 0, 0.385)), V((1, 0, -0.08)), 0


def body_parts(feathers, pose):
    """Lofts the feathered volume into bm feathers; returns what the extras and the
    skeleton (chicken_rig.py) need."""
    sitting = pose in ("sitting", "brooding")
    drop = SIT_DROP if sitting else 0.0
    neck_base, neck_bend, head, face, tilt = head_pose(pose)
    face = face.normalized()
    # Tipping forward: everything of the body turns about the hip; the head is placed as given.
    rot = Matrix.Rotation(math.radians(tilt), 3, "Y")

    def body(p):
        return HIP + rot @ (V(p) - V((0, 0, LOW)) - HIP) - V((0, 0, drop))

    # Body: (x, z centre, half_width, up, down), plump, the breast round and low in front,
    # the back rising towards the tail. Sitting, it spreads a little wider.
    spread = 1.12 if sitting else 1.0
    keys = [(-0.17, 0.275, 0.035, 0.045, 0.045), (-0.13, 0.255, 0.08, 0.075, 0.085),
            (-0.05, 0.235, 0.1 * spread, 0.09, 0.1), (0.04, 0.23, 0.095 * spread, 0.085, 0.1),
            (0.1, 0.24, 0.075, 0.07, 0.085), (0.135, 0.26, 0.035, 0.045, 0.05)]
    dense = _catmull([V(k) for k in keys], 30)
    loft(feathers, [(body((x, 0, z)), hw, up, down, 2.0) for x, z, hw, up, down in dense],
         up_hint=rot @ V((0, 0, 1)))
    # Shoulders under the wings, so the folded wings lie on a rounded body.
    for s in (-1, 1):
        sh = [(0.07, 0.065, 0.27, 0.02, 0.03, 0.03), (0.0, 0.075, 0.262, 0.03, 0.045, 0.045),
              (-0.08, 0.07, 0.258, 0.025, 0.04, 0.04)]
        loft(feathers, [(body((x, s * y * spread, z)), hw, up, down, 2.0) for x, y, z, hw, up, down in sh],
             up_hint=rot @ V((0, 0, 1)))
    # Rump the tail feathers grow from (the feathers themselves are added in extras()).
    rump = [body(p) for p in ((-0.08, 0, 0.27), (-0.13, 0, 0.3), (-0.15, 0, 0.32))]
    loft(feathers, _lerp_stations(rump, [(0, 0.045, 0.05, 0.05, 2), (1.0, 0.025, 0.03, 0.03, 2)], samples=8),
         up_hint=V((1, 0, 0)))
    tail = [body((-0.13, 0, 0.3)), body((-0.18, 0, 0.37))]
    # Feathered thighs down to the hocks (standing hens only; a sitting hen tucks them in).
    hips = []
    for s in (-1, 1):
        hip = body((0.0, s * 0.05, 0.2))
        hock = V((0.005, s * 0.045, 0.11))
        hips.append((hip, hock))
        if not sitting:
            loft(feathers, _lerp_stations([hip, hip.lerp(hock, 0.5), hock], [
                (0, 0.035, 0.035, 0.035, 2), (0.6, 0.026, 0.026, 0.026, 2), (1.0, 0.014, 0.014, 0.014, 2)],
                samples=10), up_hint=V((1, 0, 0)))
    # Neck: thick with hackle feathers, flaring into the shoulders and nearly as wide as the
    # head where it meets it, so it reads as one feathered cone rather than a thin tube.
    neck_base = body(neck_base)
    neck = [neck_base, neck_bend, head]
    loft(feathers, _lerp_stations(neck, [(0, 0.062, 0.062, 0.064, 2), (0.45, 0.046, 0.046, 0.05, 2),
                                         (0.8, 0.036, 0.036, 0.038, 2), (1.0, 0.03, 0.03, 0.03, 2)],
                                  samples=16), up_hint=V((0, 0, 1)))
    # Hackle cape: the neck feathers flowing back over the shoulders, so the back runs
    # smoothly into the neck in every pose instead of leaving a notch behind it.
    cape_start = body((-0.03, 0, 0.29))
    cape = [cape_start, neck_base, neck_base.lerp(neck_bend, 0.55)]
    loft(feathers, _lerp_stations(cape, [(0, 0.06, 0.04, 0.06, 2), (0.5, 0.066, 0.06, 0.064, 2),
                                         (1.0, 0.048, 0.047, 0.05, 2)], samples=12), up_hint=V((0, 0, 1)))
    # Head: small and round, a little longer than wide, along the beak's direction.
    side = face.cross(V((0, 0, 1))).normalized()
    head_up = side.cross(face).normalized()
    loft(feathers, [(head - face * 0.03, 0.012, 0.012, 0.012, 2), (head - face * 0.018, 0.024, 0.026, 0.024, 2),
                    (head + face * 0.005, 0.026, 0.028, 0.026, 2), (head + face * 0.026, 0.017, 0.018, 0.016, 2),
                    (head + face * 0.036, 0.008, 0.008, 0.008, 2)], up_hint=head_up)

    # Wings: folded against the sides, hinged at the shoulder. Each wing's frame: back (along
    # the folded feathers), down (along the folded arm, and outwards once the wing is raised)
    # and out (away from the body).
    up_b = rot @ V((0, 0, 1))
    wings = []
    for s in (-1, 1):
        pivot = body((0.06, s * 0.088 * spread, 0.283))
        back = (rot @ V((-1, 0, -0.04))).normalized()
        out = V((0, s, 0))
        out = (out - back * out.dot(back)).normalized()
        down = out.cross(back).normalized()
        if down.dot(up_b) > 0:
            down = -down
        roots, dirs = [], []
        for i, (ang, length, hw) in enumerate(WING_FEATHERS):
            a = math.radians(ang)
            roots.append(pivot + down * (ROOT_STEP * i) + back * (0.004 * i))
            dirs.append((back * math.cos(a) + down * math.sin(a)).normalized())
        wings.append({"side": s, "pivot": pivot, "back": back, "down": down, "out": out,
                      "forward": (rot @ V((1, 0, 0))).normalized(), "roots": roots, "dirs": dirs})
    return {"pose": pose, "neck_base": neck_base, "neck_bend": neck_bend, "head": head, "face": face, "side": side,
            "head_up": head_up, "tail": tail, "hips": hips, "sitting": sitting, "drop": drop, "tilt": tilt,
            "rot": rot, "tail_root": body((-0.12, 0, 0.29)), "cape_start": cape_start, "wings": wings,
            "upright": lambda p: HIP + rot.transposed() @ (V(p) + V((0, 0, drop)) - HIP) + V((0, 0, LOW))}


def _layers(bm):
    """The part number and UV layers of a feather bmesh."""
    part = bm.verts.layers.int.get("part") or bm.verts.layers.int.new("part")
    return part, bm.loops.layers.uv.verify()


def feather(bm, part_no, root, along, across, normal, length, half_width, bow=0.0, tuck=0.0, curl=0.15,
            nu=10, nv=4):
    """One feather as a curved surface (thickened later): narrow at the quill, widest a
    third of the way out, rounded at the tip. bow pushes its middle out along normal, tuck
    pulls the tip back in, curl cups it across. UVs: u along the shaft, v across the vane."""
    part, uv = _layers(bm)
    rows, uvs = [], {}
    for i in range(nu + 1):
        u = i / nu
        w = half_width * (0.3 + 0.7 * min(1.0, u / 0.3)) * math.sqrt(max(0.0, 1 - max(0.0, (u - 0.68) / 0.32) ** 2))
        w = max(w, 0.0006)
        centre = root + along * (length * u) + normal * (bow * math.sin(math.pi * u) - tuck * u * u)
        row = []
        for j in range(nv + 1):
            v = -1 + 2 * j / nv
            vert = bm.verts.new(centre + across * (w * v) - normal * (curl * w * (1 - v * v)))
            vert[part] = part_no
            uvs[vert] = (u, (v + 1) / 2)
            row.append(vert)
        rows.append(row)
    for i in range(nu):
        for j in range(nv):
            f = bm.faces.new((rows[i][j], rows[i][j + 1], rows[i + 1][j + 1], rows[i + 1][j]))
            for loop in f.loops:
                loop[uv].uv = uvs[loop.vert]


def wing_feathers(bm, info):
    """Both wings: flight feathers fanned slightly, primaries innermost (tucked under the
    secondaries), each covered at its root by a short covert, and a patch of small coverts
    over the shoulder. Folded, they bow out over the body and their tips tuck in near the
    tail."""
    n = len(WING_FEATHERS)
    for w in info["wings"]:
        sidecode = 0 if w["side"] < 0 else 100
        for i, (ang, length, hw) in enumerate(WING_FEATHERS):
            lift = w["out"] * (0.0014 * (n - 1 - i))                  # layering: primaries under
            across = w["out"].cross(w["dirs"][i]).normalized()
            feather(bm, PART_WING + sidecode + i, w["roots"][i] + lift, w["dirs"][i], across, w["out"],
                    length, hw, bow=0.008, tuck=0.02)
            # Covert over its root (one row of them forms the visible edge of the folded wing).
            c_dir = (w["dirs"][i] * 0.85 + w["down"] * 0.15).normalized()
            feather(bm, PART_WING + sidecode + i, w["roots"][i] + w["out"] * 0.0075 - w["dirs"][i] * 0.01,
                    c_dir, w["out"].cross(c_dir).normalized(), w["out"], 0.07, 0.02, bow=0.006, tuck=0.004)
        for k in range(3):                                                 # shoulder coverts
            d = (w["back"] * math.cos(math.radians(8 * k)) + w["down"] * math.sin(math.radians(8 * k))).normalized()
            feather(bm, PART_SHOULDER + sidecode, w["pivot"] + w["out"] * (0.009 + 0.0015 * k)
                    - w["back"] * 0.02 + w["down"] * (0.01 * k), d, w["out"].cross(d).normalized(), w["out"],
                    0.06, 0.024, bow=0.005)


def tail_feathers(bm, info):
    """Tail: a fan of long feathers rising off the rump, from about 30 degrees to nearly
    upright, alternately leaning a little left and right so that from behind they form the
    tail's roof shape."""
    root, rot = info["tail_root"], info["rot"]
    for elev, length, lean in ((30, 0.12, -0.012), (40, 0.145, 0.012), (50, 0.155, -0.01), (60, 0.155, 0.01),
                               (70, 0.14, -0.006), (80, 0.115, 0.006)):
        a = math.radians(elev)
        d = (rot @ V((-math.cos(a), 0, math.sin(a))) + V((0, lean / length, 0))).normalized()
        across = d.cross(V((0, 1, 0))).normalized()
        normal = d.cross(across).normalized()
        feather(bm, 0, root, d, across, normal, length, 0.02, bow=0.01, curl=0.1)


def loose_feathers(bm, info):
    """Small body feathers for the flapping hen to throw off: lying flat at rest inside the
    body (their bones move them out and scale them up while they fly)."""
    rng = random.Random("loose")
    centre = V((0.0, 0, 0.2))
    info["particles"] = []
    for k in range(PARTICLES):
        length = rng.uniform(0.032, 0.045)
        feather(bm, PART_PARTICLE + k, centre - V((length * 0.5, 0, 0)), V((1, 0, 0)), V((0, 1, 0)), V((0, 0, 1)),
                length, length * 0.24, bow=0.003, curl=0.25, nu=6, nv=2)
        info["particles"].append(centre)


def nest(bm_dict, rng):
    """Straw nest under a brooding hen: a low core of old straw in a ring round her, covered
    in loose straws laid roughly along the ring, a floor of straw, and eggs peeking out
    between her and the rim."""
    ring = []
    for k in range(46):                    # a little more than a full turn: the ends overlap
        a = 2 * math.pi * k / 42
        r = 0.175 + 0.012 * math.sin(5 * a) + 0.008 * math.sin(11 * a + 1.0)
        h = 0.035 + 0.008 * math.sin(7 * a + 0.5)
        ring.append((V((math.cos(a) * r, math.sin(a) * r * 0.85, 0.022)), 0.05, h, 0.02, 2))
    loft(bm_dict["straw_core"], ring, up_hint=V((0, 0, 1)))
    loft(bm_dict["straw_core"], [(V((0, 0, 0.0)), 0.17, 0.15, 0.15, 2), (V((0, 0, 0.018)), 0.17, 0.15, 0.15, 2)],
         up_hint=V((1, 0, 0)))
    for k in range(170):                   # straws over the ring and around it
        a = rng.uniform(0, 2 * math.pi)
        r = rng.gauss(0.195, 0.025)            # mostly on and outside the rim, clear of the eggs
        z = 0.03 + 0.03 * max(0.0, 1 - abs(r - 0.18) / 0.06) + rng.uniform(-0.005, 0.01)
        c = V((math.cos(a) * r, math.sin(a) * r * 0.85, z))
        tangent = V((-math.sin(a), math.cos(a) * 0.85, 0)).normalized()
        d = (tangent + V((rng.uniform(-0.6, 0.6), rng.uniform(-0.6, 0.6), rng.uniform(-0.25, 0.25)))).normalized()
        d *= rng.uniform(0.025, 0.055)
        key = "straw" if k % 3 else "straw_old"
        loft(bm_dict[key], [(c - d, 0.0045, 0.0016, 0.0016, 2), (c + d, 0.0045, 0.0016, 0.0016, 2)],
             up_hint=V((0, 0, 1)), ring=6)
    # Eggs on the inner slope beside her breast and flanks, peeking out from under her.
    for ang in (30, -35, 95, -110):
        a = math.radians(ang)
        c = V((math.cos(a) * 0.15, math.sin(a) * 0.15 * 0.95, 0.05))
        d = V((-math.sin(a), math.cos(a), 0.15)).normalized()
        loft(bm_dict["egg"], [(c - d * 0.03, 0.006, 0.006, 0.006, 2), (c - d * 0.022, 0.017, 0.017, 0.017, 2),
                              (c, 0.022, 0.022, 0.022, 2), (c + d * 0.02, 0.018, 0.018, 0.018, 2),
                              (c + d * 0.033, 0.004, 0.004, 0.004, 2)], up_hint=V((0, 0, 1)))


def extras(bm_dict, info):
    """Wings, tail, comb, wattles, beak, eyes, legs and toes."""
    wing_feathers(bm_dict["wing"], info)
    tail_feathers(bm_dict["tail"], info)
    info["particles"] = []
    if info["pose"] == "flapping":
        loose_feathers(bm_dict["wing"], info)
    head, face, side, up = info["head"], info["face"], info["side"], info["head_up"]
    # Single comb: a red blade along the top of the head with rounded points.
    loft(bm_dict["comb"], [(head - face * 0.025 + up * 0.022, 0.004, 0.008, 0.004, 2),
                           (head + up * 0.03, 0.005, 0.012, 0.006, 2),
                           (head + face * 0.028 + up * 0.018, 0.004, 0.007, 0.004, 2)], up_hint=up)
    for k, (along, height) in enumerate(((-0.018, 0.022), (-0.005, 0.03), (0.009, 0.029), (0.021, 0.02))):
        base = head + face * along + up * 0.03
        tip = base + up * height * 0.6 + face * 0.004 * (k - 1.5)
        loft(bm_dict["comb"], _lerp_stations([base, tip], [(0, 0.004, 0.006, 0.006, 2), (1, 0.003, 0.002, 0.002, 2)],
                                             samples=5), up_hint=face)
    # Wattles: two red lobes under the beak.
    for s in (-1, 1):
        top = head + face * 0.022 - up * 0.014 + side * s * 0.006
        loft(bm_dict["comb"], _lerp_stations([top, top - up * 0.014, top - up * 0.026], [
            (0, 0.004, 0.006, 0.006, 2), (0.5, 0.006, 0.01, 0.01, 2), (1, 0.004, 0.004, 0.004, 2)], samples=6),
            up_hint=face)
    # Beak: short, tapering, its tip hooked slightly down.
    b0 = head + face * 0.028 + up * 0.002
    beak = [b0, b0 + face * 0.016, b0 + face * 0.03 - up * 0.006]
    loft(bm_dict["beak"], _lerp_stations(beak, [(0, 0.011, 0.01, 0.009, 2), (0.6, 0.006, 0.005, 0.005, 2),
                                                (1, 0.0015, 0.0015, 0.0015, 2)], samples=8), up_hint=up)
    for s in (-1, 1):
        eye = head + face * 0.012 + up * 0.008 + side * s * 0.022
        loft(bm_dict["eye"], [(eye - side * s * 0.004, 0.003, 0.004, 0.004, 2),
                              (eye + side * s * 0.002, 0.004, 0.005, 0.005, 2),
                              (eye + side * s * 0.005, 0.002, 0.002, 0.002, 2)], up_hint=up)
    # Legs: scaly shanks from the hocks to the feet, three toes forward and one back.
    legs = []
    if not info["sitting"]:
        for hip, hock in info["hips"]:
            foot = V((hock.x + 0.008, hock.y * 1.05, 0.012))
            shank = [hock + V((0, 0, 0.01)), hock.lerp(foot, 0.5), foot]
            loft(bm_dict["leg"], _lerp_stations(shank, [(0, 0.011, 0.011, 0.011, 2), (1, 0.008, 0.008, 0.008, 2)],
                                                samples=6), up_hint=V((1, 0, 0)))
            for ang, length in ((-28, 0.05), (0, 0.06), (28, 0.05), (180, 0.028)):
                a = math.radians(ang) * (1 if foot.y > 0 else -1)
                d = V((math.cos(a), math.sin(a), 0))
                toe = [foot, foot + d * length * 0.5 + V((0, 0, -0.004)), foot + d * length + V((0, 0, -0.008))]
                loft(bm_dict["leg"], _lerp_stations(toe, [(0, 0.006, 0.006, 0.006, 2), (1, 0.0035, 0.003, 0.003, 2)],
                                                    samples=5), up_hint=V((0, 0, 1)))
            legs.append([hip, hock, foot])
    info["legs"] = legs


def _zones(me, info):
    """Plumage zones per point as the "zones" colour attribute: red = hackle (neck and the
    cape over the shoulders), green = saddle (in front of the tail), blue = breast."""
    import numpy as np
    from cow_rig import _project, _smooth
    n = len(me.vertices)
    co = np.empty(n * 3)
    me.vertices.foreach_get("co", co)
    co = co.reshape(n, 3)
    neck = [np.array(p) for p in (info["cape_start"], info["neck_base"], info["neck_bend"], info["head"])]
    _, d, _ = _project(co, neck)
    head_d = np.linalg.norm(co - np.array(info["head"]), axis=1)
    hackle = (1 - _smooth(0.05, 0.08, d)) * _smooth(0.035, 0.05, head_d)
    tr = np.array(info["tail_root"])
    saddle = (1 - _smooth(0.04, 0.09, np.linalg.norm(co - tr, axis=1))) * _smooth(tr[2] - 0.04, tr[2], co[:, 2])
    up = np.array([info["upright"](V(p)) for p in co])
    breast = _smooth(-0.02, 0.06, up[:, 0]) * (1 - _smooth(0.2, 0.25, up[:, 2]))
    col = np.stack([hackle, saddle, breast, np.ones(n)], axis=1).astype(np.float32)
    attr = me.color_attributes.new("zones", "FLOAT_COLOR", "POINT")
    attr.data.foreach_set("color", col.ravel())


def build_chicken_mesh(name, pose, feather_mat, mats):
    """Returns (mesh, info): the fused feathered body plus wings, tail, comb, wattles, beak,
    eyes and legs, with each point's rest position stored as the "rest" attribute."""
    coll = bpy.context.scene.collection
    feathers_bm = bmesh.new()
    info = body_parts(feathers_bm, pose)
    body = _to_object(feathers_bm, name + "-feathers", coll)
    _apply(body, "REMESH", mode="VOXEL", voxel_size=0.005)
    _apply(body, "SMOOTH", factor=0.8, iterations=4)
    _apply(body, "DECIMATE", ratio=0.75)        # small bird: keep it fine, or it looks faceted
    body.data.materials.append(mats[feather_mat])
    bms = {k: bmesh.new() for k in ("wing", "tail", "comb", "beak", "eye", "leg", "straw", "straw_old",
                                    "straw_core", "egg")}
    extras(bms, info)
    if pose == "brooding":
        nest(bms, random.Random(name))
    pieces = [body]
    for key, mat in (("wing", feather_mat + "_wing"), ("tail", feather_mat + "_tail"), ("comb", "comb_red"),
                     ("beak", "beak_yellow"), ("eye", "eye_black"), ("leg", "chicken_leg"), ("straw", "straw"),
                     ("straw_old", "straw_old"), ("straw_core", "straw_old"), ("egg", "egg_shell")):
        if len(bms[key].verts) == 0:
            bms[key].free()
            continue
        o = _to_object(bms[key], f"{name}-{key}", coll)
        if key in ("wing", "tail"):                      # thin, but with some substance
            _apply(o, "SOLIDIFY", thickness=0.0025, offset=0.0)
        o.data.materials.append(mats[mat])
        pieces.append(o)
    for o in pieces:
        for p in o.data.polygons:
            p.use_smooth = True
        # Every piece gets the same attributes, so joining keeps them for all points.
        if "part" not in o.data.attributes:
            o.data.attributes.new("part", "INT", "POINT")
        if not o.data.uv_layers:
            o.data.uv_layers.new(name="UVMap")
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
    _zones(me, info)
    return me, info
