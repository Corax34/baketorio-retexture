"""Cow built like a real model: lofted cross-sections at Holstein proportions, fused into one
surface with a voxel remesh and smoothed, instead of stacked primitives.

Ground at z=0, cow faces +x. Units: metres (1 tile = 1 m). Withers ~1.42 m.
"""

import math

import bmesh
import bpy
from mathutils import Vector

V = Vector


def loft(bm, stations, up_hint=V((0, 0, 1)), ring=24, cap=True):
    """stations: list of (center, half_width, up, down, exponent). Builds a closed tube whose
    cross-section is a superellipse: half_width sideways, 'up'/'down' along the section's up
    axis. The up axis is carried from section to section (parallel transport), so the tube
    never flips or tears where the path bends sharply."""
    rings = []
    n = len(stations)
    up = None
    for i, (c, hw, up_r, down_r, ex) in enumerate(stations):
        prev_c = stations[max(i - 1, 0)][0]
        next_c = stations[min(i + 1, n - 1)][0]
        t = (next_c - prev_c).normalized()
        if up is None:
            up = up_hint - t * up_hint.dot(t)
            if up.length < 1e-4:
                up = V((1, 0, 0)) - t * t.x
        else:
            up = up - t * up.dot(t)
        up.normalize()
        side = t.cross(up).normalized()
        verts = []
        for j in range(ring):
            th = j / ring * 2 * math.pi
            cs, sn = math.cos(th), math.sin(th)
            x = math.copysign(abs(cs) ** (2 / ex), cs) * hw
            r = up_r if sn >= 0 else down_r
            y = math.copysign(abs(sn) ** (2 / ex), sn) * r
            verts.append(bm.verts.new(c + side * x + up * y))
        rings.append(verts)
    for a, b in zip(rings, rings[1:]):
        for j in range(ring):
            k = (j + 1) % ring
            bm.faces.new((a[j], a[k], b[k], b[j]))
    if cap:
        for verts, rev in ((rings[0], True), (rings[-1], False)):
            ctr = bm.verts.new(sum((v.co for v in verts), V()) / len(verts))
            for j in range(ring):
                k = (j + 1) % ring
                f = (ctr, verts[k], verts[j]) if rev else (ctr, verts[j], verts[k])
                bm.faces.new(f)


def _catmull(points, samples):
    """Smooth curve through points (Catmull-Rom), returned as a dense list of points."""
    if len(points) == 2:
        return [points[0].lerp(points[1], i / (samples - 1)) for i in range(samples)]
    pts = [points[0] * 2 - points[1]] + list(points) + [points[-1] * 2 - points[-2]]
    out = []
    segs = len(points) - 1
    for i in range(samples):
        u = i / (samples - 1) * segs
        k = min(int(u), segs - 1)
        f = u - k
        p0, p1, p2, p3 = pts[k], pts[k + 1], pts[k + 2], pts[k + 3]
        out.append(0.5 * ((2 * p1) + (-p0 + p2) * f + (2 * p0 - 5 * p1 + 4 * p2 - p3) * f * f
                          + (-p0 + 3 * p1 - 3 * p2 + p3) * f * f * f))
    return out


def _lerp_stations(path, profile, samples=18):
    """path: control points of a smooth curve; profile: list of (t, hw, up, down, ex) keys with
    t in 0..1 along the curve. Returns densely sampled, interpolated stations."""
    dense = _catmull(path, 80)
    seg = [0.0]
    for a, b in zip(dense, dense[1:]):
        seg.append(seg[-1] + (b - a).length)
    total = seg[-1]

    def point_at(t):
        d = t * total
        for i in range(len(dense) - 1):
            if d <= seg[i + 1] or i == len(dense) - 2:
                f = (d - seg[i]) / max(seg[i + 1] - seg[i], 1e-6)
                return dense[i].lerp(dense[i + 1], min(max(f, 0), 1))

    def profile_at(t):
        for (t0, *a), (t1, *b) in zip(profile, profile[1:]):
            if t <= t1:
                f = (t - t0) / max(t1 - t0, 1e-6)
                return [x + (y - x) * f for x, y in zip(a, b)]
        return list(profile[-1][1:])

    return [(point_at(i / (samples - 1)), *profile_at(i / (samples - 1))) for i in range(samples)]


def body_parts(bm, pose, bull=False):
    """Lofts the body, neck, head, ears, legs and tail for a pose; returns the joint
    positions the skeleton (cow_rig.py) is built from. A bull is built heavier in front:
    a high crest over the withers, broad shoulders and a thick neck."""
    # Trunk: (x, z_top, z_bottom, half_width)
    trunk = [
        (-0.9, 1.30, 1.02, 0.13),   # pin bones / tail head
        (-0.8, 1.37, 0.80, 0.24),
        (-0.62, 1.41, 0.66, 0.29),  # hooks
        (-0.4, 1.38, 0.58, 0.32),
        (-0.15, 1.37, 0.53, 0.36),  # barrel / belly
        (0.12, 1.38, 0.54, 0.36),
        (0.38, 1.42, 0.60, 0.32),
        (0.55, 1.44, 0.66, 0.27),   # withers
        (0.7, 1.33, 0.70, 0.22),    # point of shoulder
        (0.8, 1.18, 0.80, 0.14),    # brisket
    ]
    if bull:   # (x, extra top, extra half width): crest and shoulders
        more = {0.38: (0.04, 0.03), 0.55: (0.08, 0.05), 0.7: (0.07, 0.05), 0.8: (0.04, 0.03)}
        trunk = [(x, top + more.get(x, (0, 0))[0], bot, hw + more.get(x, (0, 0))[1]) for x, top, bot, hw in trunk]
    stations = []
    for x, top, bot, hw in trunk:
        c = V((x, 0, (top + bot) / 2))
        h = (top - bot) / 2
        stations.append((c, hw, h, h * 1.05, 2.4))
    loft(bm, stations, up_hint=V((0, 0, 1)))

    # Neck and head pose.
    if pose == "grazing":
        neck_path = [V((0.55, 0, 1.08)), V((0.88, 0, 0.92)), V((1.05, 0, 0.6))]
        poll = V((1.08, 0, 0.55))
        face = V((0.3, 0, -1.0)).normalized()
    elif pose in ("eating", "drinking"):
        # Head stretched forward and down over a trough rim, muzzle in the feed or water.
        low = 0.08 if pose == "drinking" else 0.0
        neck_path = [V((0.55, 0, 1.08)), V((0.86, 0, 1.04 - low * 0.5)), V((1.08, 0, 0.95 - low))]
        poll = V((1.12, 0, 0.92 - low))
        face = V((0.75, 0, -0.66)).normalized()
    elif pose == "looking":
        neck_path = [V((0.55, 0, 1.08)), V((0.86, 0.1, 1.28)), V((1.0, 0.3, 1.42))]
        poll = V((1.02, 0.34, 1.46))
        face = V((0.55, 0.45, -0.7)).normalized()
    else:  # standing
        neck_path = [V((0.55, 0, 1.08)), V((0.86, 0, 1.26)), V((1.03, 0, 1.4))]
        poll = V((1.05, 0, 1.44))
        face = V((0.5, 0, -0.86)).normalized()
    thick = 1.2 if bull else 1.0                          # a bull's neck is massive
    loft(bm, _lerp_stations(neck_path + [poll, poll + face * 0.12], [
        (0.0, 0.25 * thick, 0.26 * thick * thick, 0.36, 2.2),
        (0.35, 0.2 * thick, 0.23 * thick * thick, 0.32, 2.2),    # dewlap hangs under the neck
        (0.7, 0.16 * thick, 0.17 * thick, 0.2, 2.2),
        (0.9, 0.14 * thick, 0.14, 0.14, 2.2),
        (1.0, 0.11, 0.1, 0.1, 2.2),
    ]), up_hint=V((0, 0, 1)))
    # Head: poll -> forehead (widest at the eyes) -> long face -> broad muzzle.
    nose = poll + face * 0.5
    brow = 0.02 if bull else 0.0                          # broad, curly forehead
    loft(bm, _lerp_stations([poll, nose], [
        (0.0, 0.12 + brow, 0.11 + brow, 0.11, 2.2),
        (0.18, 0.14 + brow, 0.12 + brow, 0.13, 2.4),   # widest at the eyes
        (0.55, 0.11, 0.1, 0.12, 2.3),
        (0.82, 0.1, 0.09, 0.11, 2.3),
        (0.95, 0.115, 0.085, 0.1, 2.5),  # muzzle flares
        (1.0, 0.1, 0.07, 0.08, 2.5),
    ]), up_hint=V((1, 0, 0.2)) if pose != "grazing" else V((1, 0, 0)))
    side = face.cross(V((0, 0, 1))).normalized()
    if side.length < 0.1:
        side = V((0, 1, 0))
    # Ears out sideways just behind the eyes.
    ears = []
    for s in (-1, 1):
        base = poll + face * 0.06 + side * s * 0.1
        # Broad, leaf-shaped ear hanging slightly down and back; thin in one direction only.
        tip = base + side * s * 0.19 - face * 0.05 + V((0, 0, -0.07))
        ears.append((base, tip))
        loft(bm, _lerp_stations([base, tip], [
            (0.0, 0.035, 0.03, 0.03, 2), (0.45, 0.075, 0.018, 0.018, 2.2),
            (0.8, 0.06, 0.015, 0.015, 2.2), (1.0, 0.015, 0.008, 0.008, 2)]),
            up_hint=face)

    # Legs. Front: elbow -> forearm -> knee -> cannon -> fetlock -> pastern.
    legs = []
    for sy in (-1, 1):
        x = 0.55
        legs.append([V((x, sy * 0.17, 0.85)), V((x + 0.02, sy * 0.15, 0.48)),
                     V((x + 0.02, sy * 0.15, 0.36)), V((x + 0.03, sy * 0.15, 0.12)),
                     V((x + 0.07, sy * 0.15, 0.05))])
    front_profile = [(0.0, 0.13, 0.13, 0.13, 2), (0.42, 0.075, 0.085, 0.085, 2), (0.5, 0.07, 0.08, 0.08, 2),
                     (0.62, 0.055, 0.06, 0.06, 2), (0.88, 0.06, 0.065, 0.065, 2), (1.0, 0.05, 0.055, 0.055, 2)]
    hind = []
    for sy in (-1, 1):
        x = -0.6
        # Stifle -> hock bends backwards -> cannon -> fetlock -> pastern.
        hind.append([V((x + 0.05, sy * 0.19, 0.95)), V((x - 0.12, sy * 0.15, 0.5)),
                     V((x - 0.1, sy * 0.15, 0.4)), V((x - 0.06, sy * 0.15, 0.12)),
                     V((x - 0.02, sy * 0.15, 0.05))])
    hind_profile = [(0.0, 0.17, 0.19, 0.19, 2), (0.35, 0.08, 0.1, 0.1, 2), (0.45, 0.07, 0.09, 0.09, 2),
                    (0.6, 0.055, 0.06, 0.06, 2), (0.88, 0.06, 0.065, 0.065, 2), (1.0, 0.05, 0.055, 0.055, 2)]
    for path in legs:
        loft(bm, _lerp_stations(path, front_profile), up_hint=V((1, 0, 0)))
    for path in hind:
        loft(bm, _lerp_stations(path, hind_profile), up_hint=V((1, 0, 0)))
    # Tail: from the tail head down between the pins, with a switch at the end.
    tail = [V((-0.8, 0, 1.2)), V((-0.94, 0, 1.12)), V((-0.96, 0.02, 0.8)), V((-0.93, 0.03, 0.5))]
    loft(bm, _lerp_stations(tail, [(0, 0.045, 0.045, 0.045, 2), (0.15, 0.03, 0.03, 0.03, 2),
                                   (1.0, 0.022, 0.022, 0.022, 2)]), up_hint=V((1, 0, 0)))
    return {"nose": nose, "face": face, "poll": poll, "side": side, "legs": legs + hind, "tail_end": tail[-1],
            "neck": list(neck_path) + [poll], "ears": ears, "tail": list(tail)}


def extras(bm_dict, info, horns, bull=False):
    """Small parts with their own colours (hooves, muzzle, udder, switch, horns, and a
    bull's nose ring instead of the udder)."""
    for path in info["legs"]:
        foot = path[-1]
        bm = bm_dict["hoof"]
        # Hoof continues the pastern: starts inside the leg at the same width, flares
        # slightly and ends flat on the ground.
        loft(bm, [(foot + V((0.0, 0, 0.05)), 0.05, 0.055, 0.055, 2.2),
                  (foot + V((0.02, 0, 0.0)), 0.058, 0.062, 0.06, 2.6),
                  (V((foot.x + 0.035, foot.y, 0.0)), 0.062, 0.068, 0.062, 2.8)], up_hint=V((1, 0, 0)))
    face, nose = info["face"], info["nose"]
    # Rounded nose pad that fully covers the end of the head.
    loft(bm_dict["muzzle"], [(nose - face * 0.08, 0.116, 0.094, 0.1, 2.4),
                             (nose - face * 0.02, 0.126, 0.102, 0.108, 2.5),
                             (nose + face * 0.02, 0.114, 0.09, 0.096, 2.4),
                             (nose + face * 0.045, 0.085, 0.06, 0.07, 2.2),
                             (nose + face * 0.06, 0.035, 0.025, 0.03, 2.0)], up_hint=V((1, 0, 0.2)))
    if bull:
        # Brass ring through the nose, hanging below the muzzle.
        side = info["side"]
        up = side.cross(face).normalized()
        centre = nose + face * 0.045 - up * 0.07
        ring = []
        for k in range(17):
            a = 6.2832 * k / 16
            ring.append((centre + (up * math.cos(a) + face * math.sin(a)) * 0.045, 0.008, 0.008, 0.008, 2))
        loft(bm_dict["ring"], ring, up_hint=side, cap=False)
    else:
        loft(bm_dict["udder"], [(V((-0.5, 0, 0.62)), 0.12, 0.08, 0.08, 2.2),
                                (V((-0.35, 0, 0.53)), 0.17, 0.12, 0.13, 2.2),
                                (V((-0.18, 0, 0.58)), 0.12, 0.08, 0.08, 2.2)], up_hint=V((0, 0, 1)))
        for tx in (-0.43, -0.28):
            for ty in (-0.06, 0.06):
                loft(bm_dict["udder"], [(V((tx, ty, 0.45)), 0.018, 0.018, 0.018, 2),
                                        (V((tx, ty, 0.38)), 0.014, 0.014, 0.014, 2)], up_hint=V((1, 0, 0)))
    te = info["tail_end"]
    loft(bm_dict["switch"], _lerp_stations([te + V((0, 0, 0.12)), te + V((0.01, 0, -0.25))], [
        (0.0, 0.025, 0.025, 0.025, 2), (0.35, 0.055, 0.05, 0.05, 2), (0.7, 0.045, 0.04, 0.04, 2),
        (1.0, 0.008, 0.008, 0.008, 2)]), up_hint=V((1, 0, 0)))
    if horns:
        for s in (-1, 1):
            base = info["poll"] + info["side"] * s * 0.08 + V((0, 0, 0.04))
            if bull:   # short, thick, sweeping out and forward
                mid = base + info["side"] * s * 0.12 + V((0, 0, 0.02))
                tip = mid + info["side"] * s * 0.05 + info["face"] * 0.08 + V((0, 0, 0.05))
                loft(bm_dict["horn"], _lerp_stations([base, mid, tip], [(0, 0.05, 0.05, 0.05, 2),
                                                                      (0.5, 0.035, 0.035, 0.035, 2),
                                                                      (1, 0.009, 0.009, 0.009, 2)]),
                     up_hint=V((1, 0, 0)))
                continue
            tip = base + info["side"] * s * 0.16 + V((0, 0, 0.1)) + info["face"] * 0.04
            loft(bm_dict["horn"], _lerp_stations([base, tip], [(0, 0.035, 0.035, 0.035, 2),
                                                             (1, 0.008, 0.008, 0.008, 2)]),
                 up_hint=V((1, 0, 0)))


def _to_object(bm, name, coll):
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    o = bpy.data.objects.new(name, me)
    coll.objects.link(o)
    return o


def _apply(o, mod_type, **settings):
    m = o.modifiers.new(mod_type, mod_type)
    for k, v in settings.items():
        setattr(m, k, v)
    bpy.context.view_layer.objects.active = o
    with bpy.context.temp_override(object=o, active_object=o):
        bpy.ops.object.modifier_apply(modifier=m.name)


def build_cow_mesh(name, pose, coat_mat, mats, horns=False, bull=False):
    """Returns (mesh, joints): the fused, smoothed body plus the coloured extras, with each
    point's rest position stored as the "rest" attribute so coat patterns follow the skin
    when the skeleton bends it."""
    coll = bpy.context.scene.collection
    bm = bmesh.new()
    info = body_parts(bm, pose, bull)
    body = _to_object(bm, name + "-body", coll)
    _apply(body, "REMESH", mode="VOXEL", voxel_size=0.018)
    _apply(body, "SMOOTH", factor=0.8, iterations=4)
    _apply(body, "DECIMATE", ratio=0.35)
    keys = ["hoof", "muzzle", "udder", "switch", "horn", "ring"]
    bms = {k: bmesh.new() for k in keys}
    extras(bms, info, horns or bull, bull)
    pieces = [body]
    mat_for = {"hoof": "hoof", "muzzle": "muzzle", "udder": "udder", "switch": "tuft", "horn": "horn",
               "ring": "brass"}
    for k in keys:
        if len(bms[k].verts) == 0:
            bms[k].free()
            continue
        o = _to_object(bms[k], f"{name}-{k}", coll)
        o.data.materials.append(mats[mat_for[k]])
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
