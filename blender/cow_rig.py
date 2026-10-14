"""Skeleton, skin weights and motion for the cow model.

Bones (cow-local space, cow faces +x):
    body                 static: trunk and legs
    neck1 -> neck2 -> head -> ear.L, ear.R
    body -> tail1 -> tail2 -> tail3 -> tail4

Each skin point follows the nearest bones with smooth blending across joints. Motion is
done purely with bone rotations, and every motion completes whole cycles per loop.
"""

import math

import bpy
import numpy as np
from mathutils import Quaternion, Vector

V = Vector


def _smooth(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def _project(points, poly):
    """Closest point on a polyline for each point: returns (arc length s, distance, segment index)."""
    best_d = np.full(len(points), np.inf)
    best_s = np.zeros(len(points))
    best_k = np.zeros(len(points), dtype=int)
    acc = 0.0
    for k, (a, b) in enumerate(zip(poly, poly[1:])):
        a, b = np.array(a), np.array(b)
        ab = b - a
        L = np.linalg.norm(ab)
        t = np.clip(((points - a) @ ab) / (L * L), 0, 1)
        closest = a + t[:, None] * ab
        d = np.linalg.norm(points - closest, axis=1)
        better = d < best_d
        best_d[better] = d[better]
        best_s[better] = acc + t[better] * L
        best_k[better] = k
        acc += L
    return best_s, best_d, best_k


def _chain_weights(s, spans, h):
    """Smooth 'hat' weights for consecutive bones covering arc-length spans [(a, b), ...]."""
    out = []
    for k, (a, b) in enumerate(spans):
        w = _smooth(a - h, a + h, s)
        if k < len(spans) - 1:
            w = w * (1 - _smooth(b - h, b + h, s))
        out.append(w)
    return out


def rig_cow(obj, info, name):
    """Adds an armature with skin weights to the cow mesh object (mesh in cow-local space).
    Returns the armature object; obj becomes its child with an Armature modifier."""
    coll = bpy.context.scene.collection
    arm_data = bpy.data.armatures.new(name + "-rig")
    rig = bpy.data.objects.new(name + "-rig", arm_data)
    coll.objects.link(rig)
    neck = [V(p) for p in info["neck"]]           # neck base ... poll
    poll, nose, face = V(info["poll"]), V(info["nose"]), V(info["face"])
    tp = [V(p) for p in info["tail"]]            # tail root (inside the body), tail head, mid, tip
    # Five joints for four bones: tail head -> down the tail -> end of the switch.
    tail = [tp[1], tp[2], tp[2].lerp(tp[3], 0.5), tp[3], V(info["tail_end"]) + V((0.01, 0, -0.25))]
    mid = neck[len(neck) // 2]

    bones = {}
    bpy.context.view_layer.objects.active = rig
    with bpy.context.temp_override(active_object=rig, object=rig):
        bpy.ops.object.mode_set(mode="EDIT")
        eb = arm_data.edit_bones

        def add(bname, head, tail_pt, parent=None):
            b = eb.new(bname)
            b.head, b.tail = head, tail_pt
            if parent:
                b.parent = eb[parent]
            bones[bname] = (V(head), V(tail_pt))

        add("body", V((-0.5, 0, 1.1)), V((0.5, 0, 1.1)))
        add("neck1", neck[0], mid, "body")
        add("neck2", mid, poll, "neck1")
        add("head", poll, nose, "neck2")
        for side_name, (base, tip) in zip(("ear.R", "ear.L"), info["ears"]):
            add(side_name, V(base), V(tip), "head")
        add("tail1", tail[0], tail[1], "body")
        add("tail2", tail[1], tail[2], "tail1")
        add("tail3", tail[2], tail[3], "tail2")
        add("tail4", tail[3], tail[4], "tail3")
        bpy.ops.object.mode_set(mode="OBJECT")

    # Skin weights from the rest positions.
    me = obj.data
    n = len(me.vertices)
    co = np.empty(n * 3, dtype=np.float64)
    me.vertices.foreach_get("co", co)
    co = co.reshape(n, 3)
    x, y, z = co[:, 0], co[:, 1], co[:, 2]
    weights = {b: np.zeros(n) for b in bones}

    # Neck and head: arc length along neck base -> poll -> nose.
    chain = [neck[0], mid, poll, nose]
    s, d, _ = _project(co, [np.array(p) for p in chain])
    L1 = (mid - neck[0]).length
    L2 = L1 + (poll - mid).length
    in_front = (d < 0.5) & ~((x < 0.75) & (z < 0.9)) & (x > 0.3)
    w1, w2, wh = _chain_weights(s, [(0.22, L1), (L1, L2), (L2, 99)], 0.09)
    weights["neck1"] = np.where(in_front, w1, 0)
    weights["neck2"] = np.where(in_front, w2, 0)
    weights["head"] = np.where(in_front, wh, 0)
    # Ears: points close to an ear segment, beyond its base.
    for bname in ("ear.R", "ear.L"):
        base, tip = bones[bname]
        es, ed, _ = _project(co, [np.array(base), np.array(tip)])
        ew = np.where(ed < 0.07, _smooth(0.02, 0.06, es), 0) * (weights["head"] > 0)
        weights[bname] = ew
        weights["head"] = weights["head"] * (1 - ew)
    # Tail: points behind the rump and below the tail head, along the tail chain.
    ts, td, _ = _project(co, [np.array(p) for p in tail])
    # Only the tail itself: close to the tail line (wider for the switch), behind the rump.
    reach = np.where(z < 0.75, 0.085, 0.05)   # the switch (lower tail) is thicker
    # Below the rump only the tail is there (hocks end around x = -0.8), so the cut-off can
    # move forward and take the whole thickness of the switch.
    behind = np.where(z < 0.75, x < -0.84, x < -0.9)
    tmask = behind & (z < tail[0].z - 0.02) & (td < reach)
    lens = np.cumsum([0] + [(b - a).length for a, b in zip(tail, tail[1:])])
    tw = _chain_weights(ts, [(0.03, lens[1]), (lens[1], lens[2]), (lens[2], lens[3]), (lens[3], 99)], 0.05)
    for k, w in enumerate(tw):
        weights[f"tail{k + 1}"] = np.where(tmask, w, 0)
    # Body takes what is left.
    total = sum(w for b, w in weights.items() if b != "body")
    weights["body"] = np.clip(1 - total, 0, 1)
    total = sum(weights.values())
    for b in weights:
        weights[b] = weights[b] / np.maximum(total, 1e-6)

    for b, w in weights.items():
        vg = obj.vertex_groups.new(name=b)
        idx = np.nonzero(w > 0.002)[0]
        for i in idx:
            vg.add([int(i)], float(w[i]), "REPLACE")
    mod = obj.modifiers.new("rig", "ARMATURE")
    mod.object = rig
    obj.parent = rig
    return rig


# Motion ------------------------------------------------------------------------------

def _add_rot(pb, axis_arm, angle):
    """Adds a rotation by angle (radians) around an axis given in armature space."""
    rest = pb.bone.matrix_local.to_3x3()
    local_axis = rest.inverted() @ V(axis_arm)
    pb.rotation_mode = "QUATERNION"
    pb.rotation_quaternion = pb.rotation_quaternion @ Quaternion(local_axis, angle)


def _pulse(t, at, width):
    """Periodic soft pulse (0..1) centred on 'at' in the loop phase t in [0, 1)."""
    d = (t - at + 0.5) % 1.0 - 0.5
    return math.exp(-(d / width) ** 2)


def pose_cow(rig, pose, t, seed):
    """Sets the cow's bones for loop phase t in [0, 1). seed varies timing between cows."""
    rad = math.radians
    tau = 2 * math.pi
    ph = (seed * 0.618) % 1.0
    up, side, fwd = V((0, 0, 1)), V((0, 1, 0)), V((1, 0, 0))
    pbs = rig.pose.bones
    for pb in pbs:
        pb.rotation_mode = "QUATERNION"
        pb.rotation_quaternion = Quaternion()
    if pose == "grazing":
        # Slow sideways sweep of the muzzle over the grass and a short tug when tearing a bite.
        sweep = math.sin(tau * (t + ph))
        tug = _pulse(t, ph + 0.35, 0.05) + _pulse(t, ph + 0.85, 0.05)
        _add_rot(pbs["neck1"], up, rad(2.5) * sweep)
        _add_rot(pbs["neck2"], up, rad(3.5) * sweep)
        _add_rot(pbs["head"], up, rad(4) * sweep)
        _add_rot(pbs["head"], side, -rad(7) * tug)
        _add_rot(pbs["neck2"], side, -rad(2) * tug)
    elif pose in ("eating", "drinking"):
        eat = pose == "eating"
        _add_rot(pbs["head"], up, rad(3 if eat else 1.2) * math.sin(tau * (t + ph)))
        _add_rot(pbs["neck2"], side, rad(1.5 if eat else 0.8) * math.sin(tau * 2 * (t + ph)))
    else:  # standing / looking: slow look around
        turn = math.sin(tau * (t + ph))
        nod = math.sin(tau * 2 * (t + ph) + 1.0)
        _add_rot(pbs["neck1"], up, rad(4) * turn)
        _add_rot(pbs["neck2"], up, rad(5) * turn)
        _add_rot(pbs["head"], up, rad(6) * turn)
        _add_rot(pbs["head"], side, rad(2) * nod)
    # Ears: one quick flick each per loop at different times, plus a slow drift.
    for k, ear in enumerate(("ear.R", "ear.L")):
        flick = _pulse(t, (ph + 0.2 + 0.45 * k) % 1.0, 0.035)
        _add_rot(pbs[ear], up, (rad(28) * flick + rad(4) * math.sin(tau * (t + k * 0.3))) * (1 if k else -1))
    # Tail: a swish that travels down the chain (each bone lags the one above it).
    cycles = 1 + (seed % 2)
    amp = [rad(6), rad(8), rad(10), rad(12)]
    for k in range(4):
        a = math.sin(tau * cycles * (t + ph) - 0.8 * k)
        _add_rot(pbs[f"tail{k + 1}"], fwd, amp[k] * a)
