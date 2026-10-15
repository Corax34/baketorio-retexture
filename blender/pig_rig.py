"""Skeleton, skin weights and motion for the pig model (see cow_rig.py for the method).

Bones (pig-local space, pig faces +x):
    body                 static: trunk and legs
    body -> belly        breathing (scaled, only used by lying pigs)
    body -> neck -> head -> ear.L, ear.R
    body -> tail1 -> tail2
    body -> leg.FR.upper -> leg.FR.lower    (and FL, HR, HL: front/hind, right/left)
"""

import math

import bpy
import numpy as np
from mathutils import Quaternion, Vector

from cow_rig import _add_rot, _chain_weights, _project, _pulse, _smooth

TAIL_SKIN = 0.045      # arc length along the tail (from its root in the rump) to the skin
LEGS = ("leg.FR", "leg.HR", "leg.FL", "leg.HL")   # in the order of the model's leg paths

V = Vector


def rig_pig(obj, info, name):
    """Adds an armature with skin weights to the pig mesh object (mesh in pig-local space).
    Returns the armature object; obj becomes its child with an Armature modifier."""
    coll = bpy.context.scene.collection
    arm_data = bpy.data.armatures.new(name + "-rig")
    rig = bpy.data.objects.new(name + "-rig", arm_data)
    coll.objects.link(rig)
    path = [V(p) for p in info["path"]]          # neck base, head joint, mid snout, snout tip
    tail = [V(p) for p in info["tail"]]          # dense: root inside the rump to the tip
    tail_s = [0.0]
    for a, b in zip(tail, tail[1:]):
        tail_s.append(tail_s[-1] + (b - a).length)

    def tail_at(d):
        """Point on the tail centre line at arc length d."""
        for k in range(len(tail) - 1):
            if d <= tail_s[k + 1] or k == len(tail) - 2:
                f = (d - tail_s[k]) / max(tail_s[k + 1] - tail_s[k], 1e-6)
                return tail[k].lerp(tail[k + 1], min(max(f, 0), 1))

    s_skin = TAIL_SKIN                            # where the tail leaves the rump
    s_mid = (s_skin + tail_s[-1]) / 2

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

        add("body", V((-0.5, 0, 0.56)), V((0.4, 0, 0.56)))
        add("belly", V((-0.35, 0, 0.56)), V((0.2, 0, 0.56)), "body")
        add("neck", path[0], path[1], "body")
        add("head", path[1], path[3], "neck")
        for side_name, (base, tip) in zip(("ear.R", "ear.L"), info["ears"]):
            add(side_name, V(base), V(tip), "head")
        add("tail1", tail_at(s_skin), tail_at(s_mid), "body")
        add("tail2", tail_at(s_mid), tail[-1], "tail1")
        for leg_name, leg in zip(LEGS, info["legs"]):        # hip or shoulder, knee, trotter
            add(leg_name + ".upper", V(leg[0]), V(leg[1]), "body")
            add(leg_name + ".lower", V(leg[1]), V(leg[3]), leg_name + ".upper")
        bpy.ops.object.mode_set(mode="OBJECT")

    me = obj.data
    n = len(me.vertices)
    co = np.empty(n * 3, dtype=np.float64)
    me.vertices.foreach_get("co", co)
    co = co.reshape(n, 3)
    x, y, z = co[:, 0], co[:, 1], co[:, 2]
    weights = {b: np.zeros(n) for b in bones}

    # Neck and head along neck base -> head joint -> snout tip; front legs excluded.
    s, d, _ = _project(co, [np.array(p) for p in path])
    L1 = (path[1] - path[0]).length
    near_leg = np.zeros(n, bool)
    for leg in info["legs"]:
        if leg[0].x > 0:                                   # front legs
            _, ld, _ = _project(co, [np.array(q) for q in leg])
            near_leg |= (ld < 0.1) & (z < 0.38)
    in_front = (d < 0.36) & (x > 0.25) & ~near_leg
    w_neck, w_head = _chain_weights(s, [(0.06, L1), (L1, 99)], 0.12)   # wide blends: no sharp bend
    weights["neck"] = np.where(in_front, w_neck, 0)
    weights["head"] = np.where(in_front, w_head, 0)
    # Ears: the ear sheets (their own material), each to its nearer ear bone, fading in from
    # where they leave the head.
    mat_of = np.zeros(n, dtype=int)
    for poly in me.polygons:
        mat_of[list(poly.vertices)] = poly.material_index
    ear_mats = [i for i, m in enumerate(me.materials) if m.name.split(".")[0] in ("ear_pink", "ear_black")]
    on_ear = np.isin(mat_of, ear_mats)
    proj = {b: _project(co, [np.array(bones[b][0]), np.array(bones[b][1])]) for b in ("ear.R", "ear.L")}
    nearer_r = proj["ear.R"][1] < proj["ear.L"][1]
    for bname, mine in (("ear.R", nearer_r), ("ear.L", ~nearer_r)):
        es = proj[bname][0]
        ew = np.where(on_ear & mine, _smooth(0.0, 0.05, es), 0)
        weights[bname] = ew
        weights["head"] = np.where(on_ear & mine, 1 - ew, weights["head"])
    # Tail: the points of the tail tube (close to its centre line), fading in where it
    # leaves the rump so the skin around the root bends softly.
    ts, td, _ = _project(co, [np.array(p) for p in tail])
    tmask = td < 0.022
    t1, t2 = _chain_weights(ts, [(s_skin, s_mid), (s_mid, 99)], 0.015)
    weights["tail1"] = np.where(tmask, t1, 0)
    weights["tail2"] = np.where(tmask, t2, 0)
    # Legs: two bones each, fading in below the trunk so the joint with the body bends softly.
    for leg_name, leg in zip(LEGS, info["legs"]):
        pts = [np.array(p) for p in leg]
        ls, ld, _ = _project(co, pts)
        upper_len = float(np.linalg.norm(pts[1] - pts[0]))
        w_up, w_low = _chain_weights(ls, [(0.12, upper_len), (upper_len, 99)], 0.08)
        near = 1 - _smooth(0.09, 0.17, ld)         # the whole leg, fading out into the trunk
        weights[leg_name + ".upper"] = w_up * near
        weights[leg_name + ".lower"] = w_low * near
    # Belly: the middle of the barrel, fading out towards the shoulders, hams and legs.
    belly = _smooth(-0.5, -0.3, x) * (1 - _smooth(0.15, 0.35, x)) * _smooth(0.22, 0.32, z)
    others = sum(w for b, w in weights.items() if b not in ("body", "belly"))
    weights["belly"] = belly * np.clip(1 - others, 0, 1)
    # Body takes what is left.
    total = sum(w for b, w in weights.items() if b != "body")
    weights["body"] = np.clip(1 - total, 0, 1)
    total = sum(weights.values())
    for b in weights:
        weights[b] = weights[b] / np.maximum(total, 1e-6)

    for b, w in weights.items():
        vg = obj.vertex_groups.new(name=b)
        for i in np.nonzero(w > 0.002)[0]:
            vg.add([int(i)], float(w[i]), "REPLACE")
    mod = obj.modifiers.new("rig", "ARMATURE")
    mod.object = rig
    obj.parent = rig
    return rig


def pose_pig(rig, pose, t, seed):
    """Sets the pig's bones for loop phase t in [0, 1). seed varies timing between pigs.
    Every motion completes whole cycles per loop, so the animation loops seamlessly."""
    rad = math.radians
    tau = 2 * math.pi
    ph = (seed * 0.618) % 1.0
    up, side, fwd = V((0, 0, 1)), V((0, 1, 0)), V((1, 0, 0))
    pbs = rig.pose.bones
    for pb in pbs:
        pb.rotation_mode = "QUATERNION"
        pb.rotation_quaternion = Quaternion()
        pb.scale = (1, 1, 1)
    if pose == "eating":
        # Snout working in the feed: quick small dips and a slow sideways shift.
        dip = 0.5 - 0.5 * math.cos(tau * 3 * (t + ph))
        _add_rot(pbs["head"], side, rad(4) * dip)
        _add_rot(pbs["neck"], up, rad(3) * math.sin(tau * (t + ph)))
    elif pose == "rooting":
        # Pushing the snout through the soil: two forward shoves per loop, sweeping sideways.
        shove = _pulse(t, ph + 0.2, 0.08) + _pulse(t, ph + 0.7, 0.08)
        _add_rot(pbs["neck"], side, rad(4) * shove)
        _add_rot(pbs["head"], side, -rad(5) * shove)
        _add_rot(pbs["neck"], up, rad(5) * math.sin(tau * (t + ph)))
    elif pose == "lying":
        # Asleep on its side but restless, like a dreaming pig: two deep breaths, the head
        # lifting off the ground and settling again, and the legs paddling lazily, each at its
        # own time. (Rolled onto its side, the pig's own up axis lies flat, so turning about
        # it lifts the head; the side axis now points up, so the legs swing along the ground.)
        breath = 0.5 - 0.5 * math.cos(tau * 2 * (t + ph))
        pbs["belly"].scale = (1 + 0.05 * breath, 1, 1 + 0.05 * breath)
        lift = _pulse(t, ph + 0.35, 0.1)
        _add_rot(pbs["neck"], up, rad(10) * lift)
        _add_rot(pbs["head"], up, rad(6) * lift)
        for k, leg in enumerate(LEGS):
            a = tau * (t + ph + 0.27 * k)
            _add_rot(pbs[leg + ".upper"], side, rad(9) * math.sin(a))
            _add_rot(pbs[leg + ".lower"], side, rad(14) * math.sin(a + 0.9))
    else:  # standing: slow look around
        turn = math.sin(tau * (t + ph))
        _add_rot(pbs["neck"], up, rad(7) * turn)
        _add_rot(pbs["head"], up, rad(5) * turn)
        _add_rot(pbs["head"], side, rad(3) * math.sin(tau * 2 * (t + ph) + 1.0))
    # Ears: one flick back each per loop, at different times.
    for k, ear in enumerate(("ear.R", "ear.L")):
        flick = _pulse(t, (ph + 0.15 + 0.5 * k) % 1.0, 0.04)
        _add_rot(pbs[ear], side, -rad(22) * flick)
    # Tail: pigs wag their curly tails, asleep too.
    wag = rad(18) * math.sin(tau * 2 * (t + ph))
    _add_rot(pbs["tail1"], up, wag)
    _add_rot(pbs["tail2"], up, wag * 0.6)
