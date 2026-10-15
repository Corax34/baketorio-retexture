"""Skeleton, skin weights and motion for the sheep model (see cow_rig.py for the method).

Bones (sheep-local space, sheep faces +x):
    body                 static: fleece and legs
    body -> belly        breathing (scaled, used by lying sheep)
    body -> neck -> head -> ear.L, ear.R
    body -> tail
    body -> leg.FR.upper -> leg.FR.lower    (and FL, HR, HL: front/hind, right/left)
"""

import math

import bpy
import numpy as np
from mathutils import Quaternion, Vector

from cow_rig import _add_rot, _chain_weights, _project, _pulse, _smooth

V = Vector
LEGS = ("leg.FR", "leg.HR", "leg.FL", "leg.HL")   # in the order of the model's leg paths


def rig_sheep(obj, info, name):
    """Adds an armature with skin weights to the sheep mesh object (mesh in sheep-local
    space). Returns the armature object; obj becomes its child with an Armature modifier."""
    coll = bpy.context.scene.collection
    arm_data = bpy.data.armatures.new(name + "-rig")
    rig = bpy.data.objects.new(name + "-rig", arm_data)
    coll.objects.link(rig)
    path = [V(p) for p in info["path"]]          # neck base, poll, mid face, muzzle tip
    tail = [V(p) for p in info["tail"]]
    drop = info["drop"]

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

        z = 0.6 - drop
        add("body", V((-0.5, 0, z)), V((0.4, 0, z)))
        add("belly", V((-0.35, 0, z)), V((0.2, 0, z)), "body")
        add("neck", path[0], path[1], "body")
        add("head", path[1], path[3], "neck")
        for side_name, (base, tip) in zip(("ear.R", "ear.L"), info["ears"]):
            add(side_name, V(base), V(tip), "head")
        add("tail", tail[1], tail[-1], "body")             # from where it leaves the rump
        for leg_name, leg in zip(LEGS, info["legs"]):        # shoulder or hip, knee, hoof
            add(leg_name + ".upper", V(leg[0]), V(leg[1]), "body")
            add(leg_name + ".lower", V(leg[1]), V(leg[3]), leg_name + ".upper")
        bpy.ops.object.mode_set(mode="OBJECT")

    me = obj.data
    n = len(me.vertices)
    co = np.empty(n * 3, dtype=np.float64)
    me.vertices.foreach_get("co", co)
    co = co.reshape(n, 3)
    x, y, zc = co[:, 0], co[:, 1], co[:, 2]
    weights = {b: np.zeros(n) for b in bones}

    # Neck and head along neck base -> poll -> muzzle; the front legs are left out.
    s, d, _ = _project(co, [np.array(p) for p in path])
    L1 = (path[1] - path[0]).length
    near_leg = np.zeros(n, bool)
    for leg in info["legs"][0::2]:                       # front legs
        _, ld, _ = _project(co, [np.array(q) for q in leg])
        near_leg |= (ld < 0.07) & (zc < 0.45 - drop)
    in_front = (d < 0.28) & (x > 0.2) & ~near_leg
    w_neck, w_head = _chain_weights(s, [(0.1, L1), (L1, 99)], 0.1)
    # The neck bends where it leaves the shoulders, not in the chest and shoulder wool.
    clear = _smooth(0.3, 0.44, x)
    weights["neck"] = np.where(in_front, w_neck * clear, 0)
    weights["head"] = np.where(in_front, w_head, 0)
    # A ram's horns are solid and grow from the skull: they follow the head entirely.
    mat_of = np.zeros(n, dtype=int)
    for poly in me.polygons:
        mat_of[list(poly.vertices)] = poly.material_index
    horn = np.isin(mat_of, [i for i, m in enumerate(me.materials) if m.name.split(".")[0] == "ram_horn"])
    weights["neck"] = np.where(horn, 0, weights["neck"])
    weights["head"] = np.where(horn, 1.0, weights["head"])
    # Ears: points close to an ear segment, beyond its base.
    for bname in ("ear.R", "ear.L"):
        base, tip = bones[bname]
        es, ed, _ = _project(co, [np.array(base), np.array(tip)])
        ew = np.where(ed < 0.04, _smooth(0.01, 0.03, es), 0) * (weights["head"] > 0) * ~horn
        weights[bname] = ew
        weights["head"] = weights["head"] * (1 - ew)
    # Tail: the hanging tail, fading in where it leaves the rump so the wool there bends softly.
    ts, td, _ = _project(co, [np.array(p) for p in tail])
    weights["tail"] = (1 - _smooth(0.055, 0.08, td)) * _smooth(0.03, 0.09, ts)
    # Legs: two bones each, fading in below the fleece so the joint bends softly.
    for leg_name, leg in zip(LEGS, info["legs"]):
        pts = [np.array(p) for p in leg]
        ls, ld, _ = _project(co, pts)
        upper_len = float(np.linalg.norm(pts[1] - pts[0]))
        w_up, w_low = _chain_weights(ls, [(0.15, upper_len), (upper_len, 99)], 0.06)
        near = 1 - _smooth(0.05, 0.1, ld)
        weights[leg_name + ".upper"] = w_up * near
        weights[leg_name + ".lower"] = w_low * near
    # Belly: the middle of the fleece, fading out towards the shoulders and rump.
    belly = _smooth(-0.5, -0.3, x) * (1 - _smooth(0.15, 0.35, x)) * _smooth(0.3, 0.4, zc + drop)
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


def pose_sheep(rig, pose, t, seed):
    """Sets the sheep's bones for loop phase t in [0, 1). seed varies timing between sheep.
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
    if pose == "grazing":
        # Nibbling the grass: quick tugs, the muzzle working slowly sideways.
        tug = _pulse(t, ph + 0.1, 0.05) + _pulse(t, ph + 0.43, 0.05) + _pulse(t, ph + 0.76, 0.05)
        _add_rot(pbs["head"], side, -rad(6) * tug)
        _add_rot(pbs["neck"], up, rad(7) * math.sin(tau * (t + ph)))
    elif pose == "eating":
        # Pulling hay from the trough twice, chewing in between.
        pull = _pulse(t, ph + 0.2, 0.07) + _pulse(t, ph + 0.7, 0.07)
        _add_rot(pbs["neck"], side, -rad(5) * pull)
        _add_rot(pbs["head"], side, rad(3) * math.sin(tau * 4 * (t + ph)))
    elif pose == "drinking":
        # Small dips while drinking.
        dip = 0.5 - 0.5 * math.cos(tau * 3 * (t + ph))
        _add_rot(pbs["head"], side, rad(4) * dip)
    elif pose == "lying":
        # Chewing the cud: the head bobs with each chew and turns once; slow breathing.
        _add_rot(pbs["head"], side, rad(2.5) * math.sin(tau * 4 * (t + ph)))
        _add_rot(pbs["neck"], up, rad(10) * math.sin(tau * (t + ph)))
        breath = 0.5 - 0.5 * math.cos(tau * 2 * (t + ph))
        pbs["belly"].scale = (1 + 0.04 * breath, 1, 1 + 0.04 * breath)
    else:  # standing: looking around
        turn = math.sin(tau * (t + ph))
        _add_rot(pbs["neck"], up, rad(10) * turn)
        _add_rot(pbs["head"], up, rad(6) * turn)
        _add_rot(pbs["head"], side, rad(3) * math.sin(tau * 2 * (t + ph) + 1.0))
    # Ears: one flick each per loop at different times.
    for k, ear in enumerate(("ear.R", "ear.L")):
        flick = _pulse(t, (ph + 0.3 + 0.5 * k) % 1.0, 0.04)
        _add_rot(pbs[ear], fwd, rad(25) * flick * (1 if k else -1))
    # Tail: two short swishes per loop, a few quick wags side to side each. The tail hangs
    # down, so it swings about the axis across it in the sheep's middle plane, not about up.
    bone = pbs["tail"].bone
    across = (bone.tail_local - bone.head_local).normalized().cross(side)
    swish = _pulse(t, ph + 0.1, 0.08) + 0.7 * _pulse(t, ph + 0.6, 0.06)
    _add_rot(pbs["tail"], across, rad(28) * swish * math.sin(tau * 8 * (t + ph)))
