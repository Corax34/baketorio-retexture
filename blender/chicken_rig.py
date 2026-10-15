"""Skeleton, skin weights and motion for the hen (see cow_rig.py for the method).

Bones (hen-local space, hen faces +x):
    body                 static: body and wings
    body -> belly        breathing (scaled, used by the sitting hen)
    body -> neck1 -> neck2 -> head      the S of the neck, then head with comb, wattles, beak
    body -> tail
    body -> leg.R.upper -> leg.R.lower  (and leg.L): thigh to hock, shank and toes
    body -> wing.R -> wing.R.f0 ... f6  (and wing.L): the shoulder lifts the wing, each feather
                         bone fans its feather (with its covert) open and slides it out along
                         the arm
    pf.0 ... pf.7        loose feathers of a flapping hen, moved by a small particle simulation
"""

import math

import bpy
import numpy as np
from mathutils import Euler, Quaternion, Vector

from chicken_model import PART_PARTICLE, PART_SHOULDER, PART_WING, WING_FEATHERS
from cow_rig import _add_rot, _chain_weights, _project, _pulse, _smooth

V = Vector
LEGS = ("leg.R", "leg.L")       # in the order of the model's legs (right is -y)
LOOP_SECONDS = 1.6              # one animation loop in game: 48 frames at 0.5 frames per tick
ARM_STRETCH = 0.016             # how far each feather root slides out along the arm, per feather


def _step(t, at, width):
    """Periodic 0 -> 1 step around phase 'at' over 'width' (smooth), back to 0 at t = 0."""
    d = (t - at) % 1.0
    return _smooth(0.0, width, d) if d < 0.5 + width else 1 - _smooth(0.5 + width, 0.5 + 2 * width, d)


def rig_chicken(obj, info, name):
    """Adds an armature with skin weights to the hen mesh object (mesh in hen-local space).
    Returns the armature object; obj becomes its child with an Armature modifier."""
    coll = bpy.context.scene.collection
    arm_data = bpy.data.armatures.new(name + "-rig")
    rig = bpy.data.objects.new(name + "-rig", arm_data)
    coll.objects.link(rig)
    head, face = V(info["head"]), V(info["face"])
    beak = head + face * 0.06
    path = [V(info["neck_base"]), V(info["neck_bend"]), head, beak]
    tail = [V(p) for p in info["tail"]]
    z = 0.23 - info["drop"]

    bones = {}
    bpy.context.view_layer.objects.active = rig
    with bpy.context.temp_override(active_object=rig, object=rig):
        bpy.ops.object.mode_set(mode="EDIT")
        eb = arm_data.edit_bones

        def add(bname, h, t, parent=None):
            b = eb.new(bname)
            b.head, b.tail = h, t
            if parent:
                b.parent = eb[parent]
            bones[bname] = (V(h), V(t))

        add("body", V((-0.12, 0, z)), V((0.1, 0, z)))
        add("belly", V((-0.08, 0, z)), V((0.06, 0, z)), "body")
        add("neck1", path[0], path[1], "body")
        add("neck2", path[1], path[2], "neck1")
        add("head", path[2], path[3], "neck2")
        add("tail", V(info["tail_root"]), tail[-1], "body")
        for leg_name, leg in zip(LEGS, info["legs"]):           # hip, hock, foot
            add(leg_name + ".upper", V(leg[0]), V(leg[1]), "body")
            add(leg_name + ".lower", V(leg[1]), V(leg[2]), leg_name + ".upper")
        for w in info["wings"]:
            wn = "wing." + ("L" if w["side"] > 0 else "R")
            add(wn, w["pivot"], w["pivot"] + w["back"] * 0.05, "body")
            for i, (root, d) in enumerate(zip(w["roots"], w["dirs"])):
                add(f"{wn}.f{i}", root, root + d * 0.06, wn)
        for k, c in enumerate(info["particles"]):                # pointing +y: bone axes = armature axes
            add(f"pf.{k}", V(c), V(c) + V((0, 0.03, 0)))
        bpy.ops.object.mode_set(mode="OBJECT")

    me = obj.data
    n = len(me.vertices)
    co = np.empty(n * 3, dtype=np.float64)
    me.vertices.foreach_get("co", co)
    co = co.reshape(n, 3)
    x, zc = co[:, 0], co[:, 2]
    weights = {b: np.zeros(n) for b in bones}
    part = np.zeros(n, dtype=np.int32)
    me.attributes["part"].data.foreach_get("value", part)

    # Neck and head along neck base -> bend -> head -> beak. Feathers follow them only within
    # the neck's own thickness at each point (so a breast close to a lowered head stays put);
    # the comb, wattles, beak and eyes always follow the head.
    s, d, _ = _project(co, [np.array(p) for p in path])
    L1 = (path[1] - path[0]).length
    L2 = L1 + (path[2] - path[1]).length - 0.02
    radius = np.interp(s, [0.0, (L1 + L2) / 2, L2, 99], [0.062, 0.046, 0.032, 0.03])
    near = 1 - _smooth(radius + 0.004, radius + 0.014, d)
    mat_of = np.zeros(n, dtype=int)
    for poly in me.polygons:
        mat_of[list(poly.vertices)] = poly.material_index
    head_parts = np.isin(mat_of, [i for i, m in enumerate(me.materials)
                                  if m.name.split(".")[0] in ("comb_red", "beak_yellow", "eye_black")])
    nest_parts = np.isin(mat_of, [i for i, m in enumerate(me.materials)
                                  if m.name.split(".")[0] in ("straw", "straw_old", "egg_shell")])
    w1, w2, wh = _chain_weights(s, [(0.03, L1), (L1, L2), (L2, 99)], 0.03)
    for bname, w in (("neck1", w1), ("neck2", w2), ("head", wh)):
        weights[bname] = np.where(head_parts, 0, w * near)
    weights["head"] = np.where(head_parts, 1.0, weights["head"])
    # Tail fan: everything close to the line from its root out along the fan.
    ts, td, _ = _project(co, [np.array(info["tail_root"]), np.array(tail[-1]) + (tail[-1] - V(info["tail_root"]))])
    weights["tail"] = np.where(head_parts, 0, (1 - _smooth(0.05, 0.08, td)) * _smooth(0.0, 0.04, ts)
                               * (1 - _smooth(-0.1, -0.06, x)))
    # Legs: thigh and shank with the toes; fading into the feathers at the top.
    for leg_name, leg in zip(LEGS, info["legs"]):
        pts = [np.array(p) for p in leg]
        ls, ld, _ = _project(co, pts)
        upper_len = float(np.linalg.norm(pts[1] - pts[0]))
        w_up, w_low = _chain_weights(ls, [(0.03, upper_len), (upper_len, 99)], 0.02)
        close = 1 - _smooth(0.03, 0.06, np.minimum(ld, np.linalg.norm(co - pts[2], axis=1) - 0.035))
        weights[leg_name + ".upper"] = np.where(head_parts, 0, w_up * close)
        weights[leg_name + ".lower"] = np.where(head_parts, 0, w_low * close)
    for bname in weights:                               # the nest and its eggs stay put
        weights[bname] = np.where(nest_parts | (part != 0), 0, weights[bname])
    # Belly: the middle of the body.
    belly = _smooth(-0.12, -0.05, x) * (1 - _smooth(0.04, 0.1, x)) * ~nest_parts * (part == 0)
    others = sum(w for b, w in weights.items() if b not in ("body", "belly"))
    weights["belly"] = belly * np.clip(1 - others, 0, 1)
    # Separate feathers belong wholly to their own bone.
    for side, letter in ((0, "R"), (100, "L")):
        if f"wing.{letter}" in weights:
            weights[f"wing.{letter}"] = (part == PART_SHOULDER + side).astype(float)
            for i in range(len(WING_FEATHERS)):
                weights[f"wing.{letter}.f{i}"] = (part == PART_WING + side + i).astype(float)
    for k in range(len(info["particles"])):
        weights[f"pf.{k}"] = (part == PART_PARTICLE + k).astype(float)
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


def pose_chicken(rig, pose, t, seed):
    """Sets the hen's bones for loop phase t in [0, 1). seed varies timing between hens.
    Every motion completes whole cycles per loop, so the animation loops seamlessly."""
    rad = math.radians
    tau = 2 * math.pi
    ph = (seed * 0.618) % 1.0
    up, side = V((0, 0, 1)), V((0, 1, 0))
    pbs = rig.pose.bones
    for pb in pbs:
        pb.rotation_mode = "QUATERNION"
        pb.rotation_quaternion = Quaternion()
        pb.scale = (1, 1, 1)
        pb.location = (0, 0, 0)
    for pb in _particle_bones(pbs):                       # loose feathers: hidden unless flying
        pb.scale = (0.0001,) * 3
    if pose in ("pecking", "feeding"):
        # Quick pecks: a jab of the head with only a little give in the neck, the head turning
        # a little between them.
        times = (0.1, 0.35, 0.6, 0.8) if pose == "feeding" else (0.15, 0.45, 0.75)
        peck = sum(_pulse(t, ph + a, 0.025) for a in times)
        _add_rot(pbs["neck1"], side, rad(5) * peck)
        _add_rot(pbs["neck2"], side, rad(4) * peck)
        _add_rot(pbs["head"], side, rad(16) * peck)
        _add_rot(pbs["neck1"], up, rad(8) * math.sin(tau * (t + ph)))
    elif pose == "scratching":
        # Two backward kicks of the right foot, then a look down and a peck.
        kick = _pulse(t, ph + 0.1, 0.04) + _pulse(t, ph + 0.3, 0.04)
        _add_rot(pbs["leg.R.upper"], side, rad(25) * kick)
        _add_rot(pbs["leg.R.lower"], side, rad(30) * kick)
        peck = _pulse(t, ph + 0.62, 0.03)
        _add_rot(pbs["neck1"], side, rad(8) * peck)
        _add_rot(pbs["neck2"], side, rad(6) * peck)
        _add_rot(pbs["head"], side, rad(18) * peck)
        _add_rot(pbs["neck1"], up, rad(10) * math.sin(tau * (t + ph)))
    elif pose in ("sitting", "brooding"):
        breath = 0.5 - 0.5 * math.cos(tau * 2 * (t + ph))
        pbs["belly"].scale = (1 + 0.04 * breath, 1, 1 + 0.04 * breath)
        _add_rot(pbs["neck1"], up, rad(12) * math.sin(tau * (t + ph)))
    elif pose == "flapping":
        _flapping(rig, pbs, (t + ph) % 1.0)
    else:  # standing: the jerky look-around of a hen, quick turns held still in between
        turn = -25 + 45 * _step(t, ph + 0.1, 0.03) - 20 * _step(t, ph + 0.35, 0.03)
        _add_rot(pbs["neck2"], up, rad(turn) * 0.5)
        _add_rot(pbs["head"], up, rad(turn) * 0.5)
        tilt = _pulse(t, ph + 0.7, 0.06)                     # cocks the head to look up
        _add_rot(pbs["head"], V(info_face(pbs)), rad(25) * tilt)
    # Tail: a small twitch now and then.
    twitch = _pulse(t, ph + 0.5, 0.03)
    _add_rot(pbs["tail"], up, rad(8) * twitch)


def _wing_frames(rig):
    """Per wing (letter, pivot, back, down, out, lift sign, fan sign), from the rest bones."""
    bones = rig.data.bones
    out_frames = []
    for letter, s in (("L", 1), ("R", -1)):
        b = bones.get("wing." + letter)
        if b is None:
            continue
        back = (b.tail_local - b.head_local).normalized()
        last = len(WING_FEATHERS) - 1
        span = bones[f"wing.{letter}.f{last}"].head_local - bones[f"wing.{letter}.f0"].head_local
        down = (span - back * span.dot(back)).normalized()
        out = back.cross(down)
        if out.y * s < 0:
            out = -out
        lift_sign = 1 if (-back).cross(down).dot(out) > 0 else -1     # raising swings 'down' outwards
        fan_sign = 1 if out.cross(back).dot(down) > 0 else -1         # fanning swings feathers to 'down'
        out_frames.append((letter, b.head_local.copy(), back, down, out, lift_sign, fan_sign))
    return out_frames


def _pose_wings(rig, pbs, raise_deg, fan, sweep_deg=0.0):
    """Lifts both wings raise_deg about the line they fold along (0 = folded against the
    body, 90 = straight out, more = up over the back), opens their feathers (fan 0..1:
    primaries swing out most, the arm stretches) and sweeps them forward."""
    rad = math.radians
    n = len(WING_FEATHERS)
    for letter, pivot, back, down, out, lift_sign, fan_sign in _wing_frames(rig):
        w = pbs["wing." + letter]
        _add_rot(w, -back, lift_sign * rad(raise_deg))
        _add_rot(w, V((0, 0, 1)), rad(sweep_deg) * (1 if letter == "R" else -1))
        for i in range(n):
            pb = pbs[f"wing.{letter}.f{i}"]
            k = i / (n - 1)
            _add_rot(pb, out, fan_sign * rad(78) * fan * k ** 1.3)
            rest = pb.bone.matrix_local.to_3x3()
            pb.location = rest.inverted() @ (down * (ARM_STRETCH * i * fan))


def _flapping(rig, pbs, t):
    """A burst of flapping: the wings come out, beat four times (most open on the
    downstroke, the downstroke quicker than the upstroke) and fold again, the neck stretching
    up and the tail fanning meanwhile. Loose feathers fly off at the bottom of the strokes."""
    rad = math.radians
    tau = 2 * math.pi
    env = _smooth(0.05, 0.14, t) * (1 - _smooth(0.6, 0.7, t))
    t_a, t_b, beats = 0.1, 0.66, 4
    phase = (t - t_a) / (t_b - t_a) * beats
    beat = math.cos(tau * phase + 0.35 * math.sin(tau * phase))       # +1 top, -1 bottom
    _pose_wings(rig, pbs, env * (100 + 28 * beat), env * (0.82 - 0.18 * beat), env * 8 * -beat)
    _add_rot(pbs["neck1"], V((0, 1, 0)), -rad(8) * env)
    _add_rot(pbs["head"], V((0, 1, 0)), rad(6) * env * beat)
    _add_rot(pbs["tail"], V((0, 1, 0)), -rad(12) * env)
    emit = [t_a + (k // 2 + 0.55 + 0.12 * (k % 2)) / beats * (t_b - t_a) for k in range(len(_particle_bones(pbs)))]
    _particles(rig, pbs, t, emit)


def _particle_bones(pbs):
    return [pb for pb in pbs if pb.name.startswith("pf.")]


def _particles(rig, pbs, t, emit):
    """Loose feathers, each thrown off one wing at its emission time (loop phase): tossed
    outwards and up, slowed by the air to a slow, rocking, falling-leaf descent, lying on the
    ground a while and then gone, all before the loop comes round again. Deterministic, so
    the loop is seamless."""
    import random
    frames = {f[0]: f for f in _wing_frames(rig)}
    g_term = V((0, 0, -0.38))        # terminal velocity of a falling feather, m/s
    tau = 0.16                       # how quickly the air takes up the throw, s
    for k, pb in enumerate(_particle_bones(pbs)):
        rng = random.Random(f"feather{k}")
        letter = "L" if k % 2 else "R"
        _, pivot, back, down, out, _, _ = frames[letter]
        p0 = pivot + out * rng.uniform(0.12, 0.26) + V((0, 0, rng.uniform(-0.02, 0.06))) + back * rng.uniform(0.0, 0.1)
        v0 = out * rng.uniform(0.35, 0.8) + V((0, 0, rng.uniform(0.1, 0.45))) + back * rng.uniform(-0.15, 0.3)
        sway_dir = out.cross(V((0, 0, 1))).normalized()
        flutter, phi = rng.uniform(1.8, 2.6), rng.uniform(0, 6.28)
        spin = rng.uniform(-2.5, 2.5)

        def where(a):
            pos = p0 + g_term * a + (v0 - g_term) * (tau * (1 - math.exp(-a / tau)))
            return pos + sway_dir * (0.022 * math.sin(6.283 * flutter * a + phi) * (1 - math.exp(-a / 0.3)))

        lo, hi = 0.0, 4.0                                  # landing: bisection on the height
        for _ in range(40):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if where(mid).z > 0.005 else (lo, mid)
        land = hi
        life = land / LOOP_SECONDS + 0.2 + 0.06
        age = ((t - emit[k]) % 1.0) * LOOP_SECONDS          # seconds since it came off
        a_loop = age / LOOP_SECONDS
        rest_centre = pb.bone.head_local
        rest = pb.bone.matrix_local.to_3x3()
        if a_loop >= min(life, 0.97):
            pb.scale = (0.0001,) * 3
            pb.location = (0, 0, 0)
            continue
        scale = min(1.0, 0.25 + a_loop / 0.03 * 0.75) * min(1.0, (min(life, 0.97) - a_loop) / 0.06)
        yaw0 = rng.uniform(0, 6.28)
        if age < land:
            pos = where(age)
            rot = Euler((math.radians(40) * math.sin(6.283 * flutter * age + phi),
                         math.radians(18) * math.sin(6.283 * flutter * 0.5 * age + phi * 0.7),
                         yaw0 + spin * age), "XYZ").to_matrix()
        else:
            pos = where(land)
            pos.z = 0.004
            rot = Euler((0, 0, yaw0 + spin * land), "XYZ").to_matrix()
        pb.location = rest.inverted() @ (pos - rest_centre)
        pb.rotation_quaternion = (rest.inverted() @ rot @ rest).to_quaternion()
        pb.scale = (scale,) * 3


def info_face(pbs):
    """The head bone's direction (towards the beak) in armature space."""
    b = pbs["head"].bone
    return (b.tail_local - b.head_local).normalized()


# Debug helpers for the inspection scripts ---------------------------------------------

def debug_extreme(rig):
    rad = math.radians
    up, side = V((0, 0, 1)), V((0, 1, 0))
    pbs = rig.pose.bones
    for pb in pbs:
        pb.rotation_mode = "QUATERNION"
        pb.rotation_quaternion = Quaternion()
        pb.scale = (1, 1, 1)
        pb.location = (0, 0, 0)
    _add_rot(pbs["neck1"], up, rad(20))
    _add_rot(pbs["neck2"], side, rad(20))
    _add_rot(pbs["head"], side, rad(20))
    _add_rot(pbs["tail"], up, rad(20))
    pbs["belly"].scale = (1.08, 1, 1.08)
    if "wing.L" in pbs:
        _pose_wings(rig, pbs, 100, 1.0, 6)
    if "leg.R.upper" in pbs:
        _add_rot(pbs["leg.R.upper"], side, rad(35))
        _add_rot(pbs["leg.R.lower"], side, rad(40))


def debug_colour_weights(obj, me):
    col = me.color_attributes.new("w", "FLOAT_COLOR", "POINT")
    names = {g.index: g.name for g in obj.vertex_groups}
    for v in me.vertices:
        w = {names[g.group]: g.weight for g in v.groups}
        legs = sum(val for k, val in w.items() if k.startswith("leg"))
        col.data[v.index].color = (w.get("neck1", 0) + w.get("tail", 0), w.get("head", 0) + w.get("neck2", 0) * 0.5,
                                   legs + w.get("neck2", 0) * 0.5, 1)
    m = bpy.data.materials.new("wmat")
    m.use_nodes = True
    nt = m.node_tree
    a = nt.nodes.new("ShaderNodeVertexColor")
    a.layer_name = "w"
    nt.links.new(a.outputs["Color"], nt.nodes["Principled BSDF"].inputs["Base Color"])
    me.materials.clear()
    me.materials.append(m)
