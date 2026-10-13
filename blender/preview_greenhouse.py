"""Composite the greenhouse sprites over grass the way Factorio layers them.

    blender -b --factory-startup --python blender/preview_greenhouse.py -- <out.png> [crop,crop,...] [stage]

Draws both orientations with every crop, using the shifts in
prototypes/greenhouse-sprites.lua.
"""

import bpy
import numpy as np
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GFX = os.environ.get("GH_OUT_DIR") or os.path.join(REPO, "graphics")
ARGS = sys.argv[sys.argv.index("--") + 1:]
OUT = ARGS[0]
PX = 64  # pixels per tile in the sprite files

meta = {}
for line in open(os.path.join(REPO, "prototypes", "greenhouse-sprites.lua")):
    m = re.search(r'\["(.+?)"\] = \{ width = (\d+), height = (\d+), shift = \{ (\S+), (\S+) \}', line)
    if m:
        meta[m[1]] = (int(m[2]), int(m[3]), float(m[4]), float(m[5]))


def load(path):
    img = bpy.data.images.load(path)
    a = np.array(img.pixels[:], dtype=np.float32).reshape(img.size[1], img.size[0], 4)[::-1]
    return a  # top row first


def blend(canvas, sprite, cx, cy, shadow=False):
    """Paste sprite centred at canvas pixel (cx, cy)."""
    h, w = sprite.shape[:2]
    x0, y0 = int(round(cx - w / 2)), int(round(cy - h / 2))
    region = canvas[y0:y0 + h, x0:x0 + w]
    a = sprite[..., 3:4]
    if shadow:
        region[..., :3] *= 1 - a * 0.55
    else:
        region[..., :3] = region[..., :3] * (1 - a) + sprite[..., :3] * a


STAGE = int(ARGS[2]) if len(ARGS) > 2 else 8   # harvest stage to show
crops = ARGS[1].split(",") if len(ARGS) > 1 and ARGS[1] else ["wheat", "sugarcane", "cocoa", "cinnamon", "blueberries", "strawberries",
     "apples", "peaches", "lemons", "nuts", "pumpkin", "wood"]
cell = {"ns": (13, 12), "ew": (13, 10)}
rows = []
for orient in ("ns", "ew"):
    cw, ch = cell[orient]
    canvas = np.zeros((ch * PX, cw * PX * len(crops) + 4 * PX, 4), dtype=np.float32)
    canvas[...] = (0.28, 0.36, 0.17, 1)
    rng = np.random.default_rng(1)
    canvas[..., :3] *= rng.uniform(0.9, 1.1, size=canvas.shape[:2])[..., None]
    for i, crop in enumerate(crops):
        cx, cy = (i + 0.5) * cw * PX, ch * PX / 2 + PX
        for name, file, shadow in (("shadow-" + orient, f"greenhouse-{orient}-shadow.png", True),
                                   ("main-" + orient, f"greenhouse-{orient}.png", False),
                                   (f"{crop}-{orient}-{STAGE}", f"greenhouse-{crop}-{orient}-{STAGE}.png", False)):
            path = os.path.join(GFX, file)
            if not os.path.exists(path):
                continue
            w, h, sx, sy = meta[name]
            blend(canvas, load(path), cx + sx * PX, cy + sy * PX, shadow)
    rows.append(canvas)

width = max(r.shape[1] for r in rows)
out = np.concatenate([np.pad(r, ((0, 0), (0, width - r.shape[1]), (0, 0))) for r in rows], axis=0)
img = bpy.data.images.new("preview", out.shape[1], out.shape[0], alpha=True)
img.pixels[:] = out[::-1].ravel()
img.filepath_raw = OUT
img.file_format = "PNG"
img.save()
print("Wrote", OUT)
