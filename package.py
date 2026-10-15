"""Builds the mod zip for the mod portal: <name>_<version>.zip next to this file, holding the
game files in a <name>_<version>/ folder as Factorio expects. The Blender sources and the
repository files are left out.

    python package.py
"""

import json
import os
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
FILES = ["info.json", "changelog.txt", "thumbnail.png", "data.lua", "data-updates.lua", "data-final-fixes.lua",
         "control.lua"]
FOLDERS = ["graphics", "prototypes", "scripts"]


def main():
    with open(os.path.join(HERE, "info.json"), encoding="utf8") as fh:
        info = json.load(fh)
    top = f"{info['name']}_{info['version']}"
    out = os.path.join(HERE, top + ".zip")
    paths = list(FILES)
    for folder in FOLDERS:
        for root, _, files in os.walk(os.path.join(HERE, folder)):
            for f in sorted(files):
                paths.append(os.path.relpath(os.path.join(root, f), HERE))
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for rel in sorted(paths):
            # PNGs are compressed already: store them as they are.
            kind = zipfile.ZIP_STORED if rel.endswith(".png") else zipfile.ZIP_DEFLATED
            z.write(os.path.join(HERE, rel), top + "/" + rel.replace(os.sep, "/"), compress_type=kind)
    print(f"Wrote {out}: {len(paths)} files, {os.path.getsize(out) / 2**20:.1f} MB")


if __name__ == "__main__":
    main()
