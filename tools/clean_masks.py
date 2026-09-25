#!/usr/bin/env python3
"""Drop hair-thin structures (whiskers, stray strands) from a view set's masks.

A whisker is two or three pixels wide in the front view and hidden inside the
head from the side, so the visual hull keeps it as a thin fin sticking out of
the cheek -- geometry nobody wants in a mesh, and the seed of spikes in every
fit after. It belongs in a texture.

An opening with a disk removes it, but also rounds every convex corner (ear
tips). So a piece the opening removes is dropped only if it is thin (median
inscribed radius at most --max-radius px) *and* reaches far out of the opened
body (at least --min-reach px): on the cat's front view the whiskers measure
a median radius of 2 px and reach 18-53 px, while a corner shaving is never
further out than the disk. The original masks are kept in mask_raw/.

  python3 tools/clean_masks.py --views data/cat/views --radius 6 --max-radius 2.5 --min-reach 12
"""
import argparse
import glob
import os
import shutil

import numpy as np
from PIL import Image
from scipy import ndimage


def clean(m, radius, max_radius, min_reach):
    disk = np.hypot(*np.mgrid[-radius:radius + 1, -radius:radius + 1]) <= radius
    opened = ndimage.binary_opening(m, structure=disk)
    thick = ndimage.distance_transform_edt(m)          # inscribed radius, px
    reach = ndimage.distance_transform_edt(~opened)    # how far out of the body, px
    lab, n = ndimage.label(m & ~opened)
    if not n:
        return m, 0
    idx = range(1, n + 1)
    med = np.asarray(ndimage.median(thick, lab, idx))
    far = np.asarray(ndimage.maximum(reach, lab, idx))
    drop = np.isin(lab, 1 + np.nonzero((med <= max_radius) & (far >= min_reach))[0])
    out = m & ~drop
    # a piece cut loose from the body by the removal is dropped too
    lab, n = ndimage.label(out)
    if n > 1:
        size = ndimage.sum(out, lab, range(1, n + 1))
        out = np.isin(lab, 1 + np.nonzero(size >= 0.01 * size.max())[0])
    return out, int((m & ~out).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", required=True)
    ap.add_argument("--radius", type=int, default=6)
    ap.add_argument("--max-radius", type=float, default=2.5)
    ap.add_argument("--min-reach", type=float, default=12.0)
    a = ap.parse_args()
    raw = os.path.join(a.views, "mask_raw")
    if not os.path.isdir(raw):
        shutil.copytree(os.path.join(a.views, "mask"), raw)
    for p in sorted(glob.glob(os.path.join(raw, "*.png"))):
        m = np.asarray(Image.open(p).convert("L")) > 127
        out, n = clean(m, a.radius, a.max_radius, a.min_reach)
        Image.fromarray((out * 255).astype(np.uint8)).save(
            os.path.join(a.views, "mask", os.path.basename(p)))
        print(f"{os.path.basename(p)[:-4]:10s} dropped {n} px")


if __name__ == "__main__":
    main()
