#!/usr/bin/env python3
"""Pose-conditioned multi-view depth (Depth Anything 3) for a view set, on a T4.

No normal estimator runs anywhere in Free3d. What a learned model may give is
where the surface is, from all views at once and with the cameras we already
know -- DA3 takes extrinsics and intrinsics and returns one depth per view
that agree with each other.

DA3 is a pinhole model and our views are orthographic. Each camera is handed
over as a pinhole at distance D along its axis with the focal length that
makes the object plane the same width (f = res * D / ortho_scale): the
further D, the closer to orthographic and the narrower (less familiar) the
field of view, so several D are run. The z-depth that comes back is turned
into the distance along our orthographic ray, w = z - (D - |loc|), in the
frame of cameras.json (engine/tools/fuse_field.project's w).

  push cat --views data/cat/views [--models DA3-LARGE-1.1 DA3-GIANT-1.1] [--dists 4 10]
  pull cat      -> data/cat/da3/<model>_d<D>/<view>.npz  (w, conf; process resolution)

Licences: DA3-LARGE/GIANT weights are CC BY-NC 4.0, DA3-BASE/SMALL Apache-2.0.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kaggle_job import ROOT, STAGE, UNFLATTEN, flatten, owner, pull, push_kernel, tree, upload_dataset, wait  # noqa: E402

RUNNER = UNFLATTEN + r'''
import glob, importlib, json, subprocess, sys, time
import numpy as np
from PIL import Image
t0 = time.time()
def sh(c):
    print("$", c, flush=True); return subprocess.run(c, shell=True).returncode
sh("nvidia-smi --query-gpu=name,memory.total --format=csv")
V = unflatten("views__cameras.json", "/tmp/job")
# --no-deps: DA3's requirements pin numpy<2 and pull xformers, either of which
# would reinstall PyTorch under the running image; missing modules are added
# one at a time instead
sh("pip install -q --no-deps git+https://github.com/ByteDance-Seed/Depth-Anything-3.git")
PIPNAME = {"cv2": "opencv-python", "PIL": "pillow", "yaml": "pyyaml", "skimage": "scikit-image"}
for _ in range(25):
    try:
        from depth_anything_3.api import DepthAnything3
        break
    except ModuleNotFoundError as e:
        mod = e.name.split(".")[0]
        print("missing", mod, flush=True)
        if sh(f"pip install -q --no-deps {PIPNAME.get(mod, mod)}"):
            raise
print(f"install done [{time.time()-t0:.0f}s]", flush=True)
import torch
meta = json.load(open("/tmp/job/views/cameras.json"))
names = list(meta["views"])
o = meta["ortho_scale"]
imgs = []
for v in names:
    rgb = np.asarray(Image.open(f"/tmp/job/views/rgb/{v}.png").convert("RGB")).astype(np.float32)
    m = np.asarray(Image.open(f"/tmp/job/views/mask/{v}.png").convert("L"))[..., None] / 255.0
    # white outside the silhouette: the grey backdrop reads as a wall otherwise
    imgs.append(Image.fromarray((rgb * m + 255 * (1 - m)).astype(np.uint8)))
res = imgs[0].size[0]
for model_name in MODELS:
    model = DepthAnything3.from_pretrained("depth-anything/" + model_name).to("cuda").eval()
    for D in DISTS:
        ext, ixt, shift = [], [], []
        for v in names:
            M = np.array(meta["views"][v]["matrix_world"])
            r, u, b, loc = M[:3, 0], M[:3, 1], M[:3, 2], M[:3, 3]
            C = b * D + (loc - b * (loc @ b))          # same axis as the ortho camera
            R = np.stack([r, -u, -b])                  # OpenCV: x right, y down, z forward
            E = np.eye(4); E[:3, :3] = R; E[:3, 3] = -R @ C
            f = res * D / o
            ext.append(E); ixt.append([[f, 0, res / 2], [0, f, res / 2], [0, 0, 1]])
            shift.append(D - loc @ b)
        t1 = time.time()
        with torch.no_grad():
            pred = model.inference(imgs, extrinsics=np.array(ext, np.float32),
                                   intrinsics=np.array(ixt, np.float32),
                                   process_res=PROCRES, align_to_input_ext_scale=True)
        tag = f"{model_name}_d{D:g}"
        out = f"/kaggle/working/{tag}"
        os.makedirs(out, exist_ok=True)
        for k, v in enumerate(names):
            w = pred.depth[k] - shift[k]
            conf = pred.conf[k] if pred.conf is not None else np.ones_like(w)
            np.savez_compressed(f"{out}/{v}.npz", w=w.astype(np.float32), conf=conf.astype(np.float32))
        print(f"{tag}: depth {pred.depth.shape} range {pred.depth.min():.3f}-{pred.depth.max():.3f} "
              f"[{time.time()-t1:.0f}s]", flush=True)
    del model; torch.cuda.empty_cache()
print(f"all done [{time.time()-t0:.0f}s]", flush=True)
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["push", "pull"])
    ap.add_argument("name")
    ap.add_argument("--views")
    ap.add_argument("--models", nargs="+", default=["DA3-LARGE-1.1"])
    ap.add_argument("--dists", nargs="+", type=float, default=[4.0, 10.0])
    ap.add_argument("--process-res", type=int, default=504,
                    help="multiple of 14; all views attend to each other, so memory grows as its 4th power "
                         "(756 asked a T4 for 18 GB)")
    a = ap.parse_args()
    slug = f"free3d-da3-{a.name}"
    if a.cmd == "push":
        files = [(p, r) for p, r in tree(a.views, "views") if "/mask_raw/" not in r]
        d = os.path.join(STAGE, slug + "-input")
        flatten(files, d)
        upload_dataset(owner(), slug + "-input", slug + " input", d)
        code = RUNNER.replace("MODELS", repr(a.models)).replace("DISTS", repr(a.dists)) \
            .replace("PROCRES", str(a.process_res))
        kid = push_kernel(slug, code, datasets=[slug + "-input"])
        wait(kid)
        pull(kid, os.path.join(ROOT, "data", a.name, "da3"))
    else:
        pull(f"{owner()}/{slug}", os.path.join(ROOT, "data", a.name, "da3"))


if __name__ == "__main__":
    main()
