#!/usr/bin/env python3
"""Run tools/flexi_fit.py on a Kaggle T4, several variants back to back.

nvdiffrast comes from the wheel the mvadapter-env kernel built for this very
image (kernel source flyjw12/mvadapter-env), so nothing compiles and nothing
touches PyTorch.

  push cat --views data/cat/views --depth data/cat/da3/DA3-LARGE-1.1_d10 \
      --variant base: --variant nodepth:"--w-depth 0" --variant nophoto:"--w-photo 0"
  -> data/cat/fit/<tag>/{mesh.ply, final.png, snap_*.png, log.json, stdout.txt}
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kaggle_job import ROOT, STAGE, UNFLATTEN, flatten, owner, pull, push_kernel, tree, upload_dataset, wait  # noqa: E402

RUNNER = UNFLATTEN + r'''
import glob, subprocess, sys, time
t0 = time.time()
subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv", shell=True)
unflatten("code__flexi_fit.py", "/tmp/job")
whl = glob.glob("/kaggle/input/**/nvdiffrast-*.whl", recursive=True)
r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps"] + whl)
print("nvdiffrast wheel", whl, "exit", r.returncode, f"[{time.time()-t0:.0f}s]", flush=True)
VARIANTS = __VARIANTS__
LIMIT = __LIMIT__
for tag, extra in VARIANTS:
    out = "/kaggle/working/" + tag
    os.makedirs(out, exist_ok=True)
    cmd = [sys.executable, "/tmp/job/code/flexi_fit.py", "--views", "/tmp/job/views", "--out", out] + \
          (["--depth", "/tmp/job/depth"] if os.path.isdir("/tmp/job/depth") else []) + extra
    print("$", " ".join(cmd), flush=True)
    t1 = time.time()
    with open(out + "/stdout.txt", "w") as fh:
        try:
            r = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, timeout=LIMIT, cwd="/tmp/job/code")
            code = r.returncode
        except subprocess.TimeoutExpired:
            code = "timeout"
    print(f"== {tag}: exit {code} in {time.time()-t1:.0f}s", flush=True)
    print(open(out + "/stdout.txt").read()[-2500:], flush=True)
'''


def main():
    import shlex
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["push", "pull"])
    ap.add_argument("name")
    ap.add_argument("--views")
    ap.add_argument("--depth", default=None)
    ap.add_argument("--variant", action="append", default=[], help='tag:"flexi_fit args"')
    ap.add_argument("--limit", type=int, default=2400, help="seconds per variant")
    ap.add_argument("--run", default="fit", help="output folder under data/<name>/")
    a = ap.parse_args()
    slug = f"free3d-fit-{a.name}"
    out = os.path.join(ROOT, "data", a.name, a.run)
    if a.cmd == "pull":
        return pull(f"{owner()}/{slug}", out)
    here = os.path.dirname(os.path.abspath(__file__))
    files = [(p, r) for p, r in tree(a.views, "views") if "/mask_raw/" not in r]
    if a.depth:
        files += tree(a.depth, "depth")
    files += [(os.path.join(here, "flexi_fit.py"), "code/flexi_fit.py")]
    files += tree(os.path.join(here, "flexicubes"), "code/flexicubes")
    files = [(p, r) for p, r in files if "__pycache__" not in r]
    d = os.path.join(STAGE, slug + "-input")
    flatten(files, d)
    upload_dataset(owner(), slug + "-input", slug + " input", d)
    variants = []
    for v in a.variant or ["base:"]:
        tag, _, args = v.partition(":")
        variants.append((tag, shlex.split(args)))
    code = RUNNER.replace("__VARIANTS__", repr(variants)).replace("__LIMIT__", str(a.limit))
    kid = push_kernel(slug, code, datasets=[slug + "-input"], kernels=["mvadapter-env"])
    wait(kid)
    pull(kid, out)


if __name__ == "__main__":
    main()
