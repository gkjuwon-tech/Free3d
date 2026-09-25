#!/usr/bin/env python3
"""Shared plumbing for Free3d's Kaggle kernels: upload a folder as a private
dataset, push a private T4 script kernel that sees it, poll, pull.

Kaggle flattens subdirectories of an uploaded dataset into archives, so files
are uploaded flat as dir__sub__file and rebuilt on the other side by
`unflatten()` (the same convention as engine/tools/kaggle_stage2.py).
The Kaggle token comes from the environment (KAGGLE_API_TOKEN); nothing here
stores it.
"""
import json
import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAGE = os.path.join(ROOT, "data", "_kaggle")
sys.path.insert(0, os.path.join(ROOT, "engine", "tools"))
from kaggle_stage2 import owner, sh, upload_dataset  # noqa: E402,F401

UNFLATTEN = r'''
import os, shutil
def unflatten(marker, dst):
    """rebuild dir__sub__file uploads under dst; returns dst"""
    src = None
    for r, _, fs in os.walk("/kaggle/input"):
        if marker in fs:
            src = r
            break
    if src is None:
        raise SystemExit("input not found: " + marker)
    for f in os.listdir(src):
        p = os.path.join(dst, *f.split("__"))
        os.makedirs(os.path.dirname(p), exist_ok=True)
        shutil.copy(os.path.join(src, f), p)
    return dst
'''


def flatten(pairs, folder):
    """pairs: [(local file, remote relative path)] -> flat copies in folder"""
    shutil.rmtree(folder, ignore_errors=True)
    os.makedirs(folder)
    for src, rel in pairs:
        shutil.copy(src, os.path.join(folder, rel.replace("/", "__")))


def tree(local_dir, prefix):
    """every file under local_dir as (path, prefix/relative path)"""
    out = []
    for r, _, fs in os.walk(local_dir):
        for f in fs:
            p = os.path.join(r, f)
            out.append((p, prefix + "/" + os.path.relpath(p, local_dir)))
    return out


def push_kernel(slug, code, datasets=(), kernels=(), gpu=True):
    user = owner()
    kdir = os.path.join(STAGE, "kernel_" + slug)
    shutil.rmtree(kdir, ignore_errors=True)
    os.makedirs(kdir)
    open(os.path.join(kdir, "run.py"), "w").write(code)
    meta = {"id": f"{user}/{slug}", "title": slug, "code_file": "run.py", "language": "python",
            "kernel_type": "script", "is_private": True, "enable_gpu": gpu, "enable_tpu": False,
            "enable_internet": True, "keywords": [], "dataset_sources": [f"{user}/{d}" for d in datasets],
            "kernel_sources": [f"{user}/{k}" for k in kernels], "competition_sources": [],
            "model_sources": []}
    if gpu:
        meta["machine_shape"] = "NvidiaTeslaT4"
    json.dump(meta, open(os.path.join(kdir, "kernel-metadata.json"), "w"), indent=1)
    cmd = ["kaggle", "kernels", "push", "-p", kdir] + (["--accelerator", "NvidiaTeslaT4"] if gpu else [])
    sh(cmd)
    return f"{user}/{slug}"


def wait(kid, poll=30):
    while True:
        r = subprocess.run(["kaggle", "kernels", "status", kid], capture_output=True, text=True)
        s = r.stdout.strip()
        if any(k in s.lower() for k in ("complete", "error", "cancel")):
            print(s, flush=True)
            return "complete" in s.lower()
        time.sleep(poll)


def pull(kid, out):
    os.makedirs(out, exist_ok=True)
    sh(["kaggle", "kernels", "output", kid, "-p", out, "-o"], check=False)
    log = [f for f in os.listdir(out) if f.endswith(".log")]
    if log:
        try:
            recs = json.load(open(os.path.join(out, log[0])))
            open(os.path.join(out, "run.txt"), "w").write("".join(x["data"] for x in recs))
        except ValueError:
            pass
