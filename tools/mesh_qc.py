#!/usr/bin/env python3
"""Count what is wrong with a mesh, so that methods are compared by numbers.

The owl's defects were judged by eye: spikes, torn flaps, cracks, cut-off
ears. Each has a count here.

  components     pieces; crumbs = pieces under 1% of the faces
  boundary       edges with one face (tears, holes)
  non-manifold   edges with more than two faces
  self-x         faces that cross another face (folds through itself)
  folds          edges whose two faces meet at a dihedral angle over 120 deg
                 (a surface turned back on itself: flaps, fins)
  spikes         vertices standing more than 1.5 mean edge lengths off the
                 plane of their neighbours
  silhouette     IoU of the mesh's outline against each view's mask

  python3 tools/mesh_qc.py data/cat/fit/base/mesh.ply --views data/cat/views [--json out.json]
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load(path):
    import trimesh
    m = trimesh.load(path, process=False, force="mesh")
    return np.asarray(m.vertices, np.float64), np.asarray(m.faces, np.int64)


def edge_faces(f):
    e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    e = np.sort(e, 1)
    fid = np.tile(np.arange(len(f)), 3)
    key = e[:, 0] * (f.max() + 1) + e[:, 1]
    order = np.argsort(key, kind="stable")
    key, fid, e = key[order], fid[order], e[order]
    uk, start, count = np.unique(key, return_index=True, return_counts=True)
    return e[start], start, count, fid


def face_normals(v, f):
    n = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    return n / np.linalg.norm(n, axis=1, keepdims=True).clip(1e-20)


def silhouette_iou(v, f, views, res=384):
    meta = json.load(open(os.path.join(views, "cameras.json")))
    o = meta["ortho_scale"]
    out = {}
    for name, info in meta["views"].items():
        M = np.array(info["matrix_world"])
        rel = v - M[:3, 3]
        col = (rel @ M[:3, 0] / o + 0.5) * res
        row = (0.5 - rel @ M[:3, 1] / o) * res
        img = Image.new("L", (res, res), 0)
        d = ImageDraw.Draw(img)
        for tri in f:
            d.polygon([(col[i], row[i]) for i in tri], fill=255)
        a = np.asarray(img) > 127
        m = np.asarray(Image.open(os.path.join(views, "mask", f"{name}.png")).convert("L")
                       .resize((res, res), Image.BILINEAR)) > 127
        out[name] = float((a & m).sum() / max((a | m).sum(), 1))
    return out


def qc(path, views=None):
    import pymeshlab
    import trimesh
    v, f = load(path)
    r = {"vertices": len(v), "faces": len(f)}
    tm = trimesh.Trimesh(v, f, process=False)
    comp = tm.split(only_watertight=False)
    sizes = sorted((len(c.faces) for c in comp), reverse=True)
    r["components"] = len(sizes)
    r["crumbs"] = int(sum(s < 0.01 * sizes[0] for s in sizes))
    e, start, count, fid = edge_faces(f)
    r["boundary_edges"] = int((count == 1).sum())
    r["nonmanifold_edges"] = int((count > 2).sum())
    r["watertight"] = bool(r["boundary_edges"] == 0 and r["nonmanifold_edges"] == 0)
    if r["watertight"]:
        r["genus"] = int(round((2 * len(sizes) - (len(v) - len(e) + len(f))) / 2))
    # dihedral folds, on manifold edges
    fn = face_normals(v, f)
    two = count == 2
    fa, fb = fid[start[two]], fid[start[two] + 1]
    cosd = (fn[fa] * fn[fb]).sum(1)
    r["folds_120deg"] = int((cosd < np.cos(np.radians(120))).sum())
    r["folds_frac"] = float(r["folds_120deg"] / max(two.sum(), 1))
    # spikes: height of a vertex above the plane of its one-ring
    nb_sum = np.zeros_like(v)
    deg = np.zeros(len(v))
    np.add.at(nb_sum, e[:, 0], v[e[:, 1]]); np.add.at(nb_sum, e[:, 1], v[e[:, 0]])
    np.add.at(deg, e[:, 0], 1); np.add.at(deg, e[:, 1], 1)
    vn = np.zeros_like(v)
    area = np.linalg.norm(np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]]), axis=1, keepdims=True)
    for c in range(3):
        np.add.at(vn, f[:, c], fn * area)
    vn /= np.linalg.norm(vn, axis=1, keepdims=True).clip(1e-20)
    el = np.linalg.norm(v[e[:, 0]] - v[e[:, 1]], axis=1).mean()
    h = np.abs(((v - nb_sum / deg.clip(1)[:, None]) * vn).sum(1)) / el
    r["spikes_1.5edge"] = int((h > 1.5).sum())
    r["mean_edge"] = float(el)
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(v, f))
    ms.compute_selection_by_self_intersections_per_face()
    r["selfx_faces"] = int(ms.current_mesh().selected_face_number())
    if views:
        s = silhouette_iou(v, f, views)
        r["silhouette_iou_min"] = min(s.values())
        r["silhouette_iou"] = s
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("meshes", nargs="+")
    ap.add_argument("--views", default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    allr = {}
    for p in a.meshes:
        r = qc(p, a.views)
        allr[p] = r
        print(f"{p}\n  faces {r['faces']:,}  pieces {r['components']} (crumbs {r['crumbs']})  "
              f"boundary {r['boundary_edges']}  non-manifold {r['nonmanifold_edges']}  "
              f"self-x {r['selfx_faces']}  folds {r['folds_120deg']}  spikes {r['spikes_1.5edge']}"
              + (f"  genus {r['genus']}" if "genus" in r else "")
              + (f"  silhouette IoU min {r['silhouette_iou_min']:.4f}" if "silhouette_iou_min" in r else ""))
    if a.json:
        json.dump(allr, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
