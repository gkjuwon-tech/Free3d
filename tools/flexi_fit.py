#!/usr/bin/env python3
"""The shape as a signed distance field, fitted to the views (FlexiCubes).

engine/tools/mesh_fit.py moves the vertices of one explicit mesh. On the owl
that mesh tore (flaps on the crown, cracks across the chest), grew spikes
where one view's target disagreed with the rest, and could never open a gap
the visual hull had closed (feet fused to the book): an explicit surface can
fold through itself and cannot change its genus. Here the unknown is a grid
of signed distances and the surface is its zero level set, extracted every
step by FlexiCubes (Shen et al. 2023) and rendered with nvdiffrast. A level
set cannot self-intersect, is closed and manifold by construction, and opens
or closes holes freely. The grid is refined coarse to fine (each stage's
field upsampled into the next), so large regions settle before small ones.

No normal estimator is used: single-image normals were 25-30 degrees off on
these views. The signals are

  mask    rendered coverage against each view's silhouette
  depth   pose-conditioned multi-view depth (tools/kaggle_da3.py), robust and
          confidence weighted -- where the surface is, not how it is tilted
  photo   cross-view photo-consistency: each surface point seen in view i is
          looked up in view j (visibility by view j's own depth buffer) and
          the two images are compared by normalised cross-correlation over a
          patch, which ignores the shading and brightness a painter varies
          between views. The views were drawn jointly (MV-Adapter), so where
          there is texture this is the most trustworthy geometric signal there
          is; on flat white cloth it has nothing to say and says nothing
          (patches below a contrast floor are skipped).

and the regularisers: FlexiCubes' own (L_dev, weights), the sign-change
penalty on grid edges that removes floaters, and normal consistency between
adjacent faces of the extracted mesh (the spike suppressor).

Initial field: the visual hull as an exact 2D-distance field (orthographic),
grown by --hull-margin pixels, intersected with a robust truncated fusion of
the depth maps (median of at least two views). --kofn lets k views dissent,
but only where every direction is seen twice: with six level views the 45
and 315 degree views are the only ones carving their diagonals, and k = 1
left the cat's hull 0.67 IoU fat there (k = 0: 0.985-0.990 in every view).

  python3 tools/flexi_fit.py --views data/cat/views --depth data/cat/da3/X --out data/cat/fit
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from PIL import Image
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------- views

def load_views(views, res):
    meta = json.load(open(os.path.join(views, "cameras.json")))
    names = list(meta["views"])
    M, masks, rgbs = [], [], []
    for v in names:
        M.append(np.array(meta["views"][v]["matrix_world"], np.float64))
        m = Image.open(os.path.join(views, "mask", f"{v}.png")).convert("L").resize((res, res), Image.BILINEAR)
        masks.append(np.asarray(m, np.float32) / 255)
        c = Image.open(os.path.join(views, "rgb", f"{v}.png")).convert("RGB").resize((res, res), Image.BICUBIC)
        rgbs.append(np.asarray(c, np.float32) / 255)
    return meta, names, np.stack(M), np.stack(masks), np.stack(rgbs)


def clip_matrices(M, o, near=0.1, far=4.0):
    """world -> clip for orthographic cameras (OpenGL), as engine/tools/mesh_fit.py"""
    h = o / 2
    P = np.array([[1 / h, 0, 0, 0], [0, 1 / h, 0, 0],
                  [0, 0, -2 / (far - near), -(far + near) / (far - near)], [0, 0, 0, 1]])
    return np.stack([P @ np.linalg.inv(m) for m in M]).astype(np.float32)


def camera(az, el, dist=2.0):
    """orthographic camera-to-world at azimuth az, elevation el (z up), as
    engine/tools/mvgen_views.camera"""
    d = np.array([np.cos(np.radians(el)) * np.cos(np.radians(az)),
                  np.cos(np.radians(el)) * np.sin(np.radians(az)), np.sin(np.radians(el))])
    right = np.cross([0, 0, 1], d)
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1.0, 0, 0])
    right /= np.linalg.norm(right)
    up = np.cross(d, right)
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = right, up, d, dist * d
    return m


def write_ply(path, v, f):
    with open(path, "wb") as fh:
        fh.write((f"ply\nformat binary_little_endian 1.0\nelement vertex {len(v)}\n"
                  "property float x\nproperty float y\nproperty float z\n"
                  f"element face {len(f)}\nproperty list uchar int vertex_indices\nend_header\n").encode())
        fh.write(np.asarray(v, "<f4").tobytes())
        rec = np.empty(len(f), dtype=[("n", "u1"), ("i", "<i4", (3,))])
        rec["n"] = 3
        rec["i"] = f
        fh.write(rec.tobytes())


# ---------------------------------------------------------------- grid

def voxel_grid(R, device):
    """FlexiCubes' construct_voxel_grid without its torch.unique over 8 R^3
    corners (which does not fit a T4 at R = 192): vertex (i, j, k) of the
    (R+1)^3 lattice has index (i (R+1) + j) (R+1) + k, in [-0.5, 0.5]^3, and
    cube corners follow FlexiCubes' corner table"""
    import torch
    n = R + 1
    c = torch.arange(n, device=device, dtype=torch.float32) / R - 0.5
    X = torch.stack(torch.meshgrid(c, c, c, indexing="ij"), -1).reshape(-1, 3)
    i = torch.arange(R, device=device)
    I, J, K = torch.meshgrid(i, i, i, indexing="ij")
    base = ((I * n + J) * n + K).reshape(-1)
    corners = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0), (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1)]
    cubes = torch.stack([base + (dx * n + dy) * n + dz for dx, dy, dz in corners], 1)
    return X, cubes


def upsample(s, R0, R1):
    import torch
    g = s.reshape(1, 1, R0 + 1, R0 + 1, R0 + 1)
    return torch.nn.functional.interpolate(g, size=(R1 + 1,) * 3, mode="trilinear",
                                           align_corners=True).reshape(-1)


# ---------------------------------------------------------------- initial field

def silhouette_sdf(mask):
    """signed distance to the outline, px, positive outside"""
    m = mask > 0.5
    return (ndimage.distance_transform_edt(~m) - ndimage.distance_transform_edt(m)).astype(np.float32)


def project(Mt, o, res, X):
    """(row, col) continuous pixel coords with centres at integers, and the
    distance w in front of the camera, for points X (N,3) in camera Mt (4,4)"""
    r, u, b, loc = Mt[:3, 0], Mt[:3, 1], Mt[:3, 2], Mt[:3, 3]
    rel = X - loc
    col = (rel @ r / o + 0.5) * res - 0.5
    row = (0.5 - rel @ u / o) * res - 0.5
    return row, col, -(rel @ b)


def sample(img, row, col, mode="bilinear"):
    """img (C,H,W) or (H,W) torch; row/col (N,) -> (N,) or (N,C), border clamped"""
    import torch
    two = img.dim() == 2
    im = img[None, None] if two else img[None]
    H, W = im.shape[-2:]
    g = torch.stack([(col + 0.5) / W * 2 - 1, (row + 0.5) / H * 2 - 1], -1)[None, None]
    out = torch.nn.functional.grid_sample(im, g, mode=mode, padding_mode="border", align_corners=False)
    out = out[0, :, 0]
    return out[0] if two else out.T


def initial_field(X, Mt, o, masks, depth, kofn, tau, margin_px=0.0):
    """hull (k views may dissent) intersected with robust depth fusion; world units"""
    import torch
    C, res = masks.shape[0], masks.shape[-1]
    px = o / res
    sd = []
    for k in range(C):
        row, col, _ = project(Mt[k], o, res, X)
        s = (sample(torch.tensor(silhouette_sdf(masks[k]), device=X.device), row, col) - margin_px) * px
        sd.append(s)
    sd = torch.stack(sd)                                          # C, N
    H = torch.sort(sd, 0, descending=True).values[min(kofn, C - 1)]
    if depth is None:
        return H
    Ds, valid = [], []
    for k in range(C):
        w_obs, conf = depth[k]
        dres = w_obs.shape[-1]
        row, col, w = project(Mt[k], o, dres, X)
        wo = sample(w_obs, row, col, "nearest")
        mk = sample(torch.tensor(masks[k], device=X.device), row * res / dres, col * res / dres, "nearest")
        s = wo - w                                                 # >0: in front of the surface (empty)
        ok = (mk > 0.5) & (s > -tau)
        Ds.append(s.clamp(-tau, tau))
        valid.append(ok)
    Ds, valid = torch.stack(Ds), torch.stack(valid)
    n_ok = valid.sum(0)
    med = torch.where(valid, Ds, torch.full_like(Ds, float("nan"))).nanmedian(0).values
    return torch.where(n_ok >= 2, torch.maximum(H, med), H)


# ---------------------------------------------------------------- losses

def box(x, k):
    import torch
    return torch.nn.functional.avg_pool2d(x[None, None] if x.dim() == 2 else x[:, None], k, 1, k // 2,
                                          count_include_pad=False).squeeze(1).squeeze(0)


def ncc(a, b, m, k, floor):
    """normalised cross-correlation of a and b (H,W) over k x k patches inside
    m; returns (per-pixel NCC, usable-pixel mask). Patches whose reference
    contrast (std of a) is under `floor` are not usable: flat cloth says
    nothing about depth, and its NCC is noise"""
    import torch
    S = box(m, k).clamp(min=1e-6)
    ma, mb = box(a * m, k) / S, box(b * m, k) / S
    va = (box(a * a * m, k) / S - ma * ma).clamp(min=0)
    vb = (box(b * b * m, k) / S - mb * mb).clamp(min=0)
    cov = box(a * b * m, k) / S - ma * mb
    r = cov / (va * vb + 1e-8).sqrt()
    use = (S > 0.8) & (va.sqrt() > floor) & (m > 0.5)
    return r, use


def blur3d(g, n, sigma):
    """separable gaussian over the (n,n,n) lattice, g flat"""
    import torch
    if sigma <= 0:
        return g
    r = max(1, int(3 * sigma))
    k = torch.exp(-0.5 * (torch.arange(-r, r + 1, device=g.device, dtype=g.dtype) / sigma) ** 2)
    k = k / k.sum()
    x = g.view(1, 1, n, n, n)
    for dim in range(3):
        shape = [1, 1, 1, 1, 1]
        shape[2 + dim] = -1
        pad = [0, 0, 0, 0, 0, 0]
        pad[2 * (2 - dim)] = pad[2 * (2 - dim) + 1] = r
        x = torch.nn.functional.conv3d(torch.nn.functional.pad(x, pad, mode="replicate"), k.view(shape))
    return x.reshape(-1)


def laplacian_band(sdf, n, band):
    """squared 6-neighbour Laplacian of the field near its zero set: bends
    in the surface cost, flat and gently curved regions are free"""
    s3 = sdf.view(n, n, n)
    c = s3[1:-1, 1:-1, 1:-1]
    lap = (s3[2:, 1:-1, 1:-1] + s3[:-2, 1:-1, 1:-1] + s3[1:-1, 2:, 1:-1] + s3[1:-1, :-2, 1:-1]
           + s3[1:-1, 1:-1, 2:] + s3[1:-1, 1:-1, :-2] - 6 * c)
    m = c.detach().abs() < band
    return lap[m].pow(2).mean() if m.any() else sdf.sum() * 0


class SmoothStep:
    """The field's optimiser. Adam scales every grid value's step by that
    value's own gradient history, so a vertex whose only gradient is the
    faint noise of a regulariser still moves a full learning rate each step:
    900 such steps grew the silhouette-only cat a skin of pimples wherever
    no view constrained the surface. Here the gradient is blurred over the
    lattice (neighbouring values move together: Large Steps for a grid),
    limited to a band around the surface, and then normalised by one RMS
    for the whole band, so strong signals move the surface and faint ones
    barely do. Momentum as in Adam."""

    def __init__(self, p, n, lr, sigma, band, b1=0.9, b2=0.99):
        import torch
        self.p, self.n, self.lr, self.sigma, self.band = p, n, lr, sigma, band
        self.b1, self.b2, self.t = b1, b2, 0
        self.m = torch.zeros_like(p)
        self.v = 0.0

    def step(self):
        import torch
        with torch.no_grad():
            g = self.p.grad
            if g is None:
                return
            g = blur3d(g, self.n, self.sigma)
            near = blur3d((self.p.abs() < self.band).float(), self.n, 1.0) > 0.01
            g = g * near
            self.t += 1
            self.m.mul_(self.b1).add_(g, alpha=1 - self.b1)
            ms = g[near].pow(2).mean().item() if near.any() else 0.0
            self.v = self.b2 * self.v + (1 - self.b2) * ms
            mh = self.m / (1 - self.b1 ** self.t)
            vh = self.v / (1 - self.b2 ** self.t)
            self.p.sub_(self.lr * mh / (vh ** 0.5 + 1e-12))


def sdf_reg(sdf, edges):
    """FlexiCubes/nvdiffrec: sign changes along grid edges cost a BCE, which
    removes floaters and interior sheets no view can see"""
    import torch
    s = sdf[edges.reshape(-1)].reshape(-1, 2)
    s = s[torch.sign(s[:, 0]) != torch.sign(s[:, 1])]
    if not len(s):
        return sdf.sum() * 0
    bce = torch.nn.functional.binary_cross_entropy_with_logits
    return bce(s[:, 0], (s[:, 1] > 0).float()) + bce(s[:, 1], (s[:, 0] > 0).float())


def face_normals(v, f):
    import torch
    return torch.nn.functional.normalize(torch.linalg.cross(v[f[:, 1]] - v[f[:, 0]],
                                                            v[f[:, 2]] - v[f[:, 0]], dim=-1), dim=-1)


def normal_consistency(v, f):
    """1 - cos between the normals of faces sharing an edge"""
    import torch
    e = torch.cat([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], 0)
    e = torch.sort(e, 1).values
    fid = torch.arange(len(f), device=f.device).repeat(3)
    key = e[:, 0] * (int(v.shape[0]) + 1) + e[:, 1]
    order = torch.argsort(key)
    key, fid = key[order], fid[order]
    same = key[1:] == key[:-1]
    a, b = fid[:-1][same], fid[1:][same]
    fn = face_normals(v, f)
    return (1 - (fn[a] * fn[b]).sum(-1)).mean()


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", required=True)
    ap.add_argument("--depth", default=None, help="dir of <view>.npz (w, conf) from kaggle_da3.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--res", type=int, default=768, help="render and target resolution")
    ap.add_argument("--stages", default="96:300,144:300,192:300", help="grid:steps, coarse to fine")
    ap.add_argument("--kofn", type=int, default=0, help="views allowed to dissent in the initial hull")
    ap.add_argument("--hull-margin", type=float, default=1.0, help="px the initial hull is grown by")
    ap.add_argument("--lr", type=float, default=0.03, help="sdf step, voxels (RMS over the band)")
    ap.add_argument("--grad-sigma", type=float, default=2.0, help="gradient blur on the lattice, voxels")
    ap.add_argument("--band", type=float, default=3.0, help="only grid values this near the surface move")
    ap.add_argument("--w-lap", type=float, default=0.05, help="Laplacian of the field near the surface")
    ap.add_argument("--lr-local", type=float, default=0.005, help="FlexiCubes weights and deformation")
    ap.add_argument("--w-mask", type=float, default=1.0)
    ap.add_argument("--w-depth", type=float, default=1.0)
    ap.add_argument("--depth-huber", type=float, default=2.0, help="voxels")
    ap.add_argument("--depth-decay", type=float, default=0.3,
                    help="depth weight at the end, as a fraction of its start (guides, then yields)")
    ap.add_argument("--w-photo", type=float, default=0.1)
    ap.add_argument("--photo-start", type=int, default=1, help="first stage with the photo term")
    ap.add_argument("--photo-patch", type=int, default=9)
    ap.add_argument("--photo-floor", type=float, default=0.02, help="min patch std (0-1 grey)")
    ap.add_argument("--photo-blur", default="3,2,1", help="grey image blur per stage, px")
    ap.add_argument("--photo-pairs", type=int, default=6, help="view pairs per step")
    ap.add_argument("--max-pair-deg", type=float, default=95.0)
    ap.add_argument("--w-dev", type=float, default=0.5)
    ap.add_argument("--w-weight", type=float, default=0.1)
    ap.add_argument("--w-sdfreg", type=float, default=0.2)
    ap.add_argument("--w-nc", type=float, default=0.05, help="normal consistency")
    ap.add_argument("--snap-every", type=int, default=100)
    a = ap.parse_args()

    import torch
    import nvdiffrast.torch as dr
    from flexicubes.flexicubes import FlexiCubes

    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    dev = "cuda"
    meta, names, M, masks, rgbs = load_views(a.views, a.res)
    o = float(meta["ortho_scale"])
    C = len(names)
    Mt = torch.tensor(M, device=dev, dtype=torch.float32)
    mvp = torch.tensor(clip_matrices(M, o), device=dev)
    tgt_a = torch.tensor(masks, device=dev)
    grey = 0.299 * rgbs[..., 0] + 0.587 * rgbs[..., 1] + 0.114 * rgbs[..., 2]

    depth = None
    if a.depth:
        depth = []
        for v in names:
            z = np.load(os.path.join(a.depth, f"{v}.npz"))
            w, cf = z["w"].astype(np.float32), z["conf"].astype(np.float32)
            cf = cf / max(np.percentile(cf, 95), 1e-6)
            depth.append((torch.tensor(w, device=dev), torch.tensor(np.clip(cf, 0, 1), device=dev)))
        print(f"depth: {len(depth)} maps at {depth[0][0].shape[-1]} px", flush=True)

    # object box from the silhouettes: every axis is bounded by some view
    with torch.no_grad():
        n0 = 64
        Xc, _ = voxel_grid(n0, dev)
        Xc = Xc * o
        H0 = initial_field(Xc, Mt, o, masks, None, 0, 0)
        inside = Xc[H0 < 0]
        lo, hi = inside.min(0).values, inside.max(0).values
    centre = (lo + hi) / 2
    size = float((hi - lo).max()) * 1.12
    print(f"{C} views at {a.res}px; box {size:.3f} around {centre.cpu().numpy().round(3)} [{time.time()-t0:.0f}s]",
          flush=True)

    # view pairs for photo-consistency: every ordered pair within max-pair-deg
    backs = M[:, :3, 2]
    pairs = [(i, j) for i in range(C) for j in range(C) if i != j and
             np.degrees(np.arccos(np.clip(backs[i] @ backs[j], -1, 1))) <= a.max_pair_deg]
    print(f"photo pairs: {len(pairs)}", flush=True)

    glctx = dr.RasterizeCudaContext(device=dev)
    fc = FlexiCubes(dev)
    locs = Mt[:, :3, 3]
    bks = Mt[:, :3, 2]

    def render(v, f, k_list):
        """per view (top row first): alpha (H,W), world position (H,W,3), w (H,W), tri id"""
        vh = torch.cat([v, torch.ones_like(v[:, :1])], -1)
        clip = vh[None] @ mvp[k_list].transpose(-2, -1)
        fi = f.int()
        rast, _ = dr.rasterize(glctx, clip, fi, resolution=[a.res, a.res])
        pos, _ = dr.interpolate(v[None].expand(len(k_list), -1, -1).contiguous(), rast, fi)
        alpha = (rast[..., 3:] > 0).float()
        alpha = dr.antialias(alpha, rast, clip, fi)[..., 0]
        w = -((pos - locs[k_list][:, None, None]) * bks[k_list][:, None, None]).sum(-1)
        # nvdiffrast's first row is the bottom one
        return alpha.flip(1), pos.flip(1), w.flip(1), (rast[..., 3].long() - 1).flip(1)

    stages = [tuple(int(x) for x in s.split(":")) for s in a.stages.split(",")]
    blurs = [float(x) for x in a.photo_blur.split(",")]
    sdf = None
    log = []
    step = 0
    total = sum(s for _, s in stages)
    for si, (R, steps) in enumerate(stages):
        X, cubes = voxel_grid(R, dev)
        vox = size / R
        Xw = X * size + centre
        with torch.no_grad():
            if sdf is None:
                F = initial_field(Xw, Mt, o, masks, depth, a.kofn, 3 * vox, a.hull_margin)
                init = (F / vox).clamp(-4, 4)
            else:
                init = upsample(sdf.detach() * (R / stages[si - 1][0]), stages[si - 1][0], R).clamp(-4, 4)
        sdf = torch.nn.Parameter(init.clone())
        weight = torch.nn.Parameter(torch.zeros(len(cubes), 21, device=dev))
        deform = torch.nn.Parameter(torch.zeros_like(X))
        edges = torch.unique(torch.sort(cubes[:, fc.cube_edges].reshape(-1, 2), 1).values, dim=0)
        opt = torch.optim.Adam([weight, deform], lr=a.lr_local)
        sopt = SmoothStep(sdf, R + 1, a.lr, a.grad_sigma, a.band)
        sig = blurs[min(si, len(blurs) - 1)]
        g_img = torch.tensor(np.stack([ndimage.gaussian_filter(g, sig) if sig > 0 else g for g in grey]),
                             device=dev)
        print(f"stage {si}: grid {R}^3 ({len(X):,} vertices), voxel {vox:.4f}, grey blur {sig}px "
              f"[{time.time()-t0:.0f}s]", flush=True)
        for it in range(steps):
            opt.zero_grad()
            sdf.grad = None
            t = step / max(total - 1, 1)
            gv = X + (0.5 - 1e-6) / R * torch.tanh(deform)
            v, f, L_dev = fc(gv * size + centre, sdf, cubes, R, beta_fx12=weight[:, :12],
                             alpha_fx8=weight[:, 12:20], gamma_f=weight[:, 20], training=True)
            ks = list(range(C))
            alpha, pos, w, tri = render(v, f, ks)
            l_mask = (alpha - tgt_a).pow(2).mean()
            loss = a.w_mask * l_mask
            l_depth = torch.zeros((), device=dev)
            if depth is not None and a.w_depth > 0:
                parts = []
                for k in ks:
                    wo, cf = depth[k]
                    dres = wo.shape[-1]
                    wr = torch.nn.functional.interpolate(w[k][None, None], size=(dres, dres), mode="area")[0, 0]
                    cov = torch.nn.functional.interpolate((tri[k] >= 0).float()[None, None], size=(dres, dres),
                                                          mode="area")[0, 0]
                    mk = torch.nn.functional.interpolate(tgt_a[k][None, None], size=(dres, dres), mode="area")[0, 0]
                    use = (cov > 0.99) & (mk > 0.99)
                    r = (wr - wo)[use] / vox
                    hub = torch.nn.functional.huber_loss(r, torch.zeros_like(r), delta=a.depth_huber,
                                                         reduction="none")
                    parts.append((hub * cf[use]).sum() / cf[use].sum().clamp(min=1e-6))
                l_depth = torch.stack(parts).mean()
                wd = a.w_depth * (1 - (1 - a.depth_decay) * t)
                loss = loss + wd * l_depth
            l_photo = torch.zeros((), device=dev)
            if a.w_photo > 0 and pairs and si >= a.photo_start:
                sel = [pairs[q] for q in torch.randperm(len(pairs))[:a.photo_pairs].tolist()]
                parts = []
                for i, j in sel:
                    cov_i = (tri[i] >= 0) & (tgt_a[i] > 0.5)
                    P = pos[i][cov_i]
                    row, col, wj = project(Mt[j], o, a.res, P)
                    with torch.no_grad():
                        wbuf = torch.where(tri[j] >= 0, w[j], torch.full_like(w[j], 1e3))
                        vis = (wj.detach() <= sample(wbuf, row.detach(), col.detach(), "nearest") + 2 * vox) & \
                              (sample(tgt_a[j], row.detach(), col.detach(), "nearest") > 0.5)
                    warped = torch.zeros(a.res, a.res, device=dev)
                    warped[cov_i] = sample(g_img[j], row, col)
                    m = torch.zeros(a.res, a.res, device=dev)
                    m[cov_i] = vis.float()
                    r, use = ncc(g_img[i], warped, m, a.photo_patch, a.photo_floor)
                    if use.any():
                        parts.append((1 - r[use]).mean())
                if parts:
                    l_photo = torch.stack(parts).mean()
                    loss = loss + a.w_photo * l_photo
            sdf_w = a.w_sdfreg * (1 - 0.95 * min(1.0, 4 * t))
            loss = loss + sdf_w * sdf_reg(sdf, edges) + a.w_dev * L_dev.mean() + \
                a.w_weight * weight[:, :20].abs().mean()
            l_nc = normal_consistency(v, f) if a.w_nc > 0 else torch.zeros((), device=dev)
            loss = loss + a.w_nc * l_nc
            l_lap = laplacian_band(sdf, R + 1, 2.0) if a.w_lap > 0 else torch.zeros((), device=dev)
            loss = loss + a.w_lap * l_lap
            loss.backward()
            opt.step()
            sopt.step()
            with torch.no_grad():
                iou = ((alpha > 0.5) & (tgt_a > 0.5)).sum().item() / max(((alpha > 0.5) | (tgt_a > 0.5)).sum().item(), 1)
            rec = {"step": step, "stage": si, "grid": R, "mask": l_mask.item(), "depth_vox": l_depth.item(),
                   "photo": l_photo.item(), "nc": l_nc.item(), "iou": iou, "faces": len(f)}
            log.append(rec)
            if it % 25 == 0 or it == steps - 1:
                print(f"step {step:4d} grid {R} IoU {iou:.4f} depth {l_depth.item():.3f} photo "
                      f"{l_photo.item():.3f} nc {l_nc.item():.4f} faces {len(f):,} [{time.time()-t0:.0f}s]",
                      flush=True)
            if a.snap_every and (step % a.snap_every == 0 or step == total - 1):
                with torch.no_grad():
                    snap(v, f, os.path.join(a.out, f"snap_{step:04d}.png"))
            step += 1
    with torch.no_grad():
        gv = X + (0.5 - 1e-6) / R * torch.tanh(deform)
        v, f, _ = fc(gv * size + centre, sdf, cubes, R, beta_fx12=weight[:, :12], alpha_fx8=weight[:, 12:20],
                     gamma_f=weight[:, 20], training=False)
        write_ply(os.path.join(a.out, "mesh.ply"), v.cpu().numpy(), f.cpu().numpy())
        snap(v, f, os.path.join(a.out, "final.png"), size_px=512)
    json.dump(log, open(os.path.join(a.out, "log.json"), "w"))
    print(f"wrote {a.out}/mesh.ply {len(v):,} vertices {len(f):,} faces [{time.time()-t0:.0f}s]", flush=True)


def snap(v, f, path, size_px=384):
    """shaded look at the mesh from the six level views and three from above
    (the crown and shoulders no training view saw), grey clay, key light"""
    import torch
    import nvdiffrast.torch as dr
    cams = [camera(270 + az, 0) for az in (0, 45, 90, 180, 270, 315)] + \
           [camera(270 + az, el) for az, el in ((30, 35), (200, 35), (0, 75))]
    M = np.stack(cams)
    o = 1.1
    mvp = torch.tensor(clip_matrices(M, o), device=v.device)
    glctx = snap.ctx = getattr(snap, "ctx", None) or dr.RasterizeCudaContext(device=v.device)
    fi = f.int()
    vh = torch.cat([v, torch.ones_like(v[:, :1])], -1)
    clip = vh[None] @ mvp.transpose(-2, -1)
    rast, _ = dr.rasterize(glctx, clip, fi, resolution=[size_px, size_px])
    fn = face_normals(v, f)
    tid = rast[..., 3].long() - 1
    n = torch.where((tid >= 0)[..., None], fn[tid.clamp(min=0)], torch.zeros(1, device=v.device))
    Mt = torch.tensor(M, device=v.device, dtype=torch.float32)
    key = torch.nn.functional.normalize(Mt[:, :3, 2] + Mt[:, :3, 1] * 0.6 + Mt[:, :3, 0] * 0.4, dim=-1)
    lam = (n * key[:, None, None]).sum(-1).clamp(min=0)
    shade = 0.18 + 0.82 * lam
    img = torch.where(tid >= 0, shade, torch.full_like(shade, 0.12)).flip(1)
    tiles = (img.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(np.concatenate([np.concatenate(tiles[:6], 1),
                                    np.concatenate(list(tiles[6:]) + [np.full_like(tiles[0], 30)] * 3, 1)], 0)
                    ).save(path)


if __name__ == "__main__":
    main()
