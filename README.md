# Free3d

One generated character in, a clean mesh out, on a free Kaggle T4.

- Views: MV-Adapter (SDXL, Juggernaut-XL-v9 base), six orthographic level views.
- Reconstruction engine: [`engine/`](https://github.com/gkjuwon-tech/3d) (git submodule).
- No monocular normal estimator: geometry comes from silhouettes, cross-view
  colour consistency and a pose-conditioned multi-view depth initialisation.

Data: `data/cat/views/{rgb,mask,mask_raw,cameras.json}` (engine format; `mask_raw`
is the generator's outline, `mask` has whiskers removed by `tools/clean_masks.py`).
