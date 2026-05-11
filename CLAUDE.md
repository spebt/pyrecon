# pyrecon — SPECT Reconstruction Toolkit

PyTorch-based SPECT image reconstruction library. Runs on GPU (CUDA) or CPU.
All heavy jobs are submitted via SLURM on the `vscratch` cluster.

---

## Repository Layout

```
pyrecon/
├── custom_phantoms/          # Pre-built phantom .pt files + preview .png images
├── src/
│   ├── phantom_gen/          # Scripts to generate synthetic phantoms
│   ├── recon_mlem/           # Standard ML-EM reconstruction
│   └── recon_map_tv/         # MAP-TV reconstruction (ML-EM + TV proximal)
└── CLAUDE.md
```

---

## Modules

### `src/recon_mlem/`

Standard Maximum-Likelihood Expectation-Maximization (MLEM) reconstruction.

| File | Role |
|---|---|
| `mlem_torch_nonmpi.py` | Main reconstruction loop |
| `fake_projection.py` | Forward-project a phantom (noiseless) |
| `fake_projection_noisy.py` | Forward-project a phantom and add Poisson noise |
| `generate_flist.py` | Build the HDF5 file list CSV |
| `verify_recon.py` | Sanity-check a finished reconstruction |
| `view_npz.py` | Visualize reconstruction + CNR vs iteration plot; `--noisy` adds sinogram panel |
| `configs/base_config.yml` | All settings (paths, geometry, MLEM params, noise, CNR) |
| `run_mlem.sh` | SLURM job script; accepts `--noisy` flag |

Typical run order:
```bash
python generate_flist.py --config configs/base_config.yml

# Noiseless pipeline
python fake_projection.py    --config configs/base_config.yml
python mlem_torch_nonmpi.py  --config configs/base_config.yml
python view_npz.py           --config configs/base_config.yml

# Noisy pipeline (Poisson noise)
python fake_projection_noisy.py --config configs/base_config.yml
python mlem_torch_nonmpi.py     --config configs/base_config.yml --noisy
python view_npz.py              --config configs/base_config.yml --noisy

# Via SLURM (--noisy is optional)
sbatch run_mlem.sh
sbatch run_mlem.sh --noisy
```

---

### `src/recon_map_tv/`

MAP-TV reconstruction: ML-EM outer loop + Chambolle-Pock TV proximal inner loop.

**Algorithm (EM-TV operator splitting):**
1. **Outer (EM):** `x_em = x * A^T(y / (Ax + r)) / A^T(1)`
2. **Inner (TV prox):** `x_new = argmin_{x≥0} 0.5||x - x_em||² + β·TV(x)`  via Chambolle-Pock

**Stability constraint:** `τσ < 1/8` for 2D TV (enforced at startup).

| File | Role |
|---|---|
| `map_tv_recon.py` | Main reconstruction loop |
| `tv_ops.py` | 2D TV operators: `grad_2d`, `div_2d`, `tv_proximal_chambolle_pock` |
| `fake_projection.py` | Forward-project a phantom (noiseless) |
| `fake_projection_noisy.py` | Forward-project a phantom and add Poisson noise |
| `generate_flist.py` | Build the HDF5 file list CSV |
| `view_npz.py` | Visualize reconstruction + CNR + parameter overlay; `--noisy` adds sinogram panel |
| `configs/base_config.yml` | All settings (paths, geometry, MAP-TV params, noise, CNR) |
| `run_map_tv.sh` | SLURM job script (1 GPU, 64 GB RAM, 8 CPUs); accepts `--noisy` flag |
| `GAPS.md` | Tracked implementation gaps vs the engineer guide |

Typical run order:
```bash
python generate_flist.py --config configs/base_config.yml

# Noiseless pipeline
python fake_projection.py  --config configs/base_config.yml
python map_tv_recon.py     --config configs/base_config.yml
python view_npz.py         --config configs/base_config.yml

# Noisy pipeline (Poisson noise)
python fake_projection_noisy.py --config configs/base_config.yml
python map_tv_recon.py          --config configs/base_config.yml --noisy
python view_npz.py              --config configs/base_config.yml --noisy

# Via SLURM with optional overrides (--noisy can appear anywhere)
sbatch run_map_tv.sh
sbatch run_map_tv.sh --noisy
sbatch run_map_tv.sh --noisy configs/base_config.yml map_tv.beta=0.01
```

Key tuning knob is `beta` in the config. Start at `0.01–0.1` relative to max intensity of an unregularized image; too high → cartoon/blocky, too low → behaves like plain MLEM.

---

### `src/phantom_gen/`

Scripts to generate `.pt` phantom files used as inputs to `fake_projection.py`.

| File | Role |
|---|---|
| `create_custom_hotrod_phantom.py` | Hot-rod (Derenzo) phantom |
| `create_contrast_ring_phantom.py` | Contrast ring phantom |
| `create_custom_hotrod_ring_phantom.py` | Combined hot-rod + ring |
| `create_custom_point_phantom.py` | Point source phantom |
| `_disk_shape.py` | Shared disk/circle geometry primitive |
| `_geometry_2d_transform.py` | 2D affine transform helpers |

Output `.pt` files go to `custom_phantoms/`. Each file contains a dict with key `"Phantom tensor"` holding a 2D float32 tensor.

---

## Data & Paths Convention

- **HDF5 system matrices** live on `vscratch`:
  `/vscratch/grp-rutaoyao/sid/data/<scanner_name>/outputs/position_NNN_ppdfs.hdf5`
- Two formats are supported and auto-detected in `load_system_matrix()`:
  - **Sparse CSR**: keys `data`, `indices`, `indptr` + `shape` attribute
  - **Dense**: key `ppdfs`
- **Projections** are saved as `.npy` (float32, shape `[num_layouts, num_bins]`)
- **Reconstructions** are saved as `.npz` with key `estimates` (shape `[N_saved, H, W]`)
- **File lists** are plain-text CSVs, one absolute HDF5 path per line
- `cache_data: true` in config loads all system matrices into RAM — needs ~32 GB

---

## Config Structure

Every module is config-driven (`--config configs/base_config.yml`). Common keys:

```yaml
paths:
  hdf5_dir:          # directory with position_NNN_ppdfs.hdf5 files
  flist_path:        # output of generate_flist.py
  projs_path:        # output of fake_projection.py (or real scan .npy)
  projs_noisy_path:  # output of fake_projection_noisy.py
  phantom_path:      # input phantom .pt file
  recon_out_path:    # output reconstruction .npz

geometry:
  img_dim:           # FOV pixel dimension (square: img_dim x img_dim)
  pixel_size_mm:     # physical pixel size
  num_layouts:       # number of HDF5 chunks / detector positions

noise:
  seed: 42           # RNG seed for reproducibility
  scale_factor:      # phantom activity multiplier → expected counts per pixel
                     # higher = more counts = less noise (typical range 100–10000)

cnr:
  shrink_px: 1       # shrink rod ROI inward by N px (reduces partial-volume bias)
  erode_px: 2        # erode background mask inward by N px (avoids edge artefacts)
```

---

## `--noisy` Flag

All reconstruction and visualization scripts accept `--noisy`:

| Script | Effect |
|---|---|
| `mlem_torch_nonmpi.py --noisy` | Loads `projs_noisy_path` instead of `projs_path` |
| `map_tv_recon.py --noisy` | Loads `projs_noisy_path` instead of `projs_path` |
| `view_npz.py --noisy` | Adds a full-width sinogram panel to the output PNG |
| `run_mlem.sh --noisy` | Runs `fake_projection_noisy.py`, forwards `--noisy` to recon + view |
| `run_map_tv.sh --noisy` | Runs `fake_projection_noisy.py`, forwards `--noisy` to recon + view |

The sinogram panel shows counts vs detector bin (1D line plot for a single layout, 2D `hot` colormap for multi-layout). Both `projs_path` (noiseless) and `projs_noisy_path` (Poisson) must be set in the config regardless of which pipeline is used.

---

## CNR Evaluation

`view_npz.py` in both modules computes Bushberg CNR automatically when `phantom_path` points to a `.pt` file that contains `Rods` or `Lesions` metadata:

- **Formula:** `CNR = |μ_hot − μ_bg| / σ_bg` per rod, then averaged
- **Background mask** is auto-detected from phantom metadata:
  - `bg_diameter_mm` → circular disk
  - `ring_outer_radius_mm` → annular ring
  - neither → full frame
- Rods are excluded from the background region before computing σ_bg
- Output: per-rod CNR + mean/peak printed to stdout; CNR vs iteration panel added to the saved PNG
- Silently skipped if `phantom_path` is absent or the phantom has no rod metadata (safe for real scan data)

Tuning: adjust `cnr.shrink_px` (inset ROI edge, default 1) and `cnr.erode_px` (erode BG mask, default 2) in config.

---

## Key Implementation Details

- **No MPI** — single-process, single-GPU PyTorch only (`recon_mlem/mlem_torch_nonmpi.py`)
- **`tv_ops.py`**: all working buffers in `tv_proximal_chambolle_pock` are pre-allocated before the inner loop — zero heap allocations per inner iteration
- **`grad_2d(x, out)` / `div_2d(p, out)`** take pre-allocated output buffers; `grad_buf` must be zero-initialized once (boundary positions are never written, staying 0 for Dirichlet BC)
- **`load_system_matrix`** uses `torch.as_tensor(...).to(device)` — avoids double-copy vs `torch.tensor()`
- Convergence check uses `torch.dist(estimate, prev_est)` — no throwaway tensor allocation
- Peak memory is reported at end of each run (CPU RSS + GPU VRAM if applicable)

---

## Open Gaps

See `src/recon_map_tv/GAPS.md` for tracked implementation gaps vs the professor's engineer guide:
- **Gap 1 (High):** Smoothed TV (`+ε²` inside sqrt) not yet implemented
- **Gap 2 (Medium):** Stability bound hardcoded to 2D value
- **Gap 3 (Low):** Gradient-descent TV solver path not implemented
- **Gap 4 (Deferred):** 3D TV operators
