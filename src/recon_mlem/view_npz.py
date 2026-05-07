import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import torch
import yaml
import argparse


# ── CNR helpers ──────────────────────────────────────────────────────────────

def _to_list(x):
    return x.tolist() if hasattr(x, "tolist") else x

def _px_per_mm(meta):
    mm_pp = meta.get("mm per pixel") or meta.get("mm_per_pixel")
    if mm_pp is not None:
        return 1.0 / float(_to_list(mm_pp)[0])
    if "size in mm" in meta and "n pixels" in meta:
        sz  = _to_list(meta["size in mm"])
        npx = _to_list(meta["n pixels"])
        return float(npx[0]) / float(sz[0])
    print("Warning: pixel size unknown, assuming 4.0 px/mm (0.25 mm/px).")
    return 4.0

def _make_rod_masks(rods, H, W, shrink_px=1):
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    masks = []
    for r in rods:
        cx, cy = r["center_px"]
        rad = max(int(r["radius_px"]) - int(shrink_px), 1)
        masks.append((yy - cx) ** 2 + (xx - cy) ** 2 <= rad ** 2)
    return masks

def _bg_mask(meta, H, W, erode_px=2):
    """Circular / ring / square phantom — auto-detected from metadata."""
    ppm = _px_per_mm(meta)
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    dist2 = (yy - (H - 1) / 2.0) ** 2 + (xx - (W - 1) / 2.0) ** 2

    if "bg_diameter_mm" in meta:
        rad_px = int(round(float(meta["bg_diameter_mm"]) / 2.0 * ppm)) - erode_px
        print(f"  Circular phantom  r={rad_px} px")
        return dist2 <= rad_px ** 2

    if "ring_outer_radius_mm" in meta:
        rin_px  = int(round(float(meta.get("ring_inner_radius_mm", 0.0)) * ppm)) + erode_px
        rout_px = int(round(float(meta["ring_outer_radius_mm"]) * ppm)) - erode_px
        print(f"  Ring phantom  {rin_px}–{rout_px} px")
        return (dist2 >= rin_px ** 2) & (dist2 <= rout_px ** 2)

    print("  Square/unknown phantom — using full frame as background.")
    return torch.ones((H, W), dtype=torch.bool)

def _rods_from_phantom(ph):
    """
    Extract [{center_px, radius_px}, ...] from any phantom format.

    Priority:
    1. Top-level 'Rods' key  (contrast_ring_hotrods phantom)
    2. Top-level 'Lesions' key  (legacy)
    3. metadata.rod_points_mm + rod_diameter_mm  (targeted phantom)
    4. Auto-detect from tensor via connected components  (hotrod phantom)
    """
    if ph.get("Rods"):
        return ph["Rods"]

    if ph.get("Lesions"):
        return [{"center_px": li["center_px"], "radius_px": li["radius_px"]}
                for li in ph["Lesions"]]

    meta = ph.get("Metadata", {})

    # targeted phantom: rod centres given in mm
    if "rod_points_mm" in meta and "rod_diameter_mm" in meta:
        mm_pp = meta.get("mm_per_pixel") or meta.get("mm per pixel")
        n_px  = meta.get("n_pixels")     or meta.get("n pixels")
        mm_pp = _to_list(mm_pp)
        n_px  = _to_list(n_px)
        dx, dy   = float(mm_pp[0]), float(mm_pp[1])
        nx, ny   = int(n_px[0]),    int(n_px[1])
        radius_px = max(1, int(round(float(meta["rod_diameter_mm"]) / 2.0 / dx)))
        rods = []
        for pt in meta["rod_points_mm"]:
            x_mm, y_mm = float(pt[0]), float(pt[1])
            cx = int(round(x_mm / dx + nx / 2.0))
            cy = int(round(y_mm / dy + ny / 2.0))
            rods.append({"center_px": (cx, cy), "radius_px": radius_px})
        return rods

    # hotrod phantom: no position info saved — detect from tensor
    tensor = ph.get("Phantom tensor")
    if tensor is not None:
        try:
            from scipy import ndimage
            arr = tensor.numpy() if hasattr(tensor, "numpy") else np.array(tensor)
            thresh = arr.max() * 0.1
            if thresh > 0:
                labeled, n_comp = ndimage.label(arr > thresh)
                rods = []
                for i in range(1, n_comp + 1):
                    region = labeled == i
                    if region.sum() < 4:
                        continue
                    coords = np.argwhere(region)
                    cx, cy = coords.mean(axis=0)
                    radius_px = max(1, int(round(np.sqrt(region.sum() / np.pi))))
                    rods.append({"center_px": (int(round(cx)), int(round(cy))),
                                 "radius_px": radius_px})
                if rods:
                    print(f"  Auto-detected {len(rods)} rods from phantom tensor.")
                return rods
        except ImportError:
            print("  scipy not available — cannot auto-detect rods.")

    return []


def _cnr_frame(frame, rod_masks, bg_mask):
    """Bushberg CNR: |μ_hot − μ_bg| / σ_bg, averaged across rods."""
    bg_vals = frame[bg_mask]
    if bg_vals.numel() == 0:
        return 0.0, []
    mu_b  = bg_vals.mean()
    sd_b  = torch.sqrt(bg_vals.var(unbiased=False) + 1e-12)
    cnrs  = []
    for m in rod_masks:
        hot = frame[m]
        if hot.numel() == 0:
            continue
        cnrs.append(float((hot.mean() - mu_b).abs() / sd_b))
    return (sum(cnrs) / len(cnrs)) if cnrs else 0.0, cnrs

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base_config.yml")
    parser.add_argument("--vmax", type=float, default=None)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    data = np.load(cfg["paths"]["recon_out_path"])
    reconstructions = data["estimates"]          # (N_saved, H, W)

    pix_size   = cfg["geometry"]["pixel_size_mm"]
    save_every = cfg["mlem"]["save_every"]
    h, w       = reconstructions.shape[1], reconstructions.shape[2]
    img_extent = [-(w * pix_size) / 2, (w * pix_size) / 2,
                  -(h * pix_size) / 2, (h * pix_size) / 2]
    final_iter = reconstructions.shape[0] * save_every

    # ── CNR ──────────────────────────────────────────────────────────────────
    phantom_path = cfg["paths"].get("phantom_path", "")
    cnr_cfg      = cfg.get("cnr", {})
    shrink_px    = int(cnr_cfg.get("shrink_px", 1))
    erode_px     = int(cnr_cfg.get("erode_px",  2))
    cnr_history  = None

    if phantom_path and os.path.exists(phantom_path):
        ph   = torch.load(phantom_path, map_location="cpu", weights_only=False)
        meta = ph.get("Metadata", {})
        rods = _rods_from_phantom(ph)
        if rods:
            print(f"Computing CNR  ({len(rods)} rods, shrink={shrink_px}px, erode={erode_px}px)")
            rod_masks    = _make_rod_masks(rods, h, w, shrink_px=shrink_px)
            bg           = _bg_mask(meta, h, w, erode_px=erode_px)
            rod_excl     = _make_rod_masks(rods, h, w, shrink_px=-1)
            for m in rod_excl:
                bg &= ~m
            print(f"  BG pixels: {int(bg.sum())}")

            cnr_history = []
            for i in range(reconstructions.shape[0]):
                avg, _ = _cnr_frame(torch.from_numpy(reconstructions[i]), rod_masks, bg)
                cnr_history.append(avg)

            # Per-rod CNR for the final frame
            _, per_rod = _cnr_frame(torch.from_numpy(reconstructions[-1]), rod_masks, bg)
            print("\n--- Final-frame CNR ---")
            for k, v in enumerate(per_rod):
                print(f"  Rod {k}: {v:.3f}")
            print(f"  Mean : {sum(per_rod)/len(per_rod):.3f}")
            print(f"  Max  : {max(per_rod):.3f}  at iteration "
                  f"{(np.argmax(cnr_history)) * save_every}")
        else:
            print("No Rods/Lesions in phantom — skipping CNR.")
    else:
        print("phantom_path not set or not found — skipping CNR.")

    # ── Reconstruction image ──────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2 if cnr_history else 1,
                             figsize=(14 if cnr_history else 7, 6))
    ax_img = axes[0] if cnr_history else axes

    im = ax_img.imshow(
        reconstructions[-1].T, cmap="gray_r", extent=img_extent,
        vmax=args.vmax, origin="lower",
    )
    plt.colorbar(im, ax=ax_img, label="Intensity", fraction=0.046, pad=0.04)
    ax_img.set_title(f"MLEM — iter {final_iter}", fontsize=13, fontweight="bold")
    ax_img.set_xlabel("X (mm)")
    ax_img.set_ylabel("Y (mm)")

    # ── CNR vs iteration panel ────────────────────────────────────────────────
    if cnr_history:
        iters    = [i * save_every for i in range(len(cnr_history))]
        best_idx = int(np.argmax(cnr_history))
        ax_cnr   = axes[1]
        ax_cnr.plot(iters, cnr_history, "b-o", linewidth=2, markersize=4)
        ax_cnr.axvline(iters[best_idx], color="r", linestyle="--", linewidth=1,
                       label=f"Peak iter {iters[best_idx]} (CNR={cnr_history[best_idx]:.2f})")
        ax_cnr.set_title("CNR vs Iteration")
        ax_cnr.set_xlabel("Iteration")
        ax_cnr.set_ylabel("Mean CNR")
        ax_cnr.legend(fontsize=8)
        ax_cnr.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    base    = cfg["paths"]["recon_out_path"].replace(".npz", "")
    out_img = f"{base}_iter{final_iter}.png"
    plt.savefig(out_img, dpi=300, bbox_inches="tight")
    print(f"\nPlot saved to: {out_img}")


if __name__ == "__main__":
    main()
