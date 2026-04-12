import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import yaml
import argparse

def main():
    _here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.path.join(_here, "configs", "base_config.yml"))
    parser.add_argument("--vmax", type=float, default=None, help="Colormap upper limit (default: auto)")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    data = np.load(cfg['paths']['recon_out_path'])
    reconstructions = data['estimates']  # (N_saved, H, W)

    # --- Key tracking parameters ---
    beta      = float(cfg['map_tv']['beta'])
    tau       = float(cfg['map_tv']['tau'])
    sigma     = float(cfg['map_tv']['sigma'])
    theta     = float(cfg['map_tv']['theta'])
    n_outer   = int(cfg['map_tv']['n_outer'])
    n_inner   = int(cfg['map_tv']['n_inner'])
    save_every= int(cfg['map_tv']['save_every'])
    pix_size  = float(cfg['geometry']['pixel_size_mm'])

    n_saved     = reconstructions.shape[0]
    final_iter  = n_saved * save_every
    h, w        = reconstructions.shape[1], reconstructions.shape[2]
    img_extent  = [-(w*pix_size)/2, (w*pix_size)/2, -(h*pix_size)/2, (h*pix_size)/2]

    # --- Parameter tag (used in title + filename) ---
    param_tag = f"b{beta}_t{tau}_s{sigma}_th{theta}_out{n_outer}_in{n_inner}"

    # --- Figure: final reconstruction + iteration strip ---
    fig = plt.figure(figsize=(14, 7))
    gs  = fig.add_gridspec(1, 2, width_ratios=[2, 1], wspace=0.35)

    # Left: final reconstruction
    ax_main = fig.add_subplot(gs[0])
    im = ax_main.imshow(
        reconstructions[-1],
        cmap='gray_r',
        extent=img_extent,
        vmax=args.vmax,
        origin='lower',
    )
    plt.colorbar(im, ax=ax_main, label="Intensity", fraction=0.046, pad=0.04)
    ax_main.set_title(f"MAP-TV  —  iter {final_iter}", fontsize=13, fontweight='bold')
    ax_main.set_xlabel("X (mm)")
    ax_main.set_ylabel("Y (mm)")

    # Parameter box (bottom-left of main image)
    param_text = (
        f"β = {beta}\n"
        f"τ = {tau}   σ = {sigma}   θ = {theta}\n"
        f"outer iters = {n_outer}   inner iters = {n_inner}\n"
        f"pixel = {pix_size} mm   img = {w}×{h}"
    )
    ax_main.text(
        0.02, 0.02, param_text,
        transform=ax_main.transAxes,
        fontsize=8, verticalalignment='bottom',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.75),
    )

    # Right: grid of saved snapshots (up to 9)
    ax_strip = fig.add_subplot(gs[1])
    ax_strip.axis('off')
    n_show   = min(n_saved, 9)
    indices  = np.linspace(0, n_saved - 1, n_show, dtype=int)
    cols     = 3
    rows     = int(np.ceil(n_show / cols))
    vmax_val = args.vmax if args.vmax else reconstructions.max()

    for plot_idx, rec_idx in enumerate(indices):
        sub_ax = fig.add_axes([
            gs[1].get_position(fig).x0 + (plot_idx % cols) * gs[1].get_position(fig).width / cols,
            gs[1].get_position(fig).y0 + (rows - 1 - plot_idx // cols) * gs[1].get_position(fig).height / rows,
            gs[1].get_position(fig).width / cols * 0.88,
            gs[1].get_position(fig).height / rows * 0.88,
        ])
        sub_ax.imshow(reconstructions[rec_idx], cmap='gray_r', vmin=0, vmax=vmax_val, origin='lower')
        sub_ax.set_title(f"it {(rec_idx+1)*save_every}", fontsize=6, pad=2)
        sub_ax.axis('off')

    # --- Save with parameter tag in filename ---
    base      = cfg['paths']['recon_out_path'].replace(".npz", "")
    out_img   = f"{base}_{param_tag}.png"
    plt.savefig(out_img, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {out_img}")
    plt.show()

if __name__ == "__main__":
    main()
