import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import torch
from torch import float32 as torch_float32
from torch import save as torch_save


SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
PYRECON_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
WORKSPACE_ROOT = os.path.abspath(os.path.join(PYRECON_ROOT, ".."))
HELPER_PATH = os.path.join(SCRIPT_DIR, "_brainweb_export_helper.py")
DEFAULT_OUT_PREFIX = os.path.join(PYRECON_ROOT, "custom_phantoms", "brainweb_subject04")
DEFAULT_BRAINWEB_DIR = os.path.join(PYRECON_ROOT, "data", "brainweb_cache")
DEFAULT_CACHE_DIR = os.path.join(PYRECON_ROOT, "data", "mrtwin_cache")
DEFAULT_BRAINWEB_VENDOR_DIR = os.path.join(WORKSPACE_ROOT, ".vendor", "brainweb312")
DEFAULT_BRAINWEB_PYTHON = (
    "/Users/siddharthmehta/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3"
)


# -----------------------------------------------------------------------------
# Manually editable defaults
# -----------------------------------------------------------------------------
# Any CLI argument provided by the user overrides the corresponding default.

DEFAULT_SUBJECT = 4
DEFAULT_SHAPE = (512, 512)
DEFAULT_MODE = "both"
DEFAULT_LESION_TYPE = "reduced"
DEFAULT_LESION_CENTER_MM = (34.0, -16.0)
DEFAULT_LESION_RADIUS_MM = 8.0
DEFAULT_LESION_SEVERITY = 0.45
DEFAULT_CORTEX_THRESHOLD = 0.28

DEFAULT_TISSUE_WEIGHTS = {
    "background": 0.0,
    "csf": 0.08,
    "gray_matter": 1.00,
    "white_matter": 0.58,
    "fat": 0.0,
    "muscles": 0.0,
    "muscles_skin": 0.0,
    "skull": 0.0,
    "vessels": 0.75,
    "around_fat": 0.0,
    "dura": 0.12,
    "bone_marrow": 0.0,
}


def build_coordinate_grids(
    shape: Sequence[int],
    size_mm: Sequence[float],
) -> Tuple[torch.Tensor, torch.Tensor]:
    x_coords_mm = torch.linspace(-float(size_mm[0]) / 2.0, float(size_mm[0]) / 2.0, int(shape[0]))
    y_coords_mm = torch.linspace(-float(size_mm[1]) / 2.0, float(size_mm[1]) / 2.0, int(shape[1]))
    return torch.meshgrid(x_coords_mm, y_coords_mm, indexing="ij")


def mm_to_px(
    center_mm: Tuple[float, float],
    mm_per_pixel: Sequence[float],
    shape: Sequence[int],
) -> Tuple[int, int]:
    cx = int(round(center_mm[0] / float(mm_per_pixel[0]) + float(shape[0]) / 2.0))
    cy = int(round(center_mm[1] / float(mm_per_pixel[1]) + float(shape[1]) / 2.0))
    return cx, cy


def ellipse_mask(
    xx: torch.Tensor,
    yy: torch.Tensor,
    center_mm: Tuple[float, float],
    radii_mm: Tuple[float, float],
    angle_deg: float = 0.0,
) -> torch.Tensor:
    cx, cy = center_mm
    rx, ry = radii_mm
    ang = torch.deg2rad(torch.tensor(float(angle_deg), dtype=torch_float32))
    cos_a = torch.cos(ang)
    sin_a = torch.sin(ang)
    dx = xx - cx
    dy = yy - cy
    xr = dx * cos_a + dy * sin_a
    yr = -dx * sin_a + dy * cos_a
    return (xr / rx) ** 2 + (yr / ry) ** 2 <= 1.0


def build_lesion_mask(
    xx: torch.Tensor,
    yy: torch.Tensor,
    support_mask: torch.Tensor,
    lesion_center_mm: Tuple[float, float],
    lesion_radius_mm: float,
) -> Tuple[torch.Tensor, Tuple[float, float]]:
    req_cx, req_cy = lesion_center_mm

    def make_mask(cx: float, cy: float) -> torch.Tensor:
        lesion = (
            ellipse_mask(xx, yy, center_mm=(cx, cy), radii_mm=(1.05 * lesion_radius_mm, 0.82 * lesion_radius_mm), angle_deg=18.0)
            | ellipse_mask(xx, yy, center_mm=(cx - 0.38 * lesion_radius_mm, cy + 0.08 * lesion_radius_mm), radii_mm=(0.52 * lesion_radius_mm, 0.46 * lesion_radius_mm), angle_deg=-15.0)
            | ellipse_mask(xx, yy, center_mm=(cx + 0.28 * lesion_radius_mm, cy - 0.20 * lesion_radius_mm), radii_mm=(0.42 * lesion_radius_mm, 0.34 * lesion_radius_mm), angle_deg=32.0)
        )
        return lesion & support_mask

    lesion = make_mask(req_cx, req_cy)
    if int(lesion.sum()) > 0:
        return lesion, (req_cx, req_cy)

    support_x = xx[support_mask]
    support_y = yy[support_mask]
    if support_x.numel() == 0:
        return lesion, (req_cx, req_cy)

    dist2 = (support_x - req_cx) ** 2 + (support_y - req_cy) ** 2
    best_idx = int(torch.argmin(dist2))
    adj_center = (float(support_x[best_idx]), float(support_y[best_idx]))
    lesion = make_mask(*adj_center)
    return lesion, adj_center


def apply_lesion(
    base_phantom: torch.Tensor,
    segmentation: np.ndarray,
    size_mm: Sequence[float],
    mm_per_pixel: Sequence[float],
    lesion_center_mm: Tuple[float, float],
    lesion_radius_mm: float,
    lesion_type: str,
    lesion_severity: float,
    cortex_threshold: float,
) -> Tuple[torch.Tensor, Dict, torch.Tensor]:
    phantom = base_phantom.clone()
    xx, yy = build_coordinate_grids(phantom.shape, size_mm)

    if segmentation.ndim == 2:
        cortex_support = torch.from_numpy(segmentation == 2)
    else:
        gray_prob = torch.from_numpy(segmentation[2]).to(torch_float32)
        cortex_support = gray_prob >= float(cortex_threshold)
    lesion_mask, actual_center_mm = build_lesion_mask(
        xx,
        yy,
        cortex_support,
        lesion_center_mm,
        lesion_radius_mm,
    )

    if lesion_type in ("cold", "reduced"):
        phantom[lesion_mask] *= max(0.0, 1.0 - lesion_severity)
    elif lesion_type == "hot":
        phantom[lesion_mask] *= 1.0 + lesion_severity
    else:
        raise ValueError(f"Unsupported lesion type: {lesion_type}")

    lesion_meta = {
        "requested_center_mm": [float(lesion_center_mm[0]), float(lesion_center_mm[1])],
        "center_mm": [float(actual_center_mm[0]), float(actual_center_mm[1])],
        "center_px": mm_to_px(actual_center_mm, mm_per_pixel, phantom.shape),
        "radius_mm": float(lesion_radius_mm),
        "radius_px": max(1, int(round(lesion_radius_mm / float(mm_per_pixel[0])))),
        "type": lesion_type,
        "severity": float(lesion_severity),
    }
    return phantom, lesion_meta, lesion_mask


def render_preview(
    phantom: torch.Tensor,
    out_png: str,
    size_mm: Sequence[float],
    lesion_meta: Optional[Dict] = None,
    lesion_mask: Optional[torch.Tensor] = None,
) -> None:
    extent = (
        -float(size_mm[0]) / 2.0,
        float(size_mm[0]) / 2.0,
        -float(size_mm[1]) / 2.0,
        float(size_mm[1]) / 2.0,
    )
    fig, ax = plt.subplots(figsize=(7, 7), dpi=150)
    im = ax.imshow(
        phantom.T.detach().cpu().numpy(),
        cmap="gray",
        origin="lower",
        extent=extent,
        interpolation="bilinear",
        aspect="equal",
    )
    if lesion_mask is not None:
        ax.contour(
            lesion_mask.T.detach().cpu().numpy().astype(float),
            levels=[0.5],
            colors=["cyan"],
            linewidths=1.0,
            origin="lower",
            extent=extent,
        )
    elif lesion_meta is not None:
        circ = plt.Circle(
            (lesion_meta["center_mm"][0], lesion_meta["center_mm"][1]),
            lesion_meta["radius_mm"],
            fill=False,
            color="red",
            linewidth=1.0,
        )
        ax.add_patch(circ)
    plt.colorbar(im, ax=ax, label="Relative activity")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title(os.path.basename(out_png).replace(".png", ""))
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def choose_brainweb_python() -> str:
    candidates = [
        os.environ.get("BRAINWEB_PYTHON"),
        DEFAULT_BRAINWEB_PYTHON,
        shutil.which("python3.12"),
        shutil.which("python3.11"),
    ]
    if sys.version_info >= (3, 10):
        candidates.append(sys.executable)

    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(
        "Could not find a Python 3.10+ interpreter for BrainWeb generation. "
        "Set BRAINWEB_PYTHON to a compatible interpreter."
    )


def run_brainweb_helper(
    subject: int,
    shape: Sequence[int],
    brainweb_dir: str,
    cache_dir: str,
    tissue_weights: Dict[str, float],
    force: bool,
) -> Dict[str, np.ndarray]:
    if not os.path.exists(DEFAULT_BRAINWEB_VENDOR_DIR):
        raise FileNotFoundError(
            f"Missing BrainWeb vendor packages at {DEFAULT_BRAINWEB_VENDOR_DIR}. "
            "Install mrtwin and brainweb-dl there, or set up an equivalent PYTHONPATH."
        )

    python_bin = choose_brainweb_python()
    with tempfile.TemporaryDirectory(prefix="brainweb_export_") as tmpdir:
        out_npz = os.path.join(tmpdir, "brainweb_export.npz")
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(
            [DEFAULT_BRAINWEB_VENDOR_DIR, env.get("PYTHONPATH", "")]
        ).strip(os.pathsep)
        cmd = [
            python_bin,
            HELPER_PATH,
            "--subject",
            str(subject),
            "--shape",
            str(int(shape[0])),
            str(int(shape[1])),
            "--brainweb-dir",
            brainweb_dir,
            "--cache-dir",
            cache_dir,
            "--out-npz",
            out_npz,
            "--tissue-weights-json",
            json.dumps(tissue_weights),
        ]
        if force:
            cmd.append("--force")
        subprocess.run(cmd, check=True, env=env)
        data = np.load(out_npz, allow_pickle=True)
        return {key: data[key] for key in data.files}


def save_variant(
    phantom: torch.Tensor,
    out_prefix: str,
    variant: str,
    metadata: Dict,
    lesions: Optional[List[Dict]] = None,
    lesion_mask: Optional[torch.Tensor] = None,
) -> Tuple[str, str]:
    pt_path = f"{out_prefix}_{variant}.pt"
    png_path = f"{out_prefix}_{variant}.png"

    out_dict = {
        "Description": f"BrainWeb-derived 2D brain phantom ({variant})",
        "Metadata": metadata,
        "Phantom tensor": phantom.to(torch_float32),
        "Phantom shape": tuple(int(v) for v in phantom.shape),
        "Phantom dtype": str(phantom.dtype),
    }
    if lesions:
        out_dict["Lesions"] = lesions

    torch_save(out_dict, pt_path)
    render_preview(
        phantom=phantom,
        out_png=png_path,
        size_mm=metadata["size_mm"],
        lesion_meta=lesions[0] if lesions else None,
        lesion_mask=lesion_mask,
    )
    return pt_path, png_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a BrainWeb-derived 2D phantom and save it as a .pt file for the pyrecon pipeline."
    )
    parser.add_argument("--subject", type=int, default=DEFAULT_SUBJECT)
    parser.add_argument("--shape", type=int, nargs=2, default=DEFAULT_SHAPE, metavar=("H", "W"))
    parser.add_argument("--out-prefix", type=str, default=DEFAULT_OUT_PREFIX)
    parser.add_argument("--brainweb-dir", type=str, default=DEFAULT_BRAINWEB_DIR)
    parser.add_argument("--cache-dir", type=str, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--mode", choices=("absent", "present", "both"), default=DEFAULT_MODE)
    parser.add_argument("--lesion-type", choices=("cold", "reduced", "hot"), default=DEFAULT_LESION_TYPE)
    parser.add_argument("--lesion-center-mm", type=float, nargs=2, default=DEFAULT_LESION_CENTER_MM)
    parser.add_argument("--lesion-radius-mm", type=float, default=DEFAULT_LESION_RADIUS_MM)
    parser.add_argument("--lesion-severity", type=float, default=DEFAULT_LESION_SEVERITY)
    parser.add_argument("--cortex-threshold", type=float, default=DEFAULT_CORTEX_THRESHOLD)
    parser.add_argument("--gray-weight", type=float, default=DEFAULT_TISSUE_WEIGHTS["gray_matter"])
    parser.add_argument("--white-weight", type=float, default=DEFAULT_TISSUE_WEIGHTS["white_matter"])
    parser.add_argument("--csf-weight", type=float, default=DEFAULT_TISSUE_WEIGHTS["csf"])
    parser.add_argument("--vessel-weight", type=float, default=DEFAULT_TISSUE_WEIGHTS["vessels"])
    parser.add_argument("--dura-weight", type=float, default=DEFAULT_TISSUE_WEIGHTS["dura"])
    parser.add_argument("--force-download", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(os.path.dirname(args.out_prefix), exist_ok=True)
    os.makedirs(args.brainweb_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)

    tissue_weights = dict(DEFAULT_TISSUE_WEIGHTS)
    tissue_weights["gray_matter"] = float(args.gray_weight)
    tissue_weights["white_matter"] = float(args.white_weight)
    tissue_weights["csf"] = float(args.csf_weight)
    tissue_weights["vessels"] = float(args.vessel_weight)
    tissue_weights["dura"] = float(args.dura_weight)

    exported = run_brainweb_helper(
        subject=int(args.subject),
        shape=tuple(int(v) for v in args.shape),
        brainweb_dir=args.brainweb_dir,
        cache_dir=args.cache_dir,
        tissue_weights=tissue_weights,
        force=bool(args.force_download),
    )

    base_phantom = torch.from_numpy(exported["activity"]).to(torch_float32)
    segmentation = exported["segmentation"]
    size_mm = exported["size_mm"].astype(np.float32).tolist()
    mm_per_pixel = exported["mm_per_pixel"].astype(np.float32).tolist()
    class_labels = [str(v) for v in exported["class_labels"].tolist()]

    common_metadata = {
        "source": "BrainWeb via MRTwin",
        "subject": int(args.subject),
        "shape": [int(v) for v in base_phantom.shape],
        "size_mm": [float(v) for v in size_mm],
        "mm_per_pixel": [float(v) for v in mm_per_pixel],
        "class_labels": class_labels,
        "tissue_weights": tissue_weights,
    }

    if args.mode in ("absent", "both"):
        pt_path, png_path = save_variant(
            phantom=base_phantom,
            out_prefix=args.out_prefix,
            variant="lesion_absent",
            metadata=common_metadata,
        )
        print(f"Saved phantom tensor: {pt_path}")
        print(f"Saved preview image: {png_path}")

    if args.mode in ("present", "both"):
        lesion_phantom, lesion_meta, lesion_mask = apply_lesion(
            base_phantom=base_phantom,
            segmentation=segmentation,
            size_mm=size_mm,
            mm_per_pixel=mm_per_pixel,
            lesion_center_mm=(float(args.lesion_center_mm[0]), float(args.lesion_center_mm[1])),
            lesion_radius_mm=float(args.lesion_radius_mm),
            lesion_type=args.lesion_type,
            lesion_severity=float(args.lesion_severity),
            cortex_threshold=float(args.cortex_threshold),
        )
        present_metadata = dict(common_metadata)
        present_metadata["lesion_support"] = "gray_matter_probability_threshold"
        present_metadata["cortex_threshold"] = float(args.cortex_threshold)
        pt_path, png_path = save_variant(
            phantom=lesion_phantom,
            out_prefix=args.out_prefix,
            variant="lesion_present",
            metadata=present_metadata,
            lesions=[lesion_meta],
            lesion_mask=lesion_mask,
        )
        print(f"Saved phantom tensor: {pt_path}")
        print(f"Saved preview image: {png_path}")


if __name__ == "__main__":
    main()
