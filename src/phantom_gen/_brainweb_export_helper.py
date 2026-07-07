import argparse
import json
import os

import numpy as np
from scipy import ndimage

from brainweb_dl._brainweb import BIG_RES_SHAPE, _request_get_brainweb


CLASS_LABELS = [
    "background",
    "csf",
    "gray_matter",
    "white_matter",
    "fat",
    "muscles",
    "muscles_skin",
    "skull",
    "vessels",
    "around_fat",
    "dura",
    "bone_marrow",
]

MAX_BRAINWEB_FOV_MM = 217.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a BrainWeb-derived 2D activity phantom and export it as NPZ."
    )
    parser.add_argument("--subject", type=int, default=4)
    parser.add_argument("--shape", type=int, nargs=2, required=True, metavar=("H", "W"))
    parser.add_argument("--brainweb-dir", type=str, required=True)
    parser.add_argument("--cache-dir", type=str, required=True)
    parser.add_argument("--out-npz", type=str, required=True)
    parser.add_argument("--tissue-weights-json", type=str, required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.brainweb_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)

    shape = tuple(int(v) for v in args.shape)
    weights_by_label = json.loads(args.tissue_weights_json)
    weights = np.asarray(
        [float(weights_by_label.get(label, 0.0)) for label in CLASS_LABELS],
        dtype=np.float32,
    )

    cache_path = os.path.join(args.brainweb_dir, f"brainweb_s{args.subject:02d}_crisp.npy")
    if os.path.exists(cache_path) and not args.force:
        raw_segmentation = np.load(cache_path)
    else:
        raw_segmentation, _ = _request_get_brainweb(
            f"subject{args.subject:02d}_crisp",
            cache_path,
            force=args.force,
            dtype=np.uint16,
            shape=BIG_RES_SHAPE,
        )
        raw_segmentation = (raw_segmentation >> 4).astype(np.uint8)
        np.save(cache_path, raw_segmentation)

    center_idx = raw_segmentation.shape[0] // 2
    segmentation_2d = np.flip(raw_segmentation[center_idx, :, :], axis=0)
    zoom_factors = (
        float(shape[0]) / float(segmentation_2d.shape[0]),
        float(shape[1]) / float(segmentation_2d.shape[1]),
    )
    segmentation = ndimage.zoom(segmentation_2d, zoom=zoom_factors, order=0).astype(np.int16)
    activity = weights[segmentation].astype(np.float32)
    activity = np.clip(activity, a_min=0.0, a_max=None)

    peak = float(activity.max())
    if peak > 0.0:
        activity /= peak

    pixel_size_mm = np.full(2, MAX_BRAINWEB_FOV_MM / float(max(shape)), dtype=np.float32)
    size_mm = pixel_size_mm * np.asarray(shape, dtype=np.float32)

    np.savez_compressed(
        args.out_npz,
        activity=activity,
        segmentation=segmentation,
        class_labels=np.asarray(CLASS_LABELS, dtype=object),
        size_mm=size_mm,
        mm_per_pixel=pixel_size_mm,
        weights=weights,
        subject=np.asarray([args.subject], dtype=np.int32),
    )


if __name__ == "__main__":
    main()
