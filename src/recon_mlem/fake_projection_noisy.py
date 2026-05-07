import numpy as np
import torch
import os
import h5py
import yaml
import argparse
from rich.progress import Progress, TimeElapsedColumn, BarColumn, TextColumn, MofNCompleteColumn


def get_flist(input_file: str) -> list:
    with open(input_file, "r") as f:
        return [f.strip() for f in f.readlines()]


def load_system_matrix(h5_path: str, device: torch.device) -> torch.Tensor:
    """Auto-detects sparse/dense HDF5 and returns a PyTorch tensor."""
    with h5py.File(h5_path, "r") as h5f:
        if "data" in h5f:
            indptr = torch.as_tensor(h5f["indptr"][:], dtype=torch.int32).to(device)
            indices = torch.as_tensor(h5f["indices"][:], dtype=torch.int32).to(device)
            data = torch.as_tensor(h5f["data"][:], dtype=torch.float32).to(device)
            shape = tuple(h5f.attrs["shape"])
            return torch.sparse_csr_tensor(indptr, indices, data, size=shape, device=device)
        elif "ppdfs" in h5f:
            return torch.as_tensor(h5f["ppdfs"][:], dtype=torch.float32).to(device)
        else:
            raise ValueError(f"Unknown matrix format in {h5_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Forward-project a phantom and add Poisson noise to simulate a realistic acquisition."
    )
    parser.add_argument("--config", default="configs/base_config.yml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    noise_cfg = cfg.get("noise", {})
    seed = int(noise_cfg.get("seed", 42))
    scale_factor = float(noise_cfg.get("scale_factor", 1.0))

    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Noise seed: {seed}  |  Scale factor: {scale_factor:.4g}")

    flist = get_flist(cfg["paths"]["flist_path"])
    img_size = cfg["geometry"]["img_dim"]

    # --- Phantom Loading ---
    phantom_data = torch.load(cfg["paths"]["phantom_path"], map_location="cpu", weights_only=False)
    phantom_tensor = phantom_data["Phantom tensor"].float()

    h, w = phantom_tensor.shape
    pad_h, pad_w = (img_size - h) // 2, (img_size - w) // 2
    phantom_padded = torch.nn.functional.pad(
        phantom_tensor, (pad_w, pad_w, pad_h, pad_h), "constant", 0
    )

    # Scale activity values to expected counts, then flatten to column vector [sfov, 1]
    phantom_flat = (phantom_padded * scale_factor).view(-1, 1).to(device)

    all_projs = []
    progress = Progress(
        TextColumn("[blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    )

    with torch.no_grad(), progress:
        task = progress.add_task("Forward projecting + Poisson noise...", total=len(flist))
        for fname in flist:
            matrix = load_system_matrix(fname, device)

            if matrix.is_sparse_csr:
                expected = torch.sparse.mm(matrix, phantom_flat)
            else:
                expected = torch.matmul(matrix, phantom_flat)

            # expected: [n_detectors, 1] — expected counts per detector bin
            expected = expected.squeeze().clamp_min_(0.0)

            # Sample Poisson noise: each element is an independent draw from Poisson(lambda)
            noisy = torch.poisson(expected)

            all_projs.append(noisy.cpu())
            progress.update(task, advance=1)

    final_projs = torch.stack(all_projs, dim=0).numpy().astype(np.float32)

    out_path = cfg["paths"]["projs_noisy_path"]
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    np.save(out_path, final_projs)

    print(f"\nOutput shape : {final_projs.shape}  (layouts × detector bins)")
    print(f"Count stats  : mean={final_projs.mean():.2f}  max={final_projs.max():.0f}  nonzero={np.count_nonzero(final_projs)}")
    print(f"Saved to     : {out_path}")


if __name__ == "__main__":
    main()
