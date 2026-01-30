import numpy as np
import matplotlib.pyplot as plt
import os
import yaml
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base_config.yml")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    data = np.load(cfg['paths']['recon_out_path'])
    reconstructions = data['estimates']
    
    # Spatial extent based on config
    pix_size = cfg['geometry']['pixel_size_mm']
    h, w = reconstructions.shape[1], reconstructions.shape[2]
    img_extent = [-(w*pix_size)/2, (w*pix_size)/2, -(h*pix_size)/2, (h*pix_size)/2]

    # Final Image Plot
    plt.figure(figsize=(8, 8))
    plt.imshow(reconstructions[-1], cmap='gray_r', extent=img_extent)
    plt.colorbar(label="Intensity")
    plt.title(f"Final Reconstruction (Iter {len(reconstructions)*cfg['mlem']['save_every']})")
    plt.xlabel("X (mm)"); plt.ylabel("Y (mm)")
    
    out_img = cfg['paths']['recon_out_path'].replace(".npz", ".png")
    plt.savefig(out_img, dpi=300)
    print(f"Plot saved to: {out_img}")
    plt.show()

if __name__ == "__main__":
    main()