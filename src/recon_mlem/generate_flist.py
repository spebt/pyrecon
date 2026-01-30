import os
import numpy as np
import yaml
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base_config.yml")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    topdir = config['paths']['hdf5_dir']
    flist_path = config['paths']['flist_path']
    num_layouts = config['geometry']['num_layouts']
    
    # Assuming 0-39 for this specific geometry
    file_idxs = np.arange(0, num_layouts)
    fnames = [f"position_{i:03d}_ppdfs.hdf5" for i in file_idxs]
    
    os.makedirs(os.path.dirname(flist_path), exist_ok=True)
    
    with open(flist_path, "w") as f:
        for fname in fnames:
            full_path = os.path.abspath(os.path.join(topdir, fname))
            f.write(full_path + "\n")
    
    print(f"File list generated at: {flist_path}")

if __name__ == "__main__":
    main()