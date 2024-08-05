import numpy as np
import h5py
from skimage.draw import polygon
import pathlib

top_dir = pathlib.Path(__file__).parents[1]


def get_rotated_coords(xys, theta, center):
    xy_trans = xys - center
    mr = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return np.matmul(mr, xy_trans.T).T + center


start_w = 20 
end_w = 40
xcs = np.linspace(5, 175, 32)
# generate polygon vertices
polys = np.asarray(
    [
        np.array(
            (
                (x - 0.5 * start_w, -100),
                (x - 0.5 * end_w, 800),
                (x + 0.5 * end_w, 800),
                (x + 0.5 * start_w, -100),
            )
        )
        for x in xcs
    ]
)

# create output file with h5py
N_angles = 40
with h5py.File(str(top_dir)+"/data/test_sysmat.hdf5", "w") as f:
    dset = f.create_dataset("sysmat", (N_angles, polys.shape[0], 180, 180), dtype=np.float64)
    # Create fake strips
    for rad, a_idx in zip(np.linspace(0, 2 * np.pi, N_angles), range(N_angles)):
        # fill polygon
        for poly, d_idx in zip(polys,range(polys.shape[0])):
            img = np.zeros((180, 180))
            ploy_rot = get_rotated_coords(poly, rad, np.array([90, 90]))
            xx, yy = polygon(ploy_rot[:, 0], ploy_rot[:, 1], img.shape)
            xc = poly[0, 0] + 0.5 * start_w
            # img[xx, yy] = 1 / ((xx - xc) ** 2 + (yy + 80) ** 2) ** 0.75
            img[xx, yy] = 1
            dset[a_idx, d_idx, :, :] = img
    # print(idx)
