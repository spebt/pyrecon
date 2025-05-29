def get_derenzo_section_xy(
    base_xy: tuple[int, int] = (0, 0),
    pitch_half: int = 2,
    N_layer: int = 3,
    sId: int = 0,
):
    import numpy
    import math

    pitch_short = int(pitch_half * math.sqrt(3))
    coeff_short = numpy.array([0])
    coeff_long = numpy.array([0])
    for id in range(1, N_layer):
        coeff_short = numpy.concatenate((coeff_short, numpy.full(id + 1, id)))
        coeff_long = numpy.concatenate((coeff_long, numpy.arange(-id, id + 1, 2)))
    uu = coeff_short * pitch_short
    vv = coeff_long * pitch_half
    if sId == 0:
        xx = vv + base_xy[0]
        yy = uu + base_xy[1]
    elif sId == 1:
        xx = vv + base_xy[0] - pitch_half * (N_layer - 1)
        yy = -uu + base_xy[1] + pitch_short * (N_layer - 1)
    elif sId == 2:
        xx = vv + base_xy[0] - pitch_half * (N_layer - 1)
        yy = uu + base_xy[1] - pitch_short * (N_layer - 1)
    elif sId == 3:
        xx = vv + base_xy[0]
        yy = base_xy[1] - uu
    elif sId == 4:
        xx = vv + base_xy[0] + pitch_half * (N_layer - 1)
        yy = uu + base_xy[1] - pitch_short * (N_layer - 1)
    elif sId == 5:
        xx = vv + base_xy[0] + pitch_half * (N_layer - 1)
        yy = -uu + base_xy[1] + pitch_short * (N_layer - 1)
    else:
        raise ValueError("Invalid section ID")
    return numpy.array([xx, yy]).T


def derenzo_phantom(**kwargs):
    import math
    import numpy as np
    from skimage.draw import disk

    shape = kwargs.get("shape", (100, 100))
    radii = kwargs.get("radii", np.array([1, 2, 3, 4, 5, 5]))
    pitches = kwargs.get("pitches", np.array([2, 3, 4, 5, 6, 7]))

    hw = shape[0] // 2
    hh = shape[1] // 2
    img = np.zeros(shape)
    # mask = np.zeros(shape)
    sr_base = shape[0] // 10
    # print(f"sr_base: {sr_base}")
    sr = kwargs.get("small_r", radii + sr_base)

    rmax = np.full(6, hw * 1.06)

    nlayers = np.ceil((rmax - sr - radii * 2) / (pitches * 2)).astype(int)
    print(f"nlayers: {nlayers}")
    sw = sr * math.cos(0.5236)
    sh = sr * math.sin(0.5236)
    base_xys = [
        (hw, hh + sr[0]),
        (hw - sw[1], hh + sh[1]),
        (hw - sw[2], hh - sh[2]),
        (hw, hh - sr[3]),
        (hw + sw[4], hh - sh[4]),
        (hw + sw[5], hh + sh[5]),
    ]
    for sid in range(0, 6):
        xyarray = get_derenzo_section_xy(
            base_xy=base_xys[sid],
            pitch_half=pitches[sid],
            N_layer=nlayers[sid],
            sId=sid,
        )
        for xy in xyarray:
            xx, yy = disk((int(xy[0]), int(xy[1])), int(radii[sid]))
            img[xx, yy] = 1
    return img


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    img = derenzo_phantom(
        shape=np.array((512, 512)),
        pitches=np.array([6, 10, 16, 20, 24, 28]),
        radii=np.array([4, 6, 10, 14, 18, 22]),
        small_r=np.array([50, 40, 50, 56, 70, 80]),
    )
    print(np.count_nonzero(img))
    fig = plt.figure(figsize=(10, 10),layout="constrained")
    ax = fig.add_subplot(111)
    ax.imshow(img, extent=(-64, 64, -64, 64), origin="lower", cmap="gray")
    ax.set_title("Derenzo Phantom")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    fig.savefig(os.path.join(output_dir, "derenzo_512x512.png"))
    np.save(os.path.join(output_dir, "derenzo-phantom_512x512.npy"), img)
    # plt.show()
