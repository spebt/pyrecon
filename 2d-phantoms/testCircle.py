from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

# Matrix dimension 90*90*32*6*8
NImgX_ = 90
NImgY_ = 90
NDetY_ = 32
NModule_ = 6
NDetX_ = 8

radius = 6.5
Npts = int(np.ceil(radius))
totNpts = Npts*2-1
img = np.zeros((totNpts, totNpts))
print(img.shape)


arrRadiiSqr = np.ones((totNpts, totNpts))*(radius**2)
print(arrRadiiSqr.shape)
xy_prime = np.linspace(-Npts+1, Npts-1, totNpts)
xv, yv = np.meshgrid(xy_prime, xy_prime)
print(xy_prime)
arrDistSqr = xv**2 + yv**2
print(arrDistSqr.shape)
num = 0.5
# img=np.where(arrDistSqr<arrRadiiSqr,num,img)
# img = arrDistSqr

# img = np.zeros((NImgX_,NImgY_))
# Plot the phantom

fig, ax = plt.subplots(1, 1, figsize=(4, 3), dpi=300)

# canvas = FigureCanvasAgg(fig)
xl = ax.set_xlabel("Image Voxel x", fontsize=12)
yl = ax.set_ylabel("Image Voxel y", fontsize=12)
ax.xaxis.set_major_locator(MultipleLocator(5))
ax.xaxis.set_minor_locator(MultipleLocator(1))
ax.yaxis.set_major_locator(MultipleLocator(5))
ax.yaxis.set_minor_locator(MultipleLocator(1))
# plt.rcParams["font.family"] = "sans"
# ax = fig.add_subplot()
ax.grid(color='r', linestyle='-', linewidth=0.5, which='both')
xy = np.array([49, 49])
# ax.add_patch(plt.Circle(xy, radius=1, color='0.8'))
ax.set_xlim(-1, totNpts+1)
ax.set_ylim(-1, totNpts+1)
imshow_obj = ax.imshow(img, extent=(0, totNpts, 0, totNpts))

ax.set_aspect("equal")
cbar = fig.colorbar(imshow_obj)
imshow_obj.set_clim(0, radius)

plt.tight_layout()
# numpy array.
# canvas.draw()
#
# rgba = np.asarray(canvas.buffer_rgba())
# print(rgba.shape)
# print(rgba[100,709])
# ... and pass it to PIL.
# im = Image.fromarray(rgba)
# im.show()


# pltTitle=ax.set_title('Phantom Image',fontsize=18)
# This image can then be saved to any format supported by Pillow, e.g.:
# im.save("test.bmp")
plt.show()
# Save the phantom as numpy npz file.
outFname = 'rods-phantom.npz'
# np.savez(outFname,phantom.astype(np.float32))
