import sys
import math
import numpy as np
import struct
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import rich.progress
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TaskID,
    TextColumn,
    TimeRemainingColumn,
    TaskProgressColumn,
)

# Matrix dimension 90*90*32*6*8
NImgX_ = 90
NImgY_ = 90
NDetY_ = 32
NModule_ = 6
NDetX_ = 8


radiusR = 20
bgRadius = radiusR*2
rDot = np.linspace(2, 4.5, 6)*radiusR/20

# Read in the reconstructed data
inFname = 'contrast-recon-data.npz'
dataUnpack = np.load(inFname)
# dataSize = 50
reconData = dataUnpack['arr_0']
print(f"Reconstructed data size: {reconData.shape}")

# Plot the CNR
# plt.rcParams["figure.figsize"] = (16, 12)

# bgIndex = np.nonzero(regionMapFlat == bg)
# objIndex = []
# for idx in range(0, 6):
#     objIndex.append(np.nonzero(regionMapFlat == values[idx]))

# fig, ax = plt.subplots(1, 1, figsize=(10, 10), dpi=150)
# # imshow_obj = ax.imshow(regionMap.transpose(), extent=(0, NImgX_, 0, NImgX_),
# #                        cmap=plt.get_cmap('nipy_spectral'), interpolation='None', aspect='equal')
# imshow_obj = ax.imshow(regionMap.transpose(),
#                        cmap=plt.get_cmap('nipy_spectral'),interpolation='None')

# ax.grid(color='b', linestyle='-', linewidth=0.5, which='major')
# ax.xaxis.set_major_locator(MultipleLocator(10))
# ax.xaxis.set_minor_locator(MultipleLocator(1))
# ax.yaxis.set_major_locator(MultipleLocator(10))
# ax.yaxis.set_minor_locator(MultipleLocator(1))
# ax.set_xlim(-0.5, NImgX_-0.5)
# ax.set_ylim(-0.5, NImgY_-0.5)
# # ax.set_aspect("equal")
# cbar = fig.colorbar(imshow_obj)

# for idx in range(0, 6):
#     ax.annotate(f'{values[idx]}', xy=(centerX+disX[idx],
#                 centerY+disY[idx]),  ha="left", va="bottom")
#     x=np.linspace(0,NImgX_-1,NImgX_)
#     ax.plot(x,(x-centerX)*np.tan(math.pi/3*range(0, 6)[idx])+centerY,color='r',linestyle='-.',linewidth=1)
# plt.tight_layout()
# plt.show()
# print(np.linspace(10))


NIteration = 5000

# Plot the reconstructed image
plt.rcParams["figure.figsize"] = (16, 12)
fig, ax = plt.subplots()
imshow_obj = ax.imshow(np.ones((NImgX_,NImgY_)), cmap=plt.get_cmap('nipy_spectral'))
pltTitle = ax.set_title('Reconstructed Image', fontsize=18)
cbar = fig.colorbar(imshow_obj)
ax.grid(color='r', linestyle='-', linewidth=0.5)
xl = ax.set_xlabel("Image Voxel x", fontsize=18)
yl = ax.set_ylabel("Image Voxel y", fontsize=18)
ax.set_xlim(-0.5, NImgX_-0.5)
ax.set_ylim(-0.5, NImgY_-0.5)
# imshow_obj.set_clim(0, 1)
plt.tight_layout()


def update(frame, reconData, imshow_obj, pltTitle, cbar):
    # print("Generating Frame# {:5d}".format(frame), end='\r', flush=True)
    # print("Calculating Iteration# {:<5d}".format(frame), end='\r', flush=True)
    pltTitle.set_text(
        'Reconstructed Image Iteration#: {:>5d}'.format(int(frame*100)))
    imshow_obj.set_data(reconData[frame].reshape(NImgX_,NImgY_).transpose())
    imshow_obj.norm.autoscale(imshow_obj._A)
    # imshow_obj.set_clim(0, 1)
    cbar.update_normal(imshow_obj.colorbar.mappable)

    return (imshow_obj, pltTitle,)


Nframes=reconData.shape[0]


progress = Progress(
    TextColumn("[progress.description]{task.description}"),
    BarColumn(),
    # TaskProgressColumn(),
    "{task.completed}/{task.total}",
    TimeRemainingColumn(),
)

ani = FuncAnimation(fig, update, fargs=(reconData, imshow_obj, pltTitle, cbar,), frames=np.arange(0, Nframes),
                    repeat=1,
                    interval=100,
                    blit=False)
# plt.show()
outFname = "circle-recon.mp4"
writervideo = animation.FFMpegWriter(fps=24)
with progress:
    progress.console.print("Save the reconstructed image...")
    task2 = progress.add_task("Saving frame:", total=Nframes)
    ani.save(outFname, writer=writervideo, progress_callback=lambda i,Nframes : progress.advance(task2),)
plt.close()