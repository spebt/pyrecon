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

bg = 0.2
radiusR = 20
bgRadius = radiusR*2
rDot = np.linspace(2, 4.5, 6)*radiusR/20
values = range(1, 7)
NIteration = 20000
scale = 100
# Read in the reconstructed data
inFname = 'contrast-recon-data.npz'
dataUnpack = np.load(inFname)
# dataSize = 50
reconData = dataUnpack['arr_0']
print(f"Reconstructed data size: {reconData.shape}")
Nframes=reconData.shape[0]
phantom=np.load('../2d-phantoms/circle-phantom.npz')['arr_0']
# Plot the CNR
# plt.rcParams["figure.figsize"] = (16, 12)

# Calculate the CNR
bgIndex = np.nonzero(phantom.flatten() == bg)
objIndex = []
for idx in range(0, 6):
    objIndex.append(np.nonzero(phantom.flatten() == values[idx]))
# print(len(bgIndex[0]))
# print(reconData[0][objIndex[0]].shape)
CNRs = np.zeros((6, Nframes))
bgMeans=np.zeros(Nframes)
bgStds=np.zeros(Nframes)
for iFrame in range(0, 200):
      bgMean = np.mean(reconData[iFrame][bgIndex])
      bgStd = np.std(reconData[iFrame][bgIndex])
      bgMeans[iFrame] = bgMean
      bgStds[iFrame] = bgStd
      for idx in range(0, 6):
        objMean=np.mean(reconData[iFrame][objIndex[idx]])
        CNRs[idx, iFrame]=(objMean - bgMean)/bgStd

fig, axs = plt.subplots(3, 1, figsize=(8, 6), dpi=150,sharex=True)
axs[0].set_title("Background Mean")
axs[0].plot(np.arange(1,NIteration,100),bgMeans)
axs[0].set_xlim(-1,NIteration)
axs[1].set_title("Background Standard Deviation")
axs[1].plot(np.arange(1,NIteration,100),bgStds)
axs[2].set_title("Background Mean/Std")
axs[2].plot(np.arange(1,NIteration,100),bgMeans/bgStds)
axs[2].set_xlabel("Number of Iterations")
plt.tight_layout()

fig, axs = plt.subplots(6, 1, figsize=(8, 6), dpi=150,sharex=True)
axs[0].set_xlim(-1,NIteration)
for idx in range(0,6):
    axs[idx].plot(np.arange(1,NIteration,100), CNRs[idx])
    axs[idx].annotate(f'Circle # {idx}', xy=(0,axs[idx].get_ylim()[1]))
axs[5].set_xlabel("Number of Iterations")
plt.tight_layout()
plt.show()
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

# print(np.linspace(10))








# plt.close()