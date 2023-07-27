import sys
import time
import numpy as np
import struct
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import rich.progress

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

# Open the file.
inFname = 'sysmat_8layer_noslit_1100_idxT0_numT1in1mm_IZ0_DZ0_100.sysmat'
dataSize = NImgX_*NImgY_*NDetY_*NModule_*NDetX_

with rich.progress.open('../data/sysmat3/{:s}'.format(inFname), 'rb') as inF:
    # Read in the matrix
    dataUnpack = np.asarray(struct.unpack('f'*dataSize, inF.read(dataSize*4)))
    # Reshape the 5D array into a 2D matrix
    dataMatrix = dataUnpack.reshape((NDetX_ * NModule_*NDetY_, NImgX_*NImgY_))
    # inF.close()

print("Complete Read-in Data!")
imgTemplate = np.zeros((NImgX_, NImgY_))
print("{:>28}:\t{:}".format("Read-in System Matrix Shape", dataMatrix.shape))

# Remove zero rows from the matrix
sysMatrix = dataMatrix[~np.all(dataMatrix == 0, axis=1)]
print("{:>28}:\t{:}".format("Reduced System Matrix Shape", sysMatrix.shape))

# Read in the phantom
inFname = '2d-phantoms/circle-phantom.npz'
dataUnpack = np.load(inFname)
dataSize = NImgX_*NImgY_
phantom = dataUnpack['arr_0'].reshape((NImgX_, NImgY_))

# Calculate forward projection
projection = np.matmul(sysMatrix, phantom.flatten())
print("{:>28}:\t{:}".format("Projection Shape", projection.shape))

# Implementation of the recursive Maximum-Likelihood Expectation-
# Maximization (ML-EM) algorithm.


def backwardProj(lastArr, projArr, sysMat):
    forwardLast = np.matmul(sysMat, lastArr)
    quotients = projArr/forwardLast
    return np.matmul(quotients, sysMat)/np.sum(sysMat, axis=0)*lastArr


# Iterate for 5000 times, start from a flat image with all ones.
NIteration = 5000
# reconImg = backwardProj(np.ones(NImgX_*NImgY_), projection, sysMatrix)
reconImg = np.ones(NImgX_*NImgY_)
# Plot the reconstructed image
plt.rcParams["figure.figsize"] = (16, 12)
fig, ax = plt.subplots()
imshow_obj = ax.imshow(phantom, cmap=plt.get_cmap('nipy_spectral'))
pltTitle = ax.set_title('Reconstructed Image', fontsize=18)
cbar = fig.colorbar(imshow_obj)
ax.grid(color='r', linestyle='-', linewidth=0.5)
xl = ax.set_xlabel("Image Voxel x", fontsize=18)
yl = ax.set_ylabel("Image Voxel y", fontsize=18)
ax.set_xlim(0, NImgX_-1)
ax.set_ylim(0, NImgY_-1)
imshow_obj.set_clim(0, 1)
plt.tight_layout()


def update(frame, myFrames, sampleRate, imshow_obj, pltTitle, cbar):
    # print("Generating Frame# {:5d}".format(frame), end='\r', flush=True)
    # print("Calculating Iteration# {:<5d}".format(frame), end='\r', flush=True)
    pltTitle.set_text(
        'Reconstructed Image Iteration#: {:>5d}'.format(int(frame/sampleRate)))
    imshow_obj.set_data(myFrames[frame])
    imshow_obj.norm.autoscale(imshow_obj._A)
    # imshow_obj.set_clim(0, 1)
    cbar.update_normal(imshow_obj.colorbar.mappable)

    return (imshow_obj, pltTitle,)


sampleRate_ = 0.1
totalSample = int(NIteration*sampleRate_)
myFrames = np.zeros((totalSample, NImgX_, NImgY_))


progress = Progress(
    TextColumn("[progress.description]{task.description}"),
    BarColumn(),
    # TaskProgressColumn(),
    "{task.completed}/{task.total}",
    TimeRemainingColumn(),
)

with progress:
    progress.console.print("Iterative reconstruction calculation...")
    task1 = progress.add_task("Iteration:", total=NIteration)
    for iter in range(NIteration):
        reconImg = backwardProj(reconImg, projection, sysMatrix)
        if (iter*sampleRate_).is_integer():
            myFrames[int(iter*sampleRate_)] = reconImg.reshape((NImgX_, NImgY_))
        progress.advance(task1)


ani = FuncAnimation(fig, update, fargs=(myFrames, sampleRate_, imshow_obj, pltTitle, cbar,), frames=np.arange(0, totalSample),
                    repeat=1,
                    interval=100,
                    blit=False)
# plt.show()
outFname = "circle-recon.mp4"
writervideo = animation.FFMpegWriter(fps=24)
with progress:
    progress.console.print("Save the reconstructed image...")
    task2 = progress.add_task("Saving frame:", total=totalSample)
    ani.save(outFname, writer=writervideo, progress_callback=lambda i,totalSample : progress.advance(task2),)
plt.close()

# Plot the reconstructed image
# plt.rcParams["figure.figsize"] = (16,12)
# fig, ax = plt.subplots()
# imshow_obj= ax.imshow(reconImg.reshape((NImgX_, NImgY_)), cmap=mpl.colormaps['gray'])
# pltTitle=ax.set_title('Phantom Image',fontsize=18)
# cbar=fig.colorbar(imshow_obj)
# ax.grid(color='r', linestyle='-', linewidth=0.5)
# xl=ax.set_xlabel("Image Voxel x", fontsize=18)
# yl=ax.set_ylabel("Image Voxel y" ,fontsize=18)
# ax.set_xlim(0,89)
# ax.set_ylim(0,89)
# imshow_obj.set_clim(0,1)
# plt.tight_layout()
# plt.savefig('rods-reconImg-rods-iteration-{:04d}.png'.format(NIteration))
# plt.close()
