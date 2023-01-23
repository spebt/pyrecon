import sys
import numpy as np
import struct
import matplotlib.pyplot as plt
import matplotlib as mpl
# import matplotlib.animation as animation
# from matplotlib.animation import FuncAnimation

# from matplotlib import rc
# from IPython import display


inFname = 'rods-phantom/rods-phantom.npz'
dataUnpack = np.load(inFname)
# inF = open('{:s}'.format(inFname), 'rb')
# 90*90*32*6*8
NImgX_ = 90
NImgY_ = 90
# NDetY_ = 32
# NModule_ = 6
# NDetX_ = 8
dataSize = NImgX_*NImgY_
# dataUnpack = np.asarray(struct.unpack('f'*dataSize, inF.read(dataSize*4)))
# dataMatrix = dataUnpack.reshape((NDetX_ * NModule_*NDetY_, NImgX_*NImgY_))
# inF.close()
print("\nComplete Read-in Data!")
# imgTemplate = np.zeros((NImgX_, NImgY_))
# print("{:>28}:\t{:}".format("Read-in System Matrix Shape", dataMatrix.shape))

# # Remove zero rows
# sysMatrix = dataMatrix[~np.all(dataMatrix == 0, axis=1)]
# print("{:>28}:\t{:}".format("Reduced System Matrix Shape", sysMatrix.shape))
# Create phantom for test.
# phantom = dataUnpack.reshape((NImgX_,NImgY_))
phantom = dataUnpack['arr_0'].reshape((NImgX_,NImgY_))
print(phantom.shape)
# phantom[49, 49] = 0.5


# Plot the phantom
plt.rcParams["figure.figsize"] = (16,12)
# plt.rcParams["font.family"] = "sans"
fig, ax = plt.subplots()
imshow_obj= ax.imshow(phantom, cmap=mpl.colormaps['gray'])
pltTitle=ax.set_title('Phantom Image',fontsize=18)
cbar=fig.colorbar(imshow_obj)
ax.grid(color='r', linestyle='-', linewidth=0.5)
xl=ax.set_xlabel("Image Voxel x", fontsize=18)
yl=ax.set_ylabel("Image Voxel y" ,fontsize=18)
ax.set_xlim(0,89)
ax.set_ylim(0,89)
imshow_obj.set_clim(0,1)
plt.tight_layout()
# plt.savefig("Phantom.png")
plt.show()

