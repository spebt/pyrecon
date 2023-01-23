import sys
import numpy as np
import struct
import matplotlib.pyplot as plt
import matplotlib as mpl



# Matrix dimension 90*90*32*6*8
NImgX_ = 90
NImgY_ = 90
NDetY_ = 32
NModule_ = 6
NDetX_ = 8

# Create phantom for test.
phantom =  np.zeros((NImgX_, NImgY_))
for idxx in range(90):
    for idxy in range(90):
        xDist=idxx-44
        yDist=idxy-44
        if xDist**2+yDist**2 == 30**2 or xDist**2+yDist**2 == 14**2:
            for local_idxx in range(6):
                for local_idxy in range(6):
                    if (local_idxx-2)**2 + (local_idxy-2)**2 > 4 and (local_idxx-2)**2 + (local_idxy-2)**2 < 9:
                        phantom[idxx+local_idxx-2, idxy+local_idxy-2] = 0.5

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

# Save the phantom as numpy npz file.
outFname = 'rods-phantom.npz'
np.savez(outFname,phantom.astype(np.float32))
