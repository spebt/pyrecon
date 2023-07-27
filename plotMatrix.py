import numpy as np
import struct
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rc
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import matplotlib.ticker as ticker

NImgX_ = 90
NImgY_ = 90
NDetY_ = 32
NModule_ = 1
NDetX_ = 4

# Read in the matrix
fname = 'array.dat'
f=open('./{:s}'.format(fname), 'rb')
dataSize=NImgX_*NImgY_*NDetY_*NModule_*NDetX_
dataUnpack=np.asarray(struct.unpack('f'*dataSize, f.read(dataSize*4)))
dataImage=dataUnpack.reshape((NDetX_, NModule_, NDetY_, NImgX_, NImgY_))

plt.rcParams["figure.figsize"] = (16,12)

fig, ax = plt.subplots()
imshow_obj= ax.imshow(dataImage[0,0,0,:,:], cmap=mpl.colormaps['copper'],animated=True)
pltTitle=ax.set_title('Panel#: {:02d}, Detector x: {:02d}, y: {:02d}'.format(0, 0, 0))
cbar=fig.colorbar(imshow_obj)
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
# ax.grid(color='b', linestyle='-', linewidth=1)
xl=ax.set_xlabel("Image Voxel x")
yl=ax.set_ylabel("Image Voxel y")
# ax.set_xlim(0,NImgX_)
# ax.set_ylim(0,NImgY_)
plt.tight_layout()
# plt.show()

# def update(frame,args):
#     idx_x = frame // NDetY_
#     idx_y = frame % NDetY_
#     pltTitle.set_text('Panel#: {:02d}, Detector x: {:02d}, y: {:02d}'.format(args,idx_x, idx_y))
#     imshow_obj.set_data(dataImage[idx_x,args,idx_y,:,:])
#     imshow_obj.norm.autoscale(imshow_obj._A)
#     imshow_obj.set_clim(0,3e-6)
#     cbar.update_normal(imshow_obj.colorbar.mappable)
#     print("Generating Panel# {:02d}, Frame# {:3d}".format(args,frame), end='\r', flush=True)
#     return (imshow_obj,pltTitle,)

# for panIdx in range(6):
#     ani = FuncAnimation(feig, update, fargs=[panIdx],frames=np.arange(0, NDetX_*NDetY_),\
#                     repeat = 1,\
#                     interval = 100,
#                     blit=True)

# # plt.show()
#     outFname="panel-{:02}.mp4".format(panIdx)
#     writervideo = animation.FFMpegWriter(fps=24) 
#     ani.save(outFname, writer=writervideo)
plt.savefig("test.png")
plt.close()