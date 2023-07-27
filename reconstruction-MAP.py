import numpy as np
import struct
import matplotlib.pyplot as plt
import matplotlib as mpl
import rich.progress
from PIL import Image
import json

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
with open('../SysMatConfig/Parameters.json') as json_file:
    parameters = json.load(json_file)

NImgX_ = parameters["numImageX"]
NImgY_ = parameters["numImageY"]
NDetY_ = parameters["pixelSiPM"]
NModule_ = parameters["numPanel"]
NDetX_ = parameters["numDetectorLayer"]


# Open the file.
sysmatPath = parameters["sysmatPath"]
inFname = 'sysmatMatrix.sysmat'
dataSize = NImgX_*NImgY_*NDetY_*NModule_*NDetX_
filePath = sysmatPath+inFname

with rich.progress.open(filePath, 'rb') as inF:
    # Read in the matrix
    dataUnpack = np.asarray(struct.unpack('f'*dataSize, inF.read(dataSize*4)))
    # Reshape the 5D array into a 2D matrix
    dataMatrix = dataUnpack.reshape((NDetX_ * NModule_*NDetY_, NImgX_*NImgY_))
    # inF.close()

print("Complete Read-in Data!")


# TODO - Implement Maximum A-posterior Algorithm