from mpi4py import MPI
import numpy as np
from tqdm import tqdm
import yaml
import sys
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if len(sys.argv) < 2:
    print('\nERROR: Needs a configuration file!')
    comm.Abort(1)

try:
    with open(sys.argv[1],'r') as file:
        configs = yaml.safe_load(file)
        if rank == 0:
            print('Configuration file: %s'%sys.argv[1])
except:
    print('\nLoad config file failed!')
    comm.Abort(1)
NImgX_ = configs['NImgX']
NImgY_ = configs['NImgY']
NDetY_ = configs['NDetY']
NModule_ = configs['NModule']
NDetX_ = configs['NDetX']

sysmatBinDir = configs['system matrix bin files folder']
nInputFiles = configs['N input files']

inFileIndex = rank % nInputFiles
fName = 'sysmat_8layer_Rot_1_of_2_2mmslitin10mm_1100_idxT0_numT1in1mm_IZ0_DZ%d_100.sysmat' % inFileIndex

if rank == 0:
    print('Reading in matrices...')
data = np.fromfile('%s/%s' % (sysmatBinDir, fName), dtype='single')
dataMatrix = data.reshape((NDetX_ * NModule_*NDetY_, NImgX_*NImgY_))
sysMatrix = dataMatrix[~np.all(dataMatrix == 0, axis=1)]
matSum = np.sum(sysMatrix, axis=0, dtype=float)
validIdx = np.array(np.where(matSum != 0))
# validIdx=np.where(matSum != 0)
reducedMat = np.reshape(
    sysMatrix[:, validIdx], (sysMatrix.shape[0], validIdx.shape[1]))
# reducedMat=sysMatrix[:, validIdx]
# thisInfo="{:>28}:\t{:}".format("Reduced System Matrix Shape", reducedMat.shape)
# print(thisInfo)
startIdx = rank//nInputFiles
matrixProc = reducedMat[startIdx::configs['matrix N-divide'], :].astype(float)


with np.load(configs['phantom file']) as data:
    phantom = data['phantom'].flatten()
    reducedPhantom = phantom[np.where(matSum != 0)]
fakeData = np.matmul(matrixProc, reducedPhantom)
# # thisInfo += 'Forward projection shape: %s\n' % str(fakeData.shape)
nFOV = validIdx.shape[1]
# matrix_p=sysMatrix.astype(float)
matrixSumProc = np.sum(matrixProc, axis=0, dtype=float)
if rank == 0:
    norm = np.zeros(nFOV, dtype=float)
else:
    norm = None
thisInfo = 'rank: #%d' % rank+', sum: %f' % np.sum(matrixSumProc)
# thisInfo += ''
# print(thisInfo)
# comm.Barrier()
comm.Reduce([matrixSumProc, MPI.DOUBLE], [
            norm, MPI.DOUBLE], op=MPI.SUM, root=0)
comm.Barrier()

nIteration = configs['N iterations']
scale = configs['reconstruction storage scale factor']
if rank == 0:
    # print(np.all(norm))
    thisInfo = 'Normalization shape: %s\n' % str(norm.shape)
    # thisInfo += 'Partial Corrector Shape: %s' % str(partial_corrector.shape)
    thisInfo += 'Normalization sum: %f' % np.sum(norm)
    print(thisInfo)
    # Start with a flat estimation
    estimation = np.ones(nFOV, dtype=float)
    # pbar=tqdm(total=nIteration,bar_format='{l_bar}{bar:20}{r_bar}{bar:-10b}')
    storedReconImg = np.zeros((nIteration//scale, nFOV))
    start_time = MPI.Wtime()
else:
    estimation = np.empty(nFOV, dtype=float)


for iter in range(nIteration):
    comm.Bcast(estimation, root=0)
    # comm.Barrier()
    # thisInfo = 'Rank #%d, ' % rank
    # thisInfo += 'Estimation sum: %f' % np.sum(estimation)
    # print(thisInfo)
    fwdProj = np.matmul(matrixProc, estimation)
    quotient = fakeData/fwdProj
    correctorProc = np.matmul(quotient, matrixProc)
    if rank == 0:
        correctorSum = np.zeros(nFOV, dtype=float)
    else:
        correctorSum = None
    comm.Reduce([correctorProc, MPI.DOUBLE], [
        correctorSum, MPI.DOUBLE], op=MPI.SUM, root=0)
    # comm.Barrier()
    if rank == 0:
        estimation = estimation*correctorSum/norm
        # pbar.update(1)

        if iter % scale == 0:
            print('Iteration: #%d' % iter)
            storedReconImg[iter//scale] = estimation
    comm.Barrier()
# comm.Barrier()

if rank == 0:
    stop_time = MPI.Wtime()
    print("Script executed in %.3f"%(stop_time-start_time)+' seconds.')
    # tqdm._instances.clear()
    print('\nSaving to file: %s'%configs['out npz filename'])
    np.savez_compressed(configs['out npz filename'],imgs=storedReconImg,validIds=validIdx)
MPI.Finalize()
