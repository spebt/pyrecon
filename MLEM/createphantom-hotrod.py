import numpy as np
from PIL import Image

def generate_drz_phantom():
    # drz phantom image
    numdrzImageVoxelX = 90
    numdrzImageVoxelY = 90
    numdrzImageVoxelZ = 1

    widthdrzImageVoxelX = 1
    widthdrzImageVoxelY = 1
    widthdrzImageVoxelZ = 1

    cvtrad = 180.0 / np.pi

    # hot rod phantom parameters
    R_max_drz = 50  # mm
    r_drz = [0.5, 0.75, 1, 1.25, 1.5, 1.75]
    R_min_drz = 5

    maxnumrodsection = 1000

    x_drz = np.zeros(10000)
    y_drz = np.zeros(10000)
    validFlag_drz = np.zeros(10000)

    TotalPntIdx = [0] * 6

    for SectionIdx in range(6):
        FillFlag = True
        currentPntIdx = 0
        currentGrpIdx = 0
        TotalValidPntIdx = 0
        while FillFlag:
            x_drz[SectionIdx * maxnumrodsection + TotalPntIdx[SectionIdx]] = R_min_drz + 2 * r_drz[SectionIdx] + 4 * r_drz[SectionIdx] * np.cos(30 / cvtrad) * currentGrpIdx
            y_drz[SectionIdx * maxnumrodsection + TotalPntIdx[SectionIdx]] = 4 * r_drz[SectionIdx] * np.sin(30 / cvtrad) * currentGrpIdx - 4 * r_drz[SectionIdx] * currentPntIdx
            validFlag_drz[SectionIdx * maxnumrodsection + TotalPntIdx[SectionIdx]] = 1
            if x_drz[SectionIdx * maxnumrodsection + TotalPntIdx[SectionIdx]] + r_drz[SectionIdx] > R_max_drz * np.cos(30 / cvtrad):
                validFlag_drz[SectionIdx * maxnumrodsection + TotalPntIdx[SectionIdx]] = 0
            if (y_drz[SectionIdx * maxnumrodsection + TotalPntIdx[SectionIdx]] + r_drz[SectionIdx] * np.cos(30 / cvtrad)) / (
                    x_drz[SectionIdx * maxnumrodsection + TotalPntIdx[SectionIdx]] - r_drz[SectionIdx] * np.sin(30 / cvtrad)) > np.tan(30 / cvtrad):
                validFlag_drz[SectionIdx * maxnumrodsection + TotalPntIdx[SectionIdx]] = 0
            if validFlag_drz[SectionIdx * maxnumrodsection + TotalPntIdx[SectionIdx]] == 1:
                TotalValidPntIdx += 1
            TotalPntIdx[SectionIdx] += 1
            currentPntIdx += 1
            if currentPntIdx > currentGrpIdx:
                currentPntIdx = 0
                currentGrpIdx += 1
            if R_min_drz + 4 * r_drz[SectionIdx] * currentGrpIdx > R_max_drz * np.cos(30 / cvtrad):
                FillFlag = False

        print("Total rod number of sector {}: {}".format(SectionIdx, TotalPntIdx[SectionIdx]))
        print("Radius: {}".format(r_drz[SectionIdx]))

    # numdrzImageVoxelX = int(R_max_drz * 2 / widthdrzImageVoxelX)
    # numdrzImageVoxelY = int(R_max_drz * 2 / widthdrzImageVoxelY)
    pdrzImage = np.zeros((numdrzImageVoxelY, numdrzImageVoxelX), dtype=int)

    for idxdrzImageY in range(numdrzImageVoxelY):
        for idxdrzImageX in range(numdrzImageVoxelX):
            xdrzImage = (idxdrzImageX - numdrzImageVoxelX / 2.0 + 0.5) * widthdrzImageVoxelX
            ydrzImage = (idxdrzImageY - numdrzImageVoxelY / 2.0 + 0.5) * widthdrzImageVoxelY

            for SectionIdx in range(6):
                angle = (60.0 * SectionIdx + 30) / cvtrad
                xdrzImagerotate = xdrzImage * np.cos(angle) + ydrzImage * np.sin(angle)
                ydrzImagerotate = xdrzImage * -np.sin(angle) + ydrzImage * np.cos(angle)

                for PntIdx in range(TotalPntIdx[SectionIdx]):
                    rodIdx = SectionIdx * maxnumrodsection + PntIdx
                    if validFlag_drz[rodIdx]:
                        if (xdrzImagerotate - x_drz[rodIdx]) ** 2 + (ydrzImagerotate - y_drz[rodIdx]) ** 2 < r_drz[SectionIdx] ** 2:
                            pdrzImage[idxdrzImageY, idxdrzImageX] = 1
                            break

    return pdrzImage

def save_image_to_png(image, filename):
    img = Image.fromarray((image * 255).astype(np.uint8))
    img.save(filename)

if __name__ == "__main__":
    drzImage = generate_drz_phantom()
    print(drzImage.shape)
    outFname = 'input/circle-phantom.npz'
    save_image_to_png(drzImage, "output/HotRodPhantom.png")
    np.savez(outFname,drzImage.astype(np.float32))
