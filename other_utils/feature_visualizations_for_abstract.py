from pathlib import Path
from radutils.prototype.data.dataUtil import DataUtil
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
from radutils.prototype.data.imageProcessingUtil import (
    ImageProcessingUtil as IPU,
)
import pandas as pd
import ast
from joblib import Parallel, delayed


def overlay_heatmap(orgimg, heatmap):
    plt.figure(figsize=(10, 10))

    plt.imshow(orgimg, cmap="gray", vmin=-1024, vmax=3072)

    plt.imshow(
        heatmap,
        cmap="Spectral_r",
        interpolation=None,
        vmin=0,
        vmax=1,
        # alpha=heatmap,
    )

    plt.xticks([])
    plt.yticks([])

    return plt


def overlay_mask(orgimg, mask):
    try:
        contours = measure.find_contours(mask, 0)
    except:
        import pdb

        pdb.set_trace()

    plt.figure(figsize=(5, 5))
    plt.imshow(orgimg, cmap="gray", vmin=0, vmax=1)

    for n, contour in enumerate(contours):
        plt.plot(contour[:, 1], contour[:, 0], linewidth=3, color="g")

    plt.xticks([])
    plt.yticks([])

    return plt


def save_heatmaps(imgarr, featarr, _min, _max, casename, featname, outputfolder):
    outdir = f"{outputfolder}/{casename}/{featname}"

    DataUtil.mkdir(outdir)

    slcs = np.unique(np.nonzero(featarr)[0]).tolist()

    noslcs = imgarr.shape[0]
    interval = int(np.ceil(len(slcs) / 2))

    for i in slcs[::interval]:
        imgslc = imgarr[i]
        featslc = featarr[i]

        if _min is not None and _max is not None:
            featslc = (featslc - _min) / (_max - _min)
            featslc[featslc < 0] = 0
            featslc[featslc > 1] = 1
            featslc = np.ma.masked_where(featslc == 0, featslc)

        else:
            featslc = (featslc - featslc.min()) / (featslc.max() - featslc.min())

        plt = overlay_heatmap(imgslc, featslc)
        plt.savefig(
            f"{outdir}/slc_{i}_feat.png",
            transparent=True,
            bbox_inches="tight",
            pad_inches=0,
        )

        plt.close()

        plt.imshow(imgslc, cmap="gray", vmin=-1024, vmax=3072)
        plt.savefig(
            f"{outdir}/slc_{i}_img.png",
            transparent=True,
            bbox_inches="tight",
            pad_inches=0,
        )

        plt.close()


def process_images(sb, outputfolder, imgfolder, final_ranges):
    print(sb)

    casename, lesion = sb.stem.split("__")

    imgfile = Path(imgfolder).joinpath(casename).joinpath("ADC_reg.nii.gz")
    orgimg = sitk.ReadImage(str(imgfile))

    gabor4157path = sb.joinpath("gabor2D-frequency-4-theta-1-57").joinpath(
        "filteredimage.nii.gz"
    )

    f1path = (
        sb.joinpath("gabor2D-frequency-4-theta-1-57")
        .joinpath("firstorder")
        .joinpath("Kurtosis.nii.gz")
    )

    f1 = sitk.ReadImage(str(f1path))

    gabor4157 = sitk.ReadImage(gabor4157path)
    img = IPU.resample_image(orgimg, f1.GetSpacing())

    phstart = f1.TransformIndexToPhysicalPoint((0, 0, 0))
    phend = f1.TransformIndexToPhysicalPoint(f1.GetSize())

    start = img.TransformPhysicalPointToIndex(phstart)
    end = img.TransformPhysicalPointToIndex(phend)

    startx, starty, startz = start
    endx, endy, endz = end

    img = img[startx:endx, starty:endy, startz:endz]

    f1arr = sitk.GetArrayFromImage(f1)

    gabor4157arr = sitk.GetArrayFromImage(gabor4157)

    imgarr = sitk.GetArrayFromImage(img)

    save_heatmaps(
        imgarr,
        f1arr,
        final_ranges["f1_min"],
        final_ranges["f1_max"],
        casename,
        "gabor2D-frequency-4-theta-1-57_firstorder_kurtosis",
        outputfolder,
    )

    save_heatmaps(
        imgarr,
        gabor4157arr,
        None,
        None,
        casename,
        "gabor2D-frequency-4-theta-1-57",
        outputfolder,
    )


def get_min_max_arr(sb):
    print(sb)

    casename, lesion = sb.stem.split("__")

    f1path = (
        sb.joinpath("gabor2D-frequency-4-theta-1-57")
        .joinpath("firstorder")
        .joinpath("Kurtosis.nii.gz")
    )

    f1 = sitk.ReadImage(str(f1path))

    f1arr = sitk.GetArrayFromImage(f1)

    nonzero = f1arr[np.nonzero(f1arr)]

    return (np.min(nonzero), np.max(nonzero))


if __name__ == "__main__":
    inputfolder = "../outputs/prostate_voxel_arrays_v2/"
    outputfolder = "../outputs/prostate_feature_maps_v2/"

    imgfolder = "/Users/amogh/Projects/Data/CWRU/CCFR/1_Original_Organized"

    subdirs = DataUtil.getSubDirectories(inputfolder)

    nofeats = 1
    feat_ranges = {}

    final_ranges = {}

    final_ranges["f1_min"] = 1.3
    final_ranges["f1_max"] = 4

    # arrs = Parallel(n_jobs=-1)(delayed(get_min_max_arr)(sb) for sb in subdirs)
    # mins = [x[0] for x in arrs]
    # maxs = [x[1] for x in arrs]

    Parallel(n_jobs=-1)(
        delayed(process_images)(sb, outputfolder, imgfolder, final_ranges)
        for sb in subdirs
    )

    import pdb

    pdb.set_trace()
