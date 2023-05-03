import os
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import cv2
import random
from numba import njit
from tqdm import tqdm

@njit()
def apply_seg_map(img, seg, color_map,alpha):
    """
    Overlay segmentation map onto an image, the function is jited for performance.
    @param img: input image as numpy array
    @param seg: input segmentation map as a numpy array
    @param color_map: color map as a numpy array
    @param alpha: The opacity of the segmentation overlay, 0==transparent, 1==only segmentation map
    @return: segmented image as a numpy array
    """
    height, width = seg.shape
    out = np.zeros((height, width, 3))
    for i in range(height):
        for j in range(width):
            if int(seg[i][j]) != 0:
                out[i][j] = (color_map[int(seg[i][j]) - 1] * alpha) + (img[i][j] * (1 - alpha))
            else:
                out[i][j] = img[i][j]
    return out.astype(np.int32)


def show_rand_slice(path, axis, rotate=-1, seg=None, alpha=0.5, color_map=np.asarray([[0, 255, 0], [255, 0, 0]])):
    """
    Function to display a slice from a volume nifty file, will present segmentation map overly if segmentation
    map address was passed
    @param path:  Path to volume .nii file
    @param axis:  Axis to pick the slice from
    @param rotate: rotate the output image. 1 = 90 degree right, 2=180 right etc.
    @param seg: path to segmentation map .nii file
    @param alpha: The opacity of the segmentation overlay, 0==transparent, 1==only segmentation map
    @param color_map: **numpy array** colors to map the segmented parts, 1 in the segment map == first item in the array
    """
    ct_img = nib.load(path)
    ct_numpy = ct_img.get_fdata()

    if seg:
        seg_img = nib.load(seg)
        seg_numpy = seg_img.get_fdata()

    # save Axial view
    if axis == "axial":
        slice = ct_numpy.shape[2] // 2 + random.randrange(-20, 20)
        ct_numpy = ct_numpy[:, :, slice]
        if seg:
            seg_numpy = seg_numpy[:, :, slice]
    # save cornal view
    if axis == "cornal":
        slice = ct_numpy.shape[1] // 2 + random.randrange(-20, 20)
        ct_numpy = ct_numpy[slice, :, :]
        if seg:
            seg_numpy = seg_numpy[slice, :, :]
    # save sagittal view
    if axis == "sagittal":
        slice = ct_numpy.shape[0] // 2 + random.randrange(-20, 20)
        ct_numpy = ct_numpy[:, slice, :]
        if seg:
            seg_numpy = seg_numpy[:, slice, :]

    if seg:
        ct_numpy = cv2.cvtColor(np.rot90(ct_numpy, rotate).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        ct_numpy = apply_seg_map(ct_numpy, np.rot90(seg_numpy, rotate), color_map, alpha)

    plt.figure()
    plt.imshow(ct_numpy, origin="lower")
    plt.show()


def nii_to_png(directory, rotate=0, skip=1, axial=True, cornal=False, sagittal=False, seg_dir=None, opacity=0.5,
               color_map=np.asarray([[0, 255, 0], [255, 0, 0]]), only_seg=True):
    """
    Takes .nii files as input and disassembles it to 3 axis, (axial, cornal, sagittal). Save the output slices
    in png files, open new folder in the same directory. If segmentation map directory is passed, and the segmentation
    maps with the same names as the corresponding volumes will overlay the segmentation maps on the volume and output
    png files in seperated folder.

    @param directory: Directory of a folder that contain the .nii volume files
    @param rotate: rotate the output image. 1 = 90 degree right, 2=180 right
    @param skip: Save every nth slice of the volume
    @param axial: True to save the axial axis, false to skip
    @param cornal: True to save the cornal axis, false to skip
    @param sagittal: True to save the sagittal axis, false to skip
    @param seg_dir: Directory of a folder that contain the .nii segmentation files (needs to be with the same name
    to the corresponding volume files), if no address it will not save segmentation map.
    @param opacity: The opacity of the segmentation overlay, 0==transparent, 1==only segmentation map
    @param color_map: **numpy array** colors to map the segmented parts, 1 in the segment map == first item in the array
    @param only_seg: If true will only save slices that have been segmented.
    """
    for filename in tqdm(os.listdir(directory)):
        ct_path = r'C:\Users\AMITAF\Desktop\ct_png'
        new_filename = "".join(filename.split(".")[:-1])
        # check if folder exist
        if not os.path.exists(ct_path):
            os.makedirs(ct_path)
        # check if file is nii format file
        try:
            ct_img = nib.load(os.path.join(directory, filename))
            if seg_dir:
                seg_img = nib.load(os.path.join(seg_dir, "label" + filename[3:]))
                if not os.path.exists(ct_path + "_seg"):
                    os.makedirs(ct_path + "_seg")
        except:
            print("not nii image skiping")
            break
        # nii to numpy
        ct_numpy = ct_img.get_fdata()
        seg_numpy = seg_img.get_fdata()

        # save Axial view
        if axial:
            total_slices = ct_numpy.shape[2]
            for cur_slice in range(0, total_slices, skip):
                shape = max(ct_numpy[:, :, cur_slice].shape)
                ct_out = cv2.resize(np.rot90(ct_numpy[:, :, cur_slice], rotate), dsize=(shape, shape),
                                    interpolation=cv2.INTER_CUBIC)

                if seg_dir:
                    if np.sum(seg_numpy[:, :, cur_slice])  < 10 and only_seg:

                        seg_out = cv2.resize(np.rot90(seg_numpy[:, :, cur_slice], rotate), dsize=(shape, shape),
                                            interpolation=cv2.INTER_CUBIC)
                        seg_out = apply_seg_map(ct_out, seg_out, color_map, opacity)
                        cv2.imwrite(
                            os.path.join(ct_path + "_seg", (new_filename + "_axial_" + str(cur_slice) + "_2" + ".png")),
                            seg_out)

                cv2.imwrite(os.path.join(ct_path, (new_filename + "_axial_" + str(cur_slice) + "_1" + ".png")), ct_out)

        # save cornal view
        if cornal:
            total_slices = ct_numpy.shape[1]
            for cur_slice in range(0, total_slices, skip):
                shape = max(ct_numpy[:, cur_slice, :].shape)
                ct_out = cv2.resize(np.rot90(ct_numpy[:, cur_slice, :], rotate), dsize=(shape, shape),
                                    interpolation=cv2.INTER_CUBIC)

                if seg_dir:
                    if np.sum(seg_numpy[:, cur_slice, :]) < 10 and only_seg:
                        continue
                    seg_out = cv2.resize(np.rot90(seg_numpy[:, cur_slice, :], rotate), dsize=(shape, shape),
                                         interpolation=cv2.INTER_CUBIC)
                    seg_out = apply_seg_map(ct_out, seg_out, color_map, opacity)
                    cv2.imwrite(
                        os.path.join(ct_path + "_seg", (new_filename + "_cornal_" + str(cur_slice) + "_2" + ".png")),
                        seg_out)

                cv2.imwrite(os.path.join(ct_path, (new_filename + "_cornal_" + str(cur_slice) + "_1" + ".png")), ct_out)

        # save sagittal view
        if sagittal:
            total_slices = ct_numpy.shape[0]
            for cur_slice in range(0, total_slices, skip):
                shape = max(ct_numpy[cur_slice, :, :].shape)
                ct_out = cv2.resize(np.rot90(ct_numpy[cur_slice, :, :], rotate), dsize=(shape, shape),
                                    interpolation=cv2.INTER_CUBIC)

                if seg_dir:
                    if np.sum(seg_numpy[cur_slice, :, :]) < 10 and only_seg:
                        continue
                    seg_out = cv2.resize(np.rot90(seg_numpy[cur_slice, :, :], rotate), dsize=(shape, shape),
                                         interpolation=cv2.INTER_CUBIC)
                    seg_out = apply_seg_map(ct_out, seg_out, color_map, opacity)
                    cv2.imwrite(
                        os.path.join(ct_path + "_seg", (new_filename + "_sagittal_" + str(cur_slice) + "_2" + ".png")),
                        seg_out)

                cv2.imwrite(os.path.join(ct_path, (new_filename + "_sagittal_" + str(cur_slice) + "_1" + ".png")),
                            ct_out)


if __name__ == "__main__":

    color_map = np.asarray([[255,0,0],[0,255,0],[0,0,255],[127,0,255],[255,51,255],[0,255,0],[0,0,255],[255,255,51],
                 [0,102,204],[153,153,255],[204,0,102],[255,102,102],[102,102,255]])
    nii_to_png(directory=r"Y:\amitaf\datasets\CT\Abdomen\BTCV\RegData\Training-Training\img"
               , seg_dir=r"Y:\amitaf\datasets\CT\Abdomen\BTCV\RegData\Training-Training\label",
               color_map=color_map)
    # show_rand_slice(r"Y:\amitaf\datasets\CT\Abdomen\BTCV\RegData\Training-Training\img\img0001-0002.nii.gz", axis="axial", seg=r"Y:\amitaf\datasets\CT\Abdomen\BTCV\RegData\Training-Training\label\label0001-0002.nii.gz",
    #                 color_map=color_map)
