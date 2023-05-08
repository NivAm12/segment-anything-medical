"""
Mammography Pre-processing
"""
from sys import platform
import os
import matplotlib.pyplot as plt
import matplotlib
from pydicom import dcmread
import numpy as np
import cv2
import torch
import warnings
from copy import copy
#
# try:
#     from Dataset import MuMoQ1PublicDataset, MuMoQ1PilotDataset
# except ModuleNotFoundError:
#     from util.datasets import MuMoQ1PublicDataset, MuMoQ1PilotDataset

matplotlib.use("TkAgg")


class MGPreprocessing:
    """
    Pre-processing for mammography scans
    """
    def __init__(self, image):
        self.image = image

    def mg_text_removal(self):
        """
        Text removal by using connected components
        """
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(self.image.astype('uint8'))
        stats = np.append(stats, np.arange(0, num_labels)[:, None], axis=1)  # insert a column of labels
        stats = stats[stats[:, 4] > 200]  # remove labels that were not detected correctly
        stats_area = stats[:, 4]
        idx = np.argsort(stats_area)[:-2]  # find indices of sorted stats_area and remove the background and the breast
        clean_image = copy(self.image)
        flag = 0
        for i in idx:  # loop over the suspected text objects
            # check if the connected component is rectangle by checking the top-left, top-right, bottom-left and
            # bottom-right corners of the connected components
            if labels[stats[i, 1], stats[i, 0]] == stats[i, 5] and \
                    labels[stats[i, 1], stats[i, 0] + stats[i, 2] - 1] == stats[i, 5] and \
                    labels[stats[i, 1] + stats[i, 3] - 1, stats[i, 0]] == stats[i, 5] and \
                    labels[stats[i, 1] + stats[i, 3] - 1, stats[i, 0] + stats[i, 2] - 1] == stats[i, 5]:
                # Set the text area to be the same as the value of the background:
                clean_image[stats[i, 1]:(stats[i, 1] + stats[i, 3]), stats[i, 0]:(stats[i, 0] + stats[i, 2])] = \
                    stats[0, 0]
                flag += 1
        if flag > 1:
            warnings.warn('MG text removal algorithm has detected a wrong object as text and removed it')
        return clean_image

    def clahe(self):
        clahe = cv2.createCLAHE(clipLimit=15, tileGridSize=(16, 16))
        image_255 = self.image * 255 / self.image.max()
        output = clahe.apply(image_255.astype('uint8'))

        # histogram:
        histogram, bin_edges = np.histogram(output, bins=256, range=(int(output.min()), int(output.max())))
        plt.plot(bin_edges[:-1], histogram[:])

        plt.imshow(output, cmap='gray')
        return output


def mg_remove_edges(image):
    """
    Removes black edges of mammography image
    @param image: image to be cropped
    @return:
    """
    if torch.is_tensor(image):
        image = image.numpy()
        if image.ndim == 3:
            image = image.squeeze()
    # In case the background is not black, change it to zero:
    histogram, bin_edges = np.histogram(image, bins=256, range=(np.min(image), np.max(image)))
    max_idx = np.where(histogram == np.max(histogram))[0][0]
    threshold = 0
    if max_idx != 0:
        # Setting a threshold so that black range pixels will be equal to zero
        threshold = bin_edges[max_idx] + (np.max(bin_edges) - np.min(bin_edges)) / 15
        image[image < threshold] = 0
    # Remove black edges of the image:
    sum_x = np.sum(image, axis=0)  # sum of each column
    if np.all(image == 0):
        return image
    y_min = np.nonzero(sum_x)[0][0]
    y_max = np.nonzero(sum_x)[0][-1]
    sum_y = np.sum(image, axis=1)  # sum of each row
    x_min = np.nonzero(sum_y)[0][0]
    x_max = np.nonzero(sum_y)[0][-1]
    cropped_image = image[x_min:x_max, y_min:y_max]
    # cropped_image = torch.Tensor(cropped_image).unsqueeze(dim=0)  # TODO: delete
    padded_img = padding_image(cropped_image)
    padded_img = torch.Tensor(padded_img).unsqueeze(dim=0)
    return padded_img


def padding_image(image):
    """
    # transforming the image to square (by padding with zeros)
    """
    # transforming the image to square (by padding with zeros):
    left = image[:, 0]  # first column of cropped image
    right = image[:, -1]  # last column of cropped image
    padding_dim = np.absolute(image.shape[0] - image.shape[1])
    if image.shape[0] > image.shape[1]:  # if rows > columns
        padding = np.zeros((image.shape[0], padding_dim), dtype=int)
        if np.size(np.nonzero(left)) > np.size(np.nonzero(right)):  # breast is located in the left side
            padded_image = np.append(image, padding, axis=1)  # zero padding in the right side
        else:  # breast is located in the right side
            padded_image = np.append(padding, image, axis=1)  # zero padding in the left side
    elif image.shape[0] < image.shape[1]:  # if rows < columns
        padding = np.zeros((padding_dim, image.shape[1]), dtype=int)
        padded_image = np.append(image, padding, axis=0)
    else:
        padded_image = image
    return padded_image


if __name__ == '__main__':
    # ===================
    # Load Pilot Dataset:
    # ===================
    if platform == "linux" or platform == "linux2":
        # linux
        # Assumes you work on WEXAC
        root_dir = os.path.join(
            os.sep +
            'home',
            'hsd',
            'yonina',
        )
    elif platform == "win32":
        # Windows
        # Assumes network drive nih-elda-var (\\isi.bigdata.weizmann.ac.il\nih) is mapped to Z:\
        root_dir = 'Z:' + os.sep
    else:
        raise OSError('unsupported operating system')
    dat_dir = os.path.join(
        root_dir,
        'Studies_under_Helsinki_approval',
        'Beilinson_Multi_Modality_Dr_Ahuva_Grubstein'
    )
    file_dir = os.path.join(
        'PILOT',
        'AHUVA_DOK_20220623',
        '1205000940093',
        'DICOM',
        'I2'
    )
    mg = dcmread(os.path.join(dat_dir, file_dir)).pixel_array.astype(dtype=np.float32)

    # ==============
    # Removing Text:
    # ==============
    mg_without_text = MGPreprocessing(mg).mg_text_removal()
    plt.subplot(1, 2, 1)
    plt.imshow(mg, cmap='gray')
    plt.subplot(1, 2, 2)
    plt.imshow(mg_without_text, cmap='gray')
    plt.show()

    # ======
    # CLAHE:
    # ======
    output = MGPreprocessing(mg).clahe()