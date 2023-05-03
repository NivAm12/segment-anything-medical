from skimage.morphology import binary_dilation, binary_erosion
from typing import Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import scipy
import cv2 as cv



def im_show(im1, im2, title):
    """
    Display two image in a figure

    :param im1: first image
    :param im2: second image
    :param title: Window name
    @author:Amit
    """
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(im1, cmap='gray')
    axes[1].imshow(im2, cmap='gray')
    fig.canvas.manager.set_window_title(title)
    fig.suptitle(title)
    plt.show()


def im_show3(im_arr, title):
    """
    Display images 3 in a row from arbitrary number of images in array
    Used for attention visualising
    :param im_arr: array of images
    :param title: Window name
    @author:Amit
    """
    fig, axes = plt.subplots(len(im_arr)//3 if len(im_arr) % 3 == 0 else len(im_arr)//3 + 1, 3)
    count = 0
    for i in range(len(im_arr)):
        axes[count // 3][count % 3].imshow(im_arr[i])
        axes[count // 3][count % 3].set_title("Eiganvalues " + str(i + 2))
        count = count + 1
    fig.canvas.manager.set_window_title(title)
    fig.suptitle(title)
    plt.show()

def knn_affinity(image, n_neighbors=[20, 10], distance_weights=[2.0, 0.1]):
    """Computes a KNN-based affinity matrix. Note that this function requires pymatting"""
    try:
        from pymatting.util.kdtree import knn
    except:
        raise ImportError(
            'Please install pymatting to compute KNN affinity matrices:\n'
            'pip3 install pymatting'
        )

    h, w = image.shape[:2]
    r, g, b = image.reshape(-1, 3).T
    n = w * h

    x = np.tile(np.linspace(0, 1, w), h)
    y = np.repeat(np.linspace(0, 1, h), w)

    i, j = [], []

    for k, distance_weight in zip(n_neighbors, distance_weights):
        f = np.stack(
            [r, g, b, distance_weight * x, distance_weight * y],
            axis=1,
            out=np.zeros((n, 5), dtype=np.float32),
        )

        distances, neighbors = knn(f, f, k=k)

        i.append(np.repeat(np.arange(n), k))
        j.append(neighbors.flatten())

    ij = np.concatenate(i + j)
    ji = np.concatenate(j + i)
    coo_data = np.ones(2 * sum(n_neighbors) * n)

    # This is our affinity matrix
    W = scipy.sparse.csr_matrix((coo_data, (ij, ji)), (n, n))
    return W


def rw_affinity(image, sigma=0.033, radius=1):
    """Computes a random walk-based affinity matrix. Note that this function requires pymatting"""
    try:
        from pymatting.laplacian.rw_laplacian import _rw_laplacian
    except:
        raise ImportError(
            'Please install pymatting to compute RW affinity matrices:\n'
            'pip3 install pymatting'
        )
    h, w = image.shape[:2]
    n = h * w
    values, i_inds, j_inds = _rw_laplacian(image, sigma, radius)
    W = scipy.sparse.csr_matrix((values, (i_inds, j_inds)), shape=(n, n))
    return W


class CT_MASK(object):
    """
    return a mask of interest areas in a ct slice
    :param img: image
    :param show: True == show a figure with mask and original image
    :return: Binary mask
    @author:Amit
    """

    def __init__(self, show=False):
        self.show = show

    def __call__(self, img):
        # Image process
        # process = img[6:-6, 6:-6]
        process = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        process = cv.GaussianBlur(process, (5, 5), 0)

        # Finding contours
        contours, hierarchy = cv.findContours(process, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        # sort countors by size
        sorted_cnt = sorted(contours, key=cv.contourArea)

        cur_sum = 0
        cur_countor = -1
        sum = np.sum(img)

        mask = np.zeros(img.shape, np.uint8)
        while cur_sum < 0.8 * sum:
            hull = cv.convexHull(sorted_cnt[cur_countor])
            mask = cv.drawContours(mask, [hull], -1, (255, 255, 255), -1)
            cur_sum = np.sum(cv.bitwise_and(mask, img))
            cur_countor = cur_countor - 1

        # dilate both the image and the mask
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7,7))
        mask = cv.dilate(mask, kernel, iterations=10)

        mask = cv.bitwise_and(mask, img)
        if self.show:
            im_show(img, mask, "mask")

        return mask