"""
=============================
Ultrasonography Preprocessing
=============================

Created 2022/05/30
@author: Chen Solomon
@organization: Weizmann Institute of Science
"""

from copy import copy
import cv2
import numpy as np
# from utils.det_text import text_detection_east
from ..builder.transforms import TORCHVISION_TRANS


@TORCHVISION_TRANS.register_class
class USCropScan:
    """
    Crop US scan according to connected components (CCs) size
    """
    def __init__(self):
        pass

    def __call__(self, us_img):
        """
        Per
        @param us_img: image to be cropped
        @return: cropped image
        """
        # Copy image and adjust dimensionality
        us_img_cpy = copy(us_img)
        if us_img_cpy.ndim == 3:
            us_img_cpy = us_img.mean(axis=-1)
        # Get connected components (CCs) of image with statistics
        # us_cc = (
        #   num_connected_components,
        #   labeled_array,
        #   [[w_start, h_start, w_len, h_len, num_pixels] for each component]
        # )
        us_cc = cv2.connectedComponentsWithStats(us_img_cpy.astype('uint8'))
        # get indexes of sorted sizes for CCs
        us_cc_stat = us_cc[2]
        cc_idc = np.argsort(us_cc_stat[:, -1])[::-1]
        # decision rule for crop
        if np.percentile(us_img[us_cc[1] == cc_idc[0]], 99) == 0:
            # 99th percentile of biggest connected component is 0 -> cc_idc[0] is background
            mask: np.ndarray = np.equal(us_cc[1], cc_idc[1])
        elif np.percentile(us_img[us_cc[1] == cc_idc[1]], 99) == 0:
            # 99th percentile of 2nd biggest connected component is 0 -> cc_idc[0] is background
            mask: np.ndarray = np.equal(us_cc[1], cc_idc[0])
        else:
            raise NotImplementedError('No valid decision rule for cropping')
        # Crop image according to decision
        # Convert mask to bounding box
        # Get indices of non-zeroes along axes
        nnz_idc_z = np.where(mask.any(axis=1))[0]  # z (columns)
        nnz_idc_x = np.where(mask.any(axis=0))[0]  # x    (rows)
        # crop image
        us_img_cropped = us_img[
                         nnz_idc_z.min():nnz_idc_z.max() + 1,
                         nnz_idc_x.min():nnz_idc_x.max() + 1
                         ]
        return us_img_cropped


# class USRemoveTextEAST:
#     """
#     Crop US scan according to connected components (CCs) size
#     """
#     def __init__(
#             self,
#             conf_threshold=0.4,
#             nms_threshold=0.1,
#             inp_width=1024,
#             inp_height=1024,
#             radius=5,
#             ensemble_mask=(),
#             weights_dir='weights/det_text/frozen_east_text_detection.pb'
#     ):
#         self.conf_threshold = conf_threshold
#         self.nms_threshold = nms_threshold
#         self.inp_width = inp_width
#         self.inp_height = inp_height
#         self.radius = radius
#         self.ensemble_mask = ensemble_mask
#         self.weights_dir = weights_dir
#
#     def __call__(self, us_img):
#         # Create image appropriate for processing (assuming 2D image)
#         us_img_cpy = copy(us_img)
#         # Remove text (EAST DNN, implemented by LitalBy)
#         us_img_4det_text = np.transpose(  # handle dimension's order
#             np.tile(  # handle number of channels for text removal
#                 us_img_cpy,
#                 (3, 1, 1)
#             ),
#             (1, 2, 0)
#         ).astype(  # handle datatype
#             dtype=np.uint8
#         )
#         us_img_del_text = text_detection_east(
#             us_img_4det_text,
#             conf_threshold=self.conf_threshold,
#             nms_threshold=self.nms_threshold,
#             inp_width=self.inp_width,
#             inp_height=self.inp_height,
#             radius=self.radius,
#             ensemble_mask=self.ensemble_mask,
#             weights_dir=self.weights_dir,
#         ).astype(
#             dtype=us_img.dtype
#         )
#         return us_img_del_text


@TORCHVISION_TRANS.register_class
class USRemoveTextMAGNITUDE:
    """
    Remove text from US scan according pixel valuse (intensity and RGB)
    """
    def __init__(
            self,
            radius=3
    ):
        self.radius = radius

    def __call__(self, us_img):
        # # Create image appropriate for processing (assuming 2D image)
        us_img_cpy = us_img.astype(dtype=np.uint8)
        # mask according to Pixel values threshold
        mask_thr = np.equal(us_img_cpy, 255).astype(np.uint8)
        # handle RGB images
        if mask_thr.ndim == 3:
            mask_thr = mask_thr.max(axis=-1)
            # mask according to ratios between channels (R/G, G/B)
            # Assuming that the US scan takes >50% of image area
            # Currently handles bright text
            # TODO (Optional): handle dark text as well (Options: negative, Laplacian of Gaussian)
            # Compute channels ratios, assuming that in the US scan no channel is zero
            with np.errstate(divide='ignore', invalid='ignore'):  # ignore division warnings
                ratios = np.concatenate((
                        us_img_cpy[:, :, 0:1] / us_img_cpy[:, :, 1:2],
                        us_img_cpy[:, :, 1:2] / us_img_cpy[:, :, 2:3]
                    ),
                    axis=-1,
                )
            # filter out nan-s, inf-s and zeros
            ratios_indices = np.invert(
                np.isnan(ratios) +
                np.isinf(ratios) +
                (ratios == 0) > 0
            )
            ratios_indices = np.logical_and(
                ratios_indices[:, :, 0],
                ratios_indices[:, :, 1],
            )
            # compute medians per channel
            median_ratios = np.median(
                ratios[ratios_indices, :],
                axis=0,
            )[np.newaxis, np.newaxis, :]
            # compute mask according to ratios
            mask_rat = np.logical_and(
                (
                    abs(ratios - median_ratios) / median_ratios > 0.05  # pixels which deviation from median >5%
                ).sum(axis=-1) > 0,  # in one or two of the ratios
                # and their RGB values are not too low
                us_img_cpy.max(axis=-1) > np.percentile(us_img_cpy[ratios_indices], 20)
            )
            # Dilate mask to detect edges
            element = cv2.getStructuringElement(cv2.MORPH_RECT, (self.radius, self.radius))
            mask_rat = cv2.dilate(mask_rat.astype(dtype=np.uint8), element)
            # combine with threshold mask
            mask = np.logical_or(mask_thr, mask_rat).astype(dtype=np.uint8)
        else:
            mask = mask_thr
        us_img_del_text = cv2.inpaint(us_img_cpy, mask, self.radius, cv2.INPAINT_TELEA)
        us_img_del_text = us_img_del_text.astype(dtype=us_img.dtype)
        return us_img_del_text
