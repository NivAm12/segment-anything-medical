import cv2 as cv
import numpy as np


class BiggestContour(object):
    """
    Calculates the largest contour in input image and creates a binary
    mask for that contour. Tested with US sonogram

    :param process: image
    :param approx: mask edges approximation (lower == sharper approximation)
    :param show: True == show a figure with mask and original image
    :return: Binary mask
    @author:Amit
    """
    def __init__(self, approx=0.001):
        self.approx = approx

    def __call__(self, process):
        # Image process
        process = process[6:-6, 6:-6]
        process = cv.GaussianBlur(process, (5, 5), 0)
        # Finding contours
        contours, hierarchy = cv.findContours(process, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        perimeter = max(contours, key=cv.contourArea)
        peri = cv.arcLength(perimeter, True)
        perimeter = cv.approxPolyDP(perimeter, self.approx * peri, True)

        # Create mask
        mask = np.zeros(process.shape)
        cv.fillPoly(mask, [perimeter], 1)
        mask = mask.astype(bool)
        mask = np.pad(mask, ((6, 6), (6, 6)), mode='constant', constant_values=0)
        return mask


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

        return mask


if __name__ == "__main__":
    B = BiggestContour(0.001, True)
    # img = cv.imread('examples/228.png')
    img = cv.imread(
        r'Z:\Studies_under_Helsinki_approval\Emek_US_to_CT_Dr_Elik_Aharony\Patients_data\2. splitted\UTC001\all\US\IM_0060.png')
    B(img)
