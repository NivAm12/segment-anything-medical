import cv2
import numpy as np
import torch
from skimage.measure import LineModelND, ransac
from sympy.geometry import Line, Point, intersection


def to_grayscale_numpy(image):
    if isinstance(image, torch.Tensor):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        return image


def get_leftmost_and_rightmost_non_zero(image):
    boolean = image != 0.0
    flipped = np.fliplr(boolean)
    left = boolean.cumsum(axis=1).cumsum(axis=1) == 1
    right = np.fliplr(flipped.cumsum(axis=1).cumsum(axis=1) == 1)
    return np.logical_or(left, right).astype(float)


def delete_bottom(image):
    bottom_left_col = np.nonzero(image.sum(axis=0))[0].min()
    bottom_left_row = np.nonzero(image[:, bottom_left_col])[0].max()
    bottom_right_col = np.nonzero(image.sum(axis=0))[0].max()
    bottom_right_row = np.nonzero(image[:, bottom_right_col])[0].max()
    image[min(bottom_right_row, bottom_left_row):, :] = False
    return image


def split_left_and_right(image):
    right_image = image.copy()
    left_image = image.copy()
    right_image[:, :right_image.shape[1] // 2] = False
    left_image[:, left_image.shape[1] // 2:] = False
    return right_image, left_image


def match_line_to_points(image):
    data = np.column_stack(np.nonzero(image)[::-1])
    model, _ = ransac(data, LineModelND, min_samples=2,
                      residual_threshold=1, max_trials=1000)
    x = np.nonzero(image)[1]
    y = model.predict_y(x)
    mse = (np.square(y - np.nonzero(image)[0])).mean()
    return x, y, mse


def draw_lines_from_xy(xr, yr, mse_r, xl, yl, mse_l, image_shape):
    empty = np.zeros(image_shape)
    if mse_l > mse_r:
        start_point = xr.min(), int(yr[np.argmin(xr)])
        end_point = xr.max(), int(yr[np.argmax(xr)])
        image = cv2.line(empty, start_point, end_point, 1, 5)
        mirror_start = image_shape[1] - xr.min(), int(yr[np.argmin(xr)])
        mirror_end = image_shape[1] - xr.max(), int(yr[np.argmax(xr)])
        image = cv2.line(image, mirror_start, mirror_end, 1, 5)

    else:
        start_point = xl.min(), int(yl[np.argmin(xl)])
        end_point = xl.max(), int(yl[np.argmax(xl)])
        image = cv2.line(empty, start_point, end_point, 1, 5)
        mirror_start = image_shape[1] - xl.min(), int(yl[np.argmin(xl)])
        mirror_end = image_shape[1] - xl.max(), int(yl[np.argmax(xl)])
        image = cv2.line(image, mirror_start, mirror_end, 1, 5)
    return image, start_point, end_point, mirror_start, mirror_end


def find_circle_center(start_point, end_point, mirror_start, mirror_end):
    """
    finds the probe's origin by calculating the left and right outlines' intersection
    :return: center: SymPy Point
             center_x: x coordinate of center
             center_y: y coordinate of center
    """
    line1 = Line(start_point, end_point)
    line2 = Line(mirror_start, mirror_end)

    # find the intersection point
    intersection_point = intersection(line1, line2)
    center_x = intersection_point[0][0]
    center_y = intersection_point[0][1]

    center = Point(center_x, center_y)
    assert center.distance(Point(start_point)) == center.distance(Point(mirror_start))
    return center, center_x, center_y, line1, line2


def get_circle_params(center, center_x, center_y, start_point, end_point, line1, line2):
    """
    calculates parameters for cv2.ellipse to draw the top and bottom of the us mask
    """
    short_radius = int(center.distance(Point(start_point)))
    long_radius = int(center.distance(Point(end_point)))
    center = (int(center_x), int(center_y))
    short_axes = (short_radius, short_radius)
    long_axes = (long_radius, long_radius)
    alpha = line1.angle_between(line2) / 2
    start_angle = 270 - np.degrees(float(alpha))
    end_angle = 270 + np.degrees(float(alpha))
    return center, short_axes, long_axes, start_angle, end_angle


def draw_sectors(image, center, short_axes, long_axes, start_angle, end_angle):
    image = cv2.ellipse(image, center, short_axes, 180, start_angle, end_angle, 1, 5)
    image = cv2.ellipse(image, center, long_axes, 180, start_angle, end_angle, 1, 5)
    return image


def fill_mask(image):
    formatted_image = np.copy(image).astype(np.uint8) * 255
    th, im_th = cv2.threshold(formatted_image, 220, 255, cv2.THRESH_BINARY_INV)
    im_floodfill = im_th.copy()
    h, w = im_th.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    filled_mask = cv2.floodFill(im_floodfill, mask, (0, 0), 0)
    return filled_mask[1] / 255


def get_us_mask(image):
    gray = to_grayscale_numpy(image)
    left_and_right = get_leftmost_and_rightmost_non_zero(gray)
    left_and_right = delete_bottom(left_and_right)
    right_image, left_image = split_left_and_right(left_and_right)

    xr, yr, mse_r = match_line_to_points(right_image)
    xl, yl, mse_l = match_line_to_points(right_image)
    image, start_point, end_point, mirror_start, mirror_end = \
        draw_lines_from_xy(xr, yr, mse_r, xl, yl, mse_l, gray.shape)

    center, center_x, center_y, line1, line2 = find_circle_center(start_point, end_point, mirror_start, mirror_end)
    center, short_axes, long_axes, start_angle, end_angle = \
        get_circle_params(center, center_x, center_y, start_point, end_point, line1, line2)

    image = draw_sectors(image, center, short_axes, long_axes, start_angle, end_angle)
    return fill_mask(image)
