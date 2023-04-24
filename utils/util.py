import numpy as np


def iou_loss(pred, target):
    intersection = np.logical_and(pred, target)
    union = np.logical_or(pred, target)
    iou = np.sum(intersection) / np.sum(union)
    loss = 1 - iou

    return loss