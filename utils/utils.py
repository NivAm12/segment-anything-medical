import numpy as np


def iou_loss(pred, target):
    intersection = np.logical_and(pred, target)
    union = np.logical_or(pred, target)
    iou = np.sum(intersection) / np.sum(union)
    loss = 1 - iou

    return loss


def load_mask_generator(sam_checkpoint, model_type, model_registry, mask_generator, device):
    sam = model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    mask_generator = mask_generator(sam)

    return mask_generator
