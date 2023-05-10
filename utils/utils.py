import numpy as np


def iou_loss(pred, target):
    intersection = np.logical_and(pred, target)
    union = np.logical_or(pred, target)
    iou = np.sum(intersection) / np.sum(union)
    loss = 1 - iou

    return loss


def load_mask_generator(config, model_registry, mask_generator):
    sam = model_registry[config['model_type']](checkpoint=config['checkpoint'])
    sam.to(device=config['device'])

    mask_generator = mask_generator(sam)

    return mask_generator
