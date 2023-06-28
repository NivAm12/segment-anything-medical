import numpy as np
import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from PIL import Image
import os
import cv2


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


class CocoDataset(Dataset):
    def __init__(self, coco_file, image_dir, transform=None):
        self.coco = COCO(coco_file)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.coco.getImgIds())

    def __getitem__(self, idx):
        # fix the coco bug that images id's starting from 1
        idx += 1
        img_id = self.coco.getImgIds(imgIds=idx)
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.image_dir, img_info['file_name'])
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if self.transform:
            image = self.transform(image)

        annotations_ids = self.coco.getAnnIds(imgIds=img_info['id'])
        annotations = self.coco.loadAnns(annotations_ids)

        return image, annotations
