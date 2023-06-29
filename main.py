import cv2
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from utils.plots import show_masks_on_image, show_coco_anns
from utils.utils import load_mask_generator, CocoDataset
import wandb
from tqdm import tqdm
from utils.preprocess.mask import BiggestContour
import random
from skimage.feature import hog
import matplotlib.pyplot as plt
import numpy as np


def segment(img_dir, masks_dir, config, project, run_name):
    # current_run = wandb.init(
    #     project=project,
    #     name=run_name,
    #     config=config
    # )
    columns = ["image", "preprocess", "predicted_no_pre", "predicted_pre"]
    table_date = []

    dataset = CocoDataset(coco_file=masks_dir,
                          image_dir=img_dir)

    mask_generator = load_mask_generator(config, sam_model_registry, SamAutomaticMaskGenerator)

    # select random image from the directory
    for _ in tqdm(range(config['num_examples'])):
        image, anns = random.choice(dataset)
        preprocess_image = preprocess(image)

        # hog
        fd, hog_image = hog(preprocess_image, orientations=9, pixels_per_cell=(8, 8),
                            cells_per_block=(1, 1), visualize=True)
        t1 = np.expand_dims(preprocess_image, axis=-1)
        t2 = np.expand_dims(hog_image, axis=-1)
        t3 = np.zeros_like(t1)
        t4 = np.concatenate([t1, t2, t3], axis=-1).astype(np.uint8)

        predicted_pre, predicted_pre_masks = generate_masks(mask_generator, t4, config)

        predicted_no_pre, predicted_no_pre_masks = generate_masks(mask_generator,
                                                                  cv2.cvtColor(image, cv2.COLOR_GRAY2RGB), config)

        plt.imshow(predicted_pre)
        plt.show()
        # wandb data
    #     images_to_display = [wandb.Image(image), wandb.Image(preprocess_image), wandb.Image(predicted_no_pre),
    #                          wandb.Image(predicted_pre)]
    #     table_date.append(images_to_display)
    #
    # images_table = wandb.Table(columns=columns, data=table_date)
    # current_run.log({"results": images_table})


def preprocess(image):
    # get the biggest contour of the image
    contour = BiggestContour()
    contour_of_the_image = contour(image)
    contour_image = contour_of_the_image * image

    # remove text from the image
    remove_text_mask = cv2.threshold(contour_image, 210, 255, cv2.THRESH_BINARY)[1]
    masked_image = cv2.inpaint(contour_image, remove_text_mask, 7, cv2.INPAINT_NS)

    return masked_image


def generate_masks(mask_generator, image, config):
    masks = mask_generator.generate(image)

    masks = [mask['segmentation'] for mask in masks]
    res = show_masks_on_image(image, masks)

    return res, masks


if __name__ == '__main__':
    project = 'sam_ultrasound'
    run_name = 'us_kidney'
    img_dir = '/home/projects/yonina/SAMPL_training/public_datasets/RadImageNet/radiology_ai/US/liver'
    masks_dir = '/home/projects/yonina/SAMPL_training/public_datasets/RadImageNet/masks/liver/masks.json'

    run_config = {
        'checkpoint': 'pretrained/sam_vit_h_4b8939.pth',
        'model_type': 'vit_h',
        'device': 'cuda',
        'num_examples': 20,
    }

    segment(img_dir, masks_dir, run_config, project, run_name)
