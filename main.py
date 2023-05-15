import cv2
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from utils.plots import show_masks_on_image
from utils.utils import load_mask_generator
import wandb
from tqdm import tqdm
from utils.preprocess.mask import BiggestContour
import os
import random
import numpy as np


def segment(img_dir, config, project, run_name):
    # current_run = wandb.init(
    #     project=project,
    #     name=run_name,
    #     config=config
    # )
    columns = ["image", "preprocess", "predicted_no_pre", "predicted_pre"]
    table_date = []

    files_list = os.listdir(img_dir)

    # select random image from the directory
    for _ in tqdm(range(config['num_examples'])):
        file_name = random.choice(files_list)

        # create the image masks
        img_path = os.path.join(img_dir, file_name)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        preprocess_image = preprocess(image)

        predicted_pre, _ = generate_masks(preprocess_image, config)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        predicted_no_pre, _ = generate_masks(image, config)

        # wandb data
        images_to_display = [wandb.Image(image), wandb.Image(preprocess_image), wandb.Image(predicted_no_pre),
                             wandb.Image(predicted_pre)]
        table_date.append(images_to_display)

    images_table = wandb.Table(columns=columns, data=table_date)
    # current_run.log({"results": images_table})


def preprocess(image):
    # get the biggest contour of the image
    contour = BiggestContour()
    contour_of_the_image = contour(image)
    contour_image = contour_of_the_image * image

    # remove text from the image
    remove_text_mask = cv2.threshold(contour_image, 210, 255, cv2.THRESH_BINARY)[1]
    no_text_image = cv2.inpaint(contour_image, remove_text_mask, 7, cv2.INPAINT_NS)

    # fill the background of the image with the mean color of the image
    foreground_mask = cv2.threshold(no_text_image, 0.05, 255, cv2.THRESH_BINARY)[1]
    background = cv2.bitwise_not(foreground_mask)
    masked_image = cv2.inpaint(no_text_image, background, 7, cv2.INPAINT_NS)

    masked_image = cv2.cvtColor(masked_image, cv2.COLOR_GRAY2RGB)
    return masked_image


def generate_masks(image, config):
    mask_generator = load_mask_generator(config, sam_model_registry, SamAutomaticMaskGenerator)
    masks = mask_generator.generate(image)

    masks = [mask['segmentation'] for mask in masks]
    res = show_masks_on_image(image, masks)

    return res, masks


if __name__ == '__main__':
    project = 'sam_ultrasound'
    run_name = 'us_liver'
    img_dir = '/home/projects/yonina/SAMPL_training/public_datasets/RadImageNet/radiology_ai/US/liver'

    run_config = {
        'checkpoint': 'pretrained/sam_vit_h_4b8939.pth',
        'model_type': 'vit_h',
        'device': 'cuda',
        'num_examples': 20,
    }

    segment(img_dir, run_config, project, run_name)
