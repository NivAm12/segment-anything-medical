import cv2
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from utils.plots import show_masks_on_image, show_points, show_mask
from utils.utils import iou_loss, load_mask_generator
import wandb
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils.preprocess.mask import BiggestContour
import numpy as np


def pre_us():
    sam_checkpoint = "pretrained/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = "cuda"
    contour = BiggestContour()

    image = cv2.imread(
        f'/home/projects/yonina/SAMPL_training/public_datasets/RadImageNet/radiology_ai/CT/lung/normal/lung-normal000032.png',
        cv2.IMREAD_GRAYSCALE)

    # preprocess
    contour_image = contour(image)
    masked_image = contour_image * image
    remove_text_mask = cv2.threshold(masked_image, 210, 255, cv2.THRESH_BINARY)[1]
    masked_image = cv2.inpaint(masked_image, remove_text_mask, 7, cv2.INPAINT_NS)
    masked_image = cv2.cvtColor(masked_image, cv2.COLOR_GRAY2RGB)

    # create masks
    mask_generator = load_mask_generator(sam_checkpoint, model_type, sam_model_registry,
                                         SamPredictor, device)
    mask_generator.set_image(masked_image)
    input_point = np.array([[100, 100], [20, 25], [185, 25]])
    input_label = np.array([1, 0, 0])
    # show_points(masked_image, input_point, input_label, plt.gca())

    masks, scores, logits = mask_generator.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )

    show_masks_on_image(masked_image, masks)


def test_iou():
    loss_list = []
    columns = ["image", "pred_mask", "gt_mask", "iou"]
    test_data = []

    sam_checkpoint = "pretrained/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = "cuda"
    num_examples = 100

    current_run = wandb.init(
        project="sam",
        name=f"medical_sam_with_ultrasound_preprocess)",
        config={
            "model": "sam",
            "vit": "vit_h",
            "dataset": "Dataset_BUSI_with_GT",
            "num_examples": num_examples,
            "loss": "iou"
        }
    )
    contour = BiggestContour()

    for i in tqdm(range(1, num_examples + 1)):
        image = cv2.imread(
            f'/home/projects/yonina/SAMPL_training/public_datasets/Dataset_BUSI_with_GT/malignant/malignant ({i}).png',
            cv2.IMREAD_GRAYSCALE)
        contour_image = contour(image)
        masked_image = contour_image * image

        image = cv2.cvtColor(masked_image, cv2.COLOR_GRAY2RGB)
        gt_mask = cv2.imread(
            f'/home/projects/yonina/SAMPL_training/public_datasets/Dataset_BUSI_with_GT/malignant/malignant ({i})_mask.png',
            cv2.IMREAD_GRAYSCALE)
        _, gt_mask = cv2.threshold(gt_mask, 0, 255, cv2.THRESH_BINARY)

        mask_generator = load_mask_generator(sam_checkpoint, model_type, sam_model_registry,
                                             SamAutomaticMaskGenerator, device)
        masks = mask_generator.generate(image)

        best_mask, best_iou = min(
            ((mask['segmentation'], iou_loss(mask['segmentation'], gt_mask)) for mask in masks),
            key=lambda x: x[1])

        loss_list.append(best_iou)
        test_data.append([wandb.Image(image), wandb.Image(gt_mask), wandb.Image(best_mask),
                          best_iou])

    images_table = wandb.Table(columns=columns, data=test_data)
    current_run.log({"results": images_table})

    avg_loss = sum(loss_list) / len(loss_list)
    current_run.log({"average iou": avg_loss})


if __name__ == '__main__':
    pre_us()
