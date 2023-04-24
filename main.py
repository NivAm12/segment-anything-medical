import matplotlib.pyplot as plt
from transformers import pipeline
import wandb
from tqdm import tqdm
import utils.plots
from utils.util import iou_loss
from PIL import Image


def test_iou():
    generator = pipeline("mask-generation", model="facebook/sam-vit-huge", device=0)
    loss_list = []
    columns = ["image", "pred_mask", "gt_mask", "iou"]
    test_data = []
    num_examples = 100

    current_run = wandb.init(
        project="sam",
        name=f"busi_dataset_iou_hugging_face_model",
        config={
            "model": "sam",
            "vit": "vit_h",
            "dataset": "Dataset_BUSI_with_GT",
            "num_examples": num_examples,
            "loss": "iou"
        }
    )

    for i in tqdm(range(1, num_examples+1)):
        image = Image.open(
            f'/home/projects/yonina/SAMPL_training/public_datasets/Dataset_BUSI_with_GT/malignant/malignant ({i}).png')\
            .convert("RGB")
        gt_mask = Image.open(
            f'/home/projects/yonina/SAMPL_training/public_datasets/Dataset_BUSI_with_GT/malignant/malignant ({i})_mask.png')
        gt_mask = gt_mask.point(lambda p: True if p > 0 else False)

        outputs = generator(image, points_per_batch=64)
        masks = outputs["masks"]

        best_mask = None
        best_iou = 1.1

        for mask in masks:
            iou = iou_loss(mask, gt_mask)

            if iou < best_iou:
                best_mask = mask
                best_iou = iou

        loss_list.append(best_iou)
        test_data.append([wandb.Image(image), wandb.Image(gt_mask), wandb.Image(best_mask),
                          best_iou])

    images_table = wandb.Table(columns=columns, data=test_data)
    current_run.log({"results": images_table})

    avg_loss = sum(loss_list) / len(loss_list)
    current_run.log({"average iou": avg_loss})


if __name__ == '__main__':
    test_iou()
