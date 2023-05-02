import cv2
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from utils.util import iou_loss, load_mask_generator
import wandb
from tqdm import tqdm


def test_iou():
    loss_list = []
    columns = ["image", "pred_mask", "gt_mask", "iou"]
    test_data = []

    sam_checkpoint = "pretrained/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = "cuda"
    num_examples = 5

    current_run = wandb.init(
        project="sam",
        name=f"busi_dataset_medical_sam)",
        config={
            "model": "sam",
            "vit": "vit_h",
            "dataset": "Dataset_BUSI_with_GT",
            "num_examples": num_examples,
            "loss": "iou"
        }
    )

    for i in tqdm(range(1, num_examples + 1)):
        image = cv2.imread(
            f'/home/projects/yonina/SAMPL_training/public_datasets/Dataset_BUSI_with_GT/malignant/malignant ({i}).png',
            cv2.IMREAD_GRAYSCALE)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
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
    test_iou()
