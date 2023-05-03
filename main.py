import cv2
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from utils.utils import iou_loss, load_mask_generator
import wandb
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils.preprocess.mask import BiggestContour


def pre_process():
    # Load the ultrasound image
    image_path = f'/home/projects/yonina/SAMPL_training/public_datasets/RadImageNet/radiology_ai/US/liver/usn053022.png'

    image = cv2.imread(image_path)

    # # Threshold the image to create a binary mask
    # _, binary_mask = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #
    # # Find contours in the binary mask
    # contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    #
    # # Find the contour with the largest area
    # largest_contour = max(contours, key=cv2.contourArea)
    #
    # # Get the bounding box coordinates of the contour
    # x, y, w, h = cv2.boundingRect(largest_contour)
    #
    # # Crop the image using the bounding box coordinates
    # cropped_image = image[y:y + h, x:x + w]
    contour = BiggestContour()
    ct_image = contour(image)

    fig, ax = plt.subplots(1, 1)
    ax[0].imshow(image, cmap='gray')
    # ax[1].imshow(cropped_image, cmap='gray')
    # ax[2].imshow(ct_image, cmap='gray')
    plt.show()


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
    pre_process()
    # test_iou()
