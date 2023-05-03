import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from lxml import etree


def split_images(input_img):
    arr = input_img
    if arr.shape[1] % 2 != 0:
        arr = arr[:, 1:]
    edge = int(arr.shape[1] / 2)
    arr1 = arr[:, :edge]
    arr2 = arr[:, edge:]
    im1 = Image.fromarray(arr1)
    im2 = Image.fromarray(arr2)
    return im1, im2


def xml_to_masks(out_dir, images_dir, cvat_xml, scale_factor):
    labels = ["Liver", "Kidney", "Gallbladder", "Spleen", "Stomach", "Intestine", "Bone", "Blood vessels", "lung"]
    dir_create(out_dir, labels)
    img_list = [f for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f))]
    ct_dict = {}
    us_dict = {}
    mask_bitness = 24
    for img in tqdm(img_list):
        img_path = os.path.join(images_dir, img)
        anno = parse_anno_file(cvat_xml, img)
        is_first_image = True
        ct_list = []
        us_list = []
        for image in anno:
            if is_first_image:
                current_image = cv2.imread(img_path)
                height, width, _ = current_image.shape
                is_first_image = False
            for label in labels:
                background = np.zeros((height, width, 3), np.uint8)
                background = create_mask_file(width,
                                              height,
                                              mask_bitness,
                                              background,
                                              image['shapes'],
                                              scale_factor,
                                              label)
                background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
                ct, us = split_images(background)
                out_ct = os.path.join(out_dir + "\\CT\\annotations\\" + label, img.split('.')[0] + '.png')
                out_us = os.path.join(out_dir + "\\US\\annotations\\" + label, img.split('.')[0] + '.png')
                ct_list.append(ct)
                us_list.append(us)
        us_dict[img] = us_list
        ct_dict[img] = ct_list
    return ct_dict, us_dict


def parse_anno_file(cvat_xml, image_name):
    root = etree.parse(cvat_xml).getroot()
    anno = []

    image_name_attr = ".//image[@name='{}']".format(image_name)
    for image_tag in root.iterfind(image_name_attr):
        image = {}
        for key, value in image_tag.items():
            image[key] = value
        image['shapes'] = []
        for poly_tag in image_tag.iter('polygon'):
            polygon = {'type': 'polygon'}
            for key, value in poly_tag.items():
                polygon[key] = value
            image['shapes'].append(polygon)
        for box_tag in image_tag.iter('box'):
            box = {'type': 'box'}
            for key, value in box_tag.items():
                box[key] = value
            box['points'] = "{0},{1};{2},{1};{2},{3};{0},{3}".format(
                box['xtl'], box['ytl'], box['xbr'], box['ybr'])
            image['shapes'].append(box)
        image['shapes'].sort(key=lambda x: int(x.get('z_order', 0)))
        anno.append(image)
    return anno


def create_mask_file(width, height, bitness, background, shapes, scale_factor, label):
    mask = np.full((height, width, bitness // 8), background, dtype=np.uint8)
    for shape in shapes:
        if shape['label'] == label:
            points = [tuple(map(float, p.split(','))) for p in shape['points'].split(';')]
            points = np.array([(int(p[0]), int(p[1])) for p in points])
            points = points*scale_factor
            points = points.astype(int)
            mask = cv2.drawContours(mask, [points], -1, color=(255, 255, 255))
            mask = cv2.fillPoly(mask, [points], color=(255, 255, 255))
    return mask


def dir_create(path, labels):
    if not os.path.exists(path):
        for label in labels:
            ct_path = os.path.join(path + "\\CT\\annotations", label)
            us_path = os.path.join(path + "\\US\\annotations", label)
            os.mkdir(ct_path)
            os.mkdir(us_path)


def create_stacked_image(in_path, out_path, images_dir, xml_path):
    # labels = ["Liver", "Kidney", "Gallbladder", "Spleen", "Stomach", "Intestine", "Bone", "Blood vessels", "lung"]
    ct_dict, us_dict = xml_to_masks(out_path, images_dir, xml_path, scale_factor=1)
    for us_image in us_dict.keys():
        us = cv2.imread(os.path.join(in_path, "US\\images\\" + us_image), cv2.IMREAD_GRAYSCALE)
        for seg_image in us_dict[us_image]:
            us = np.dstack((us, seg_image))
        np.save(out_path + "\\US\\" + us_image[:-3] + "npy", us)
    for ct_image in ct_dict.keys():
        ct = cv2.imread(os.path.join(in_path, "CT\\images\\" + ct_image), cv2.IMREAD_GRAYSCALE)
        for seg_image in ct_dict[ct_image]:
            ct = np.dstack((ct, seg_image))
        np.save(out_path + "\\CT\\" + ct_image[:-3] + "npy", ct)


input_dir = r"Z:\yonina\Studies_under_Helsinki_approval\Emek_US_to_CT_Dr_Elik_Aharony\Patients_data\annotations"
images_dir = r"Z:\yonina\Studies_under_Helsinki_approval\Emek_US_to_CT_Dr_Elik_Aharony\Patients_data\annotations\images"
outdir = r"Z:\yonina\Studies_under_Helsinki_approval\Emek_US_to_CT_Dr_Elik_Aharony\Patients_data\annotation_images"
xml_path = r"Z:\yonina\Studies_under_Helsinki_approval\Emek_US_to_CT_Dr_Elik_Aharony\Patients_data\annotations\annotations.xml"
create_stacked_image(input_dir, outdir, images_dir, xml_path)
