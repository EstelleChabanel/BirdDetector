import os
import glob
import pandas as pd
import shutil
from PIL import Image
import shutil
import yaml
import albumentations as A
import cv2
import pybboxes as pbx

from reformatting_utils import load_config, extract_dataset_config

CLASSES = ['bird']

def get_inp_data(img_file, current_folder):
    """
    Get input data for image processing.
    Args:
        img_file (str): Name of the input image file.
    Returns:
        tuple: A tuple containing the image, ground truth bounding boxes, and augmented file name.
    """
    file_name = os.path.splitext(img_file)[0]
    # Get image
    image = cv2.imread(os.path.join(current_folder, "images", img_file))

    # Get label and transformed into albumentations format
    lab_pth = os.path.join(current_folder, "labels", f"{file_name}.txt")
    yolo_str_labels = open(lab_pth, "r").read()
    if not yolo_str_labels:
        print("No object")
        gt_bboxes = []
    else:
        lines = [line.strip() for line in yolo_str_labels.split("\n") if line.strip()]
        gt_bboxes = []
        for yolo_str_label in lines:
            if yolo_str_label:
                str_bbox_list = yolo_str_label.split()
                class_number = int(str_bbox_list[0])
                class_name = CLASSES[class_number]
                bbox_values = list(map(float, str_bbox_list[1:]))
                album_bb_list = bbox_values + [class_name]
                gt_bboxes.append(album_bb_list)

    return image, gt_bboxes


def draw_yolo(image, labels, file_name):
    """
    Draw bounding boxes on an image based on YOLO format.

    Args:
        image (numpy.ndarray): Input image.
        labels (list): List of bounding boxes in YOLO format.

    """
    H, W = image.shape[:2]
    for label in labels:
        yolo_normalized = label[1:]
        box_voc = pbx.convert_bbox(tuple(yolo_normalized), from_type="yolo", to_type="voc", image_size=(W, H))
        cv2.rectangle(image, (box_voc[0], box_voc[1]),
                      (box_voc[2], box_voc[3]), (0, 0, 255), 1)
    cv2.imwrite(f"bb_image/{file_name}.jpg", image)
    # cv2.imshow(f"{file_name}.jpg", image)
    # cv2.waitKey(0)

def single_obj_bb_yolo_conversion(transformed_bboxes, class_names):
    """
    Convert bounding boxes for a single object to YOLO format.

    Parameters:
    - transformed_bboxes (list): Bounding box coordinates and class name.
    - class_names (list): List of class names.

    Returns:
    - list: Bounding box coordinates in YOLO format.
    """
    if transformed_bboxes:
        class_num = class_names.index(transformed_bboxes[-1])
        bboxes = list(transformed_bboxes)[:-1]
        bboxes.insert(0, class_num)
    else:
        bboxes = []
    return bboxes


def multi_obj_bb_yolo_conversion(aug_labs, class_names):
    """
    Convert bounding boxes for multiple objects to YOLO format.

    Parameters:
    - aug_labs (list): List of bounding box coordinates and class names.
    - class_names (list): List of class names.

    Returns:
    - list: List of bounding box coordinates in YOLO format for each object.
    """
    yolo_labels = [single_obj_bb_yolo_conversion(aug_lab, class_names) for aug_lab in aug_labs]
    return yolo_labels


source_folder = r'/gpfs/gibbs/project/jetz/eec42/data/formatted_data_no_background_augm'
#saving_folder = r'/gpfs/gibbs/project/jetz/eec42/data/baseline1_pfeifer_newmexico_no_background'

datasets =['global_birds_newmexico']

for subdataset in datasets:
    dataset_folder = os.path.join(source_folder, subdataset)
    available_img = os.listdir(os.path.join(dataset_folder, "images"))

    for img_name in available_img:

        image, gt_bboxes = get_inp_data(img_name, dataset_folder)
        transform = A.Compose([
            #A.RandomCrop(width=300, height=300),
            A.HorizontalFlip(p=0.8),
            A.VerticalFlip(p=0.8),
            A.GaussNoise (var_limit=(10.0, 50.0), mean=0, per_channel=True, always_apply=False, p=0.3),
            A.RandomBrightness(limit=0.2, always_apply=False, p=0.5),
            #A.CLAHE(clip_limit=(0, 1), tile_grid_size=(8, 8), always_apply=True),
            #A.Resize(300, 300)
            ], bbox_params=A.BboxParams(format='yolo'))
        
        # Apply the augmentations
        transformed = transform(image=image, bboxes=gt_bboxes)
        transformed_image, transformed_bboxes = transformed['image'], transformed['bboxes']

        tot_objs = len(transformed_bboxes)
        if tot_objs:
            # Convert bounding boxes to YOLO format
            trans_bboxes = multi_obj_bb_yolo_conversion(transformed_bboxes, CLASSES) if tot_objs > 1 else [single_obj_bb_yolo_conversion(transformed_bboxes[0], CLASSES)]
            #if not any(element < 0 for row in transformed_bboxes for element in row):
            # Save augmented label and image
            lab_out_pth = os.path.join(dataset_folder, "labels", img_name.split('.jpg')[0] + '_aug_.txt')
            with open(lab_out_pth, 'w') as output:
                for bbox in trans_bboxes:
                    #updated_bbox = str(bbox).replace(',', ' ').replace('[', '').replace(']', '')
                    line = '\t'.join(map(str, bbox))
                    output.write(line + '\n')

            out_img_path = os.path.join(dataset_folder, "images", img_name.split('.jpg')[0] + '_aug_.jpg')
            cv2.imwrite(out_img_path, transformed_image)
            # Draw bounding boxes on the augmented image
            draw_yolo(transformed_image, trans_bboxes, img_name.split('.jpg')[0] + '_aug_.jpg')
            #else:
             #   print("Found Negative element in Transformed Bounding Box...")
        else:
            print("Label file is empty")


        