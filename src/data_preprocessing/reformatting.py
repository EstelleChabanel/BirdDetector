import os
import json
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import math
import yaml

from reformatting_utils import load_config, extract_dataset_config, preview_few_images
from windowCropping import WindowCropper


# ======= PARAMETERS =======

ORIGINAL_DATASET_FOLDER = r'/gpfs/gibbs/project/jetz/eec42/data/original'
SAVING_DATASET_FOLDER = r'/gpfs/gibbs/project/jetz/eec42/data/formatted_data'
if not os.path.exists(os.path.join(SAVING_DATASET_FOLDER)):
    os.mkdir(os.path.join(SAVING_DATASET_FOLDER))

YAML_PATH = r'/home/eec42/BirdDetector/src/data_preprocessing/source_datasets_config.yaml'
config = load_config(YAML_PATH)


# ======= FUNCTIONS =======

def retrieve_detections_from_csv(current_folder):
    '''
    retrieve all bounding boxes and labels annotations from csv files given in a dataset
    Args:
        - current_folder (str): path to the current images and labels folder
    Returns: 
        df (pd.DataFrame) with all images annotations
    '''
    csv_files = glob.glob(current_folder + '/**/*.csv', recursive=True) # should be 1 or 2 max (train+test or all together)
    df = pd.DataFrame()
    for annotation_file in tqdm(csv_files):
        df_ = pd.read_csv(annotation_file)
        df = pd.concat([df, df_])
    return df


def retrieve_img_list(current_folder, config):
    '''
    Retrieve all images in a dataset
    Args: 
        - current_folder (str): path to the current images and labels folder
        - config (dict): dictionnary with config parameters of the current dataset
    Returns: 
        - available_img (list): list of path to available images in current dataset
    '''
    available_img = [] 
    for subdataset in config['image_path']:
        source_subdataset_folder = os.path.join(current_folder, subdataset)
        available_img.extend([os.path.join(source_subdataset_folder, fn) for fn in os.listdir(source_subdataset_folder) if fn.endswith(config["image_extension"])])
    return available_img 


def get_cropping_parameters(img_w, img_h):
    '''
    define patch size from original image size: closest size multiple of 32,
    return dictionnary with new size, corresponding stride and overlap of patches
    '''

    new_img_size = 0
    if (img_w>=640) and (img_h>=640):
        new_img_size = 640
    else:
        new_img_size = (min(img_w, img_h)//32)*32

    if img_w==new_img_size:
        overlap_w = 0
    else:
        overlap_w = (new_img_size*math.ceil(img_w/new_img_size) - img_w)/(math.ceil(img_w/new_img_size)-1)
    if img_h==new_img_size:
        overlap_h = 0
    else:
        overlap_h = (new_img_size*math.ceil(img_h/new_img_size) - img_h)/(math.ceil(img_h/new_img_size)-1)

    stride_w = new_img_size - overlap_w
    stride_h = new_img_size - overlap_h
    cropping_param = {'new_size': new_img_size, 'overlaps': [overlap_w, overlap_h], 'strides': [stride_w, stride_h]}

    return cropping_param


# ======= IMAGES SELECTION =======

# Dictionnary to store all classes and corresponding int id
category_name_to_id = {}

for dataset in config.keys():

    # Extract specific dataset config
    dataset_config = extract_dataset_config(config, dataset)
    print(dataset_config)
    dataset_name = dataset_config["name"]
    # Create metadata dictionnary to store information on current dataset
    meta = {'dataset': dataset, 'name': dataset_name}

    # Source dataset folder
    source_dataset_folder = os.path.join(ORIGINAL_DATASET_FOLDER, dataset_name)
    if not os.path.exists(source_dataset_folder):
        print("Error: can't find the source dataset at following path", source_dataset_folder, "\n going to the next dataset")
        continue

    # Create new folder to store current dataset
    dataset_folder = os.path.join(SAVING_DATASET_FOLDER, dataset_name)
    saving_img_folder = os.path.join(dataset_folder, "images")
    saving_label_folder = os.path.join(dataset_folder, "labels")
    if not os.path.exists(dataset_folder):
        os.makedirs(saving_img_folder)
        os.makedirs(saving_label_folder)


    # Retrieve all csv annotations files
    df = retrieve_detections_from_csv(source_dataset_folder)

    # Retrieve list of images
    available_img = retrieve_img_list(source_dataset_folder, dataset_config)

    # For dataset stats: store original image size
    im = Image.open(available_img[0])
    image_w, image_h = im.size[0], im.size[1]
    meta['original_img'] = {'nb': len(available_img), 'size': [image_w, image_h]}
    # Compute new image sizes, cropping overlaps and strides
    cropping_parameters = get_cropping_parameters(image_w, image_h)
    meta['patch_parameters'] = cropping_parameters
    
    # For dataset stats: keep record of the detections:
    stats_dict = {'nb_patches': 0, 
                'nb_detections': 0, 
                'categories': category_name_to_id, 
                'count_per_category': {'all': 0}}
    #category_name_to_count = {'all': 0}
    #nb_patches = 0
    #nb_detect = 0
    
    for img_path in available_img:
        
        img = os.path.basename(img_path)

        # Extract annotations for this image
        df_img_annotations = df[df[dataset_config['annotation_col_names'][0]] == img]

        # Retrieve annotations info
        annotations_labels = np.repeat('bird', len(df_img_annotations)) if not dataset_config['annotation_col_names'][1] else np.array(list(map(lambda x: x.lower(), df_img_annotations[dataset_config['annotation_col_names'][1]]))) #.to_numpy()

        bboxes_coords = df_img_annotations[[dataset_config['annotation_col_names'][2], 
                                            dataset_config['annotation_col_names'][3], 
                                            dataset_config['annotation_col_names'][4], 
                                            dataset_config['annotation_col_names'][5]]].to_numpy(dtype=float)

        if len(bboxes_coords):
            if dataset_config['annotation_format']=="XYWH":
                # Convert to LTBR format required by the WindowCropper class
                bboxes_coords[:,0] -= bboxes_coords[:,2]/2
                bboxes_coords[:,1] -= bboxes_coords[:,3]/2
                bboxes_coords[:,2] += bboxes_coords[:,0]
                bboxes_coords[:,3] += bboxes_coords[:,1]

        logits = []
        image = Image.open(img_path)
        
        # Crop image and corresponding annotations into patches
        cropper = WindowCropper(patchSize=cropping_parameters['new_size'], exportEmptyPatches=False, cropMode='strided',
                                stride=cropping_parameters['strides'], minBBoxArea=dataset_config['minbbox_pixels'], 
                                minBBoxAreaFrac=0.70, cropSize=None, minCropSize=None, maxCropSize=None, 
                                forcePatchSizeAspectRatio=True, maintainAspectRatio=True, searchStride=(10,10,))
        
        stats_dict = cropper.splitImageIntoPatches(image=image, bboxes=bboxes_coords, labels=annotations_labels, 
                                                    logits=logits, saving_img_folder=saving_img_folder, 
                                                    saving_label_folder=saving_label_folder, 
                                                    dataset_config=dataset_config, image_name=img, 
                                                    stats_dictionnary=stats_dict)

    # Store dataset stats
    meta['nb_patches'] = stats_dict['nb_patches']
    meta['detections'] = stats_dict['nb_detections']
    meta['categories'] = stats_dict['categories']
    meta['detections_per_category'] = stats_dict['count_per_category']
    fname = 'dataset_meta.yaml'
    with open(os.path.join(dataset_folder, fname), 'w') as yaml_file:
        yaml_file.write(yaml.dump(meta, default_flow_style=False))

    # Preview a few images
    preview_few_images(dataset_config, dataset_folder, category_name_to_id)


    print("DONE, next source")

# ...for each dataset=source

