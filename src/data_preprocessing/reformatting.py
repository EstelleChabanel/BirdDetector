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

original_dataset_folder = r'/gpfs/gibbs/project/jetz/eec42/data/original'
saving_dataset_folder = r'/gpfs/gibbs/project/jetz/eec42/data/formatted_data'
if not os.path.exists(os.path.join(saving_dataset_folder)):
    os.mkdir(os.path.join(saving_dataset_folder))

yaml_path = r'/home/eec42/BirdDetector/src/data_preprocessing/source_datasets_config.yaml'
config = load_config(yaml_path)

# Dictionnary to store all classes and corresponding int id
category_name_to_id = {}

for dataset in config.keys():

    # Extract specific dataset config
    dataset_config = extract_dataset_config(config, dataset)
    print(dataset_config)

    # Source dataset folder
    source_dataset_folder = os.path.join(original_dataset_folder, dataset_config["name"])
    if not os.path.exists(source_dataset_folder):
        print("Error: can't find the source dataset at following path", source_dataset_folder, "\n going to the next dataset")
        continue

    # Create new folder to store current dataset
    dataset_folder = os.path.join(saving_dataset_folder, dataset_config["name"])
    saving_img_folder = os.path.join(dataset_folder, "images")
    saving_label_folder = os.path.join(dataset_folder, "labels")
    if not os.path.exists(dataset_folder):
        os.mkdir(dataset_folder)
        os.mkdir(saving_img_folder)
        os.mkdir(saving_label_folder)
    
    # Create metadata.txt file to store information on current dataset
    metadata = open(dataset_folder +'/metadata.txt', 'a')
    metadata.write("Dataset: " + repr(dataset) + "\n")
    meta = {'dataset': dataset, 'name': dataset_config['name']}

    # Retrieve all csv annotations files
    csv_files = glob.glob(source_dataset_folder + '/**/*.csv', recursive=True) # should be 1 or 2 max (train+test or all together)
    df = pd.DataFrame()
    for annotation_file in tqdm(csv_files):
            df_ = pd.read_csv(annotation_file)
            df = pd.concat([df, df_])

    # Retrieve list of images
    available_img = [] 
    for subdataset in dataset_config['image_path']:
        source_subdataset_folder = os.path.join(source_dataset_folder, subdataset)
        available_img.extend([os.path.join(source_subdataset_folder, fn) for fn in os.listdir(source_subdataset_folder) if fn.endswith(dataset_config["image_extension"])])
    metadata.write("Nb of images: " + repr(len(available_img)) + "\n")
    # Store original image size
    im = Image.open(available_img[0])
    image_w, image_h = im.size[0], im.size[1]
    metadata.write("Original images size: width=" + repr(image_w) + ", high=" + repr(image_h) + "\n")
    meta['original_img'] = {'nb': len(available_img), 'size': [image_w, image_h]}
    new_img_size = 0
    if (image_w>=640) and (image_h>=640):
        new_img_size = 640
    else:
        new_img_size = (min(image_w, image_h)//32)*32

    if image_w==new_img_size:
        overlap_w = 0
    else:
        overlap_w = (new_img_size*math.ceil(image_w/new_img_size) - image_w)/(math.ceil(image_w/new_img_size)-1)
    if image_h==new_img_size:
        overlap_h = 0
    else:
        overlap_h = (new_img_size*math.ceil(image_h/new_img_size) - image_h)/(math.ceil(image_h/new_img_size)-1)

    stride_w = new_img_size - overlap_w
    stride_h = new_img_size - overlap_h
    metadata.write("Need to resize images, \n New images size: width=" 
                   + repr(new_img_size) + ", high=" + repr(new_img_size) +
                    " \n patches are created with strides (" + repr(stride_w)
                      + ", " + repr(stride_h) + " ) \n" )

    # Keep record of the detections:
    category_name_to_count = {'all': 0}
    nb_patches = 0
    nb_detect = 0
    
    for img_path in available_img:
        
        img = os.path.basename(img_path)
        # Extract annotations for this image
        df_img_annotations = df[df[dataset_config['annotation_col_names'][0]] == img]

        # Crop image and corresponding annotations into patches
        cropper = WindowCropper(patchSize=new_img_size, exportEmptyPatches=False, cropMode='strided',
                                stride=(stride_w, stride_h), minBBoxArea=10, minBBoxAreaFrac=0.25, cropSize=None,
                                minCropSize=None, maxCropSize=None, forcePatchSizeAspectRatio=True,
                                maintainAspectRatio=True, searchStride=(10,10,))
        
        annotations_labels = []
        if not dataset_config['annotation_col_names'][1]:
            annotations_labels = np.repeat('bird', len(df_img_annotations))
        else:
            annotations_labels = np.array(list(map(lambda x: x.lower(), df_img_annotations[dataset_config['annotation_col_names'][1]]))) #.to_numpy()
        

        bboxes_coords = df_img_annotations[[dataset_config['annotation_col_names'][2], 
                                            dataset_config['annotation_col_names'][3], 
                                            dataset_config['annotation_col_names'][4], 
                                            dataset_config['annotation_col_names'][5]]]
        bboxes_coords = bboxes_coords.to_numpy(dtype=float)

        if len(bboxes_coords):
            if dataset_config['annotation_format']=="XYWH":
                # Convert to LTBR format required by the WindowCropper class
                bboxes_coords[:,0] -= bboxes_coords[:,2]/2
                bboxes_coords[:,1] -= bboxes_coords[:,3]/2
                bboxes_coords[:,2] += bboxes_coords[:,0]
                bboxes_coords[:,3] += bboxes_coords[:,1]
        
        logits = []
        image = Image.open(img_path)
        patches, nb_patches, nb_detect, category_name_to_id, category_name_to_count = cropper.splitImageIntoPatches(image=image, bboxes=bboxes_coords, labels=annotations_labels, 
                                                                                        logits=logits, saving_img_folder=saving_img_folder, 
                                                                                        saving_label_folder=saving_label_folder, 
                                                                                        dataset_config=dataset_config, image_name=img, 
                                                                                        category_name_to_id=category_name_to_id,
                                                                                        category_name_to_count=category_name_to_count,
                                                                                        nb_patches=nb_patches, nb_detect=nb_detect)
                    
    meta['patch'] = {'nb': nb_patches, 'size': [new_img_size, new_img_size],  'stride': [stride_w, stride_h]}
    meta['detections'] = nb_detect
    meta['categories'] = category_name_to_id
    meta['detections_per_category'] = category_name_to_count

    metadata.write("Nb of patches: " + repr(nb_patches) + "\n")
    metadata.write("Nb of detections: " + repr(category_name_to_count) + "\n")
    metadata.write("Nb of detections: " + repr(nb_detect) + "\n")
    metadata.write("Distinct labels: " + repr(category_name_to_id) + "\n")

    fname = 'dataset_meta.yaml'
    with open(os.path.join(dataset_folder, fname), 'w') as yaml_file:
        yaml_file.write(yaml.dump(meta, default_flow_style=False))

    preview_few_images(dataset_config, dataset_folder, category_name_to_id)

    print("DONE, next source")

# ...for each dataset=source

