import os
import numpy as np
from PIL import Image
import yaml

from preprocessing_utils import load_config, extract_dataset_config, preview_few_images, retrieve_detections_from_csv, retrieve_img_list, get_cropping_parameters, get_imglabel_pair
from windowCropping import WindowCropper
import visualization_utils as visutils

# PIL gets very sad when you try to load large images, suppress the error
Image.MAX_IMAGE_PIXELS = None

# ======= PARAMETERS =======

ORIGINAL_DATASET_FOLDER = r'/gpfs/gibbs/project/jetz/eec42/data/original'
SAVING_DATASET_FOLDER = r'/gpfs/gibbs/project/jetz/eec42/data/formatted_data_'
if not os.path.exists(os.path.join(SAVING_DATASET_FOLDER)):
    os.mkdir(os.path.join(SAVING_DATASET_FOLDER))

YAML_PATH = r'/home/eec42/BirdDetector/src/data_preprocessing/source_datasets_config.yaml'
config = load_config(YAML_PATH)


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
    df = df.drop_duplicates()

    # Retrieve list of images
    available_img = retrieve_img_list(source_dataset_folder, dataset_config)

    # For dataset stats: store original image size
    im = visutils.open_image(available_img[0]) #Image.open(available_img[0])
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
    
    nb_detections = 0
    
    for img_path in available_img:
        
        img = os.path.basename(img_path)

        # Extract annotations for this image
        if not dataset=='terns_africa':
            df_img_annotations = df[df[dataset_config['annotation_col_names'][0]] == img]
            subset = dataset_config['annotation_col_names'] if dataset_config['annotation_col_names'][1] else [dataset_config['annotation_col_names'][0], dataset_config['annotation_col_names'][2], dataset_config['annotation_col_names'][3], dataset_config['annotation_col_names'][4], dataset_config['annotation_col_names'][5]]
            df_img_annotations = df_img_annotations.drop_duplicates(subset=subset) 
        else:
            df_img_annotations = df

        # Retrieve annotations info
        annotations_labels = np.repeat('bird', len(df_img_annotations)) if not dataset_config['annotation_col_names'][1] else np.array(list(map(lambda x: x.lower(), df_img_annotations[dataset_config['annotation_col_names'][1]]))) #.to_numpy()

        bboxes_coords = [[]]
        if not dataset_config['annotation_format']=="XY":
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
        else:
            image_width_meters = 292.482 # 305.691
            image_height_meters = 305.691 # 292.482
            df_img_annotations["x_pixels"] = (df_img_annotations[dataset_config['annotation_col_names'][2]]/ image_width_meters * image_w )
            df_img_annotations["y_pixels"] = image_h - (df_img_annotations[dataset_config['annotation_col_names'][3]]/ image_height_meters * image_h )
            bboxes_coords = df_img_annotations[[dataset_config['annotation_col_names'][2], 
                                                dataset_config['annotation_col_names'][3], 
                                                dataset_config['annotation_col_names'][2], 
                                                dataset_config['annotation_col_names'][3]]].to_numpy(dtype=float)
            # Convert to LTBR format required by the WindowCropper class
            bboxes_coords[:,0] = df_img_annotations["x_pixels"]-20
            bboxes_coords[:,1] = df_img_annotations["y_pixels"]-20
            bboxes_coords[:,2] = df_img_annotations["x_pixels"]+20
            bboxes_coords[:,3] = df_img_annotations["y_pixels"]+20


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

