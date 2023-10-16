import os
import pandas as pd
import shutil
import glob
import numpy as np
import yaml
from pathlib import Path
import json

from collections import defaultdict
from tqdm import tqdm
from PIL import ImageDraw, Image
import random

# TODO: Am I allowed to use that ??
from md_visualization import visualization_utils as visutils
from md_utils import path_utils



# TODO: place in a preprocessing utils.py
def isnan(v):
    if not isinstance(v,float):
        return False
    return np.isnan(v)


def load_config(yaml_path):
    with open(yaml_path, "r") as yamlfile:
        data = yaml.load(yamlfile, Loader=yaml.FullLoader)
        print("Datasets config read successful")
    return data


def extract_dataset_config(yaml_data, dataset_name):
    return yaml_data.get(dataset_name)


def from_ind_csv_to_labels(dataset_config, source_dataset_folder, dataset_folder, category_name_to_id):

    category_name_to_count = {}
    category_name_to_count["all"] = 0
    selected_image_name = None
    selected_image_path = None

    subdatasets = os.listdir(source_dataset_folder)

    for subdataset in subdatasets:

        if not os.path.exists(os.path.join(dataset_folder, subdataset)):
            os.mkdir(os.path.join(dataset_folder, subdataset))
            os.mkdir(os.path.join(dataset_folder, subdataset, "image"))
            os.mkdir(os.path.join(dataset_folder, subdataset, "label"))

        csv_files = glob.glob(os.path.join(source_dataset_folder, subdataset, dataset_config["annotation_path"]) + '/**/*.csv',recursive=True)

        for annotation_csv_file in tqdm(csv_files):
            # TODO: do we still want to keep only munal annotations ??
            # Only look at manually verified annotations
            if 'Manually' not in annotation_csv_file:
                continue
            if os.stat(annotation_csv_file).st_size < 50:
                continue

            df = pd.read_csv(annotation_csv_file)

            image_name = os.path.basename(annotation_csv_file).split('.')[0].split('_')[0]

            available_images = os.listdir(os.path.join(source_dataset_folder, subdataset, dataset_config["image_path"]))
            available_image_names = set([os.path.splitext(s)[0] for s in available_images])

            # Save annotations only for existing images
            if image_name not in available_image_names:
                continue

            selected_image_name = image_name

            selected_image_path = os.path.join(source_dataset_folder, subdataset, dataset_config["image_path"], image_name + dataset_config["image_extension"])
            pil_im = visutils.open_image(selected_image_path)
            image_w = pil_im.size[0]
            image_h = pil_im.size[1]

            for i_row,row in df.iterrows():

                if 'SpeciesCategory' in row:
                    label = row['SpeciesCategory']
                else:
                    label = row['Category']
                    
                if isnan(label) or len(label) == 0:
                    continue

                label = label.lower()
                x = row['X']
                y = row['Y']

                if label not in category_name_to_id:
                    category_name_to_id[label] = len(category_name_to_id)
                if label not in category_name_to_count:
                    category_name_to_count[label] = 0
                
                category_name_to_count["all"] += 1
                category_name_to_count[label] += 1

                ann_radius = 70
                annot = [category_name_to_id[label], 
                            (x-ann_radius)/image_w,
                            (y-ann_radius)/image_h,
                            2*ann_radius/image_w,
                            2*ann_radius/image_h]
                
                # Create .txt label file with all detection boxes
                with open(os.path.join(dataset_folder, subdataset, "label")+'/'+image_name+'.txt', 'a') as f:
                    line = '\t'.join(map(str, annot))
                    # Write the line to the file
                    f.write(line + '\n')

            # ...for each row=annotation in this csv file 
        
            shutil.copyfile(selected_image_path,os.path.join(dataset_folder, subdataset, "image")+'/'+selected_image_name+dataset_config["image_extension"])

            # ...for each csv file=image

    # Save metadata about annotations in this dataset
    with open(dataset_folder +'/metadata.json', 'w') as f:
        json.dump(category_name_to_count, f)

    return category_name_to_id


def from_global_csv_to_labels(dataset_config, source_dataset_folder, dataset_folder, category_name_to_id):
                            
    category_name_to_count = {}
    category_name_to_count["all"] = 0

    annotation_file = os.path.join(source_dataset_folder, dataset_config["annotation_path"])
    df = pd.read_csv(annotation_file)

    available_images = os.listdir(os.path.join(source_dataset_folder, dataset_config["image_path"]))
    available_images_names = set([os.path.splitext(s)[0] for s in available_images])
    label = "waterfowl"
    category_name_to_id[label] = len(category_name_to_id)
    category_name_to_count[label] = 0

    for i_row,row in df.iterrows():

        category_name_to_count["all"] += 1
        category_name_to_count[label] += 1

        image_name = row['imageFilename'].split('.')[0]
        assert image_name in available_images_names
        image_path = os.path.join(source_dataset_folder, dataset_config["image_path"], image_name+dataset_config["image_extension"])
        pil_im = visutils.open_image(image_path)
        image_w = pil_im.size[0]
        image_h = pil_im.size[1]

        annot = [category_name_to_id["waterfowl"], 
                row['x(column)']/image_w,
                row['y(row)']/image_h,
                row['width']/image_w,
                row['height']/image_h]
            
        # Create .txt label file with all detection boxes
        with open(os.path.join(dataset_folder, "label", image_name+'.txt'), 'a') as f:
            line = '\t'.join(map(str, annot))
            # Write the line to the file
            f.write(line + '\n')
        
        if not os.path.exists(os.path.join(dataset_folder, "image")+'/'+image_name+dataset_config["image_extension"]):
            shutil.copyfile(image_path,os.path.join(dataset_folder, "image")+'/'+image_name+dataset_config["image_extension"])

    # for each annotation = detected bird

    # Save metadata about annotations in this dataset
    with open(dataset_folder +'/metadata.json', 'w') as f:
        json.dump(category_name_to_count, f)

    return category_name_to_id



def from_multiple_global_csv_to_labels(dataset_config, source_dataset_folder, dataset_folder, category_name_to_id):
    
    category_name_to_count = {}
    category_name_to_count["all"] = 0

    # Retrieve all csv annotations files
    csv_files = glob.glob(source_dataset_folder + '/**/*.csv', recursive=True)

    for annotation_file in tqdm(csv_files):

        df = pd.read_csv(annotation_file)
        # path to sub-dataset
        subdataset = os.path.basename(os.path.dirname(annotation_file))

        for i_row,row in df.iterrows():

            image_path = os.path.join(source_dataset_folder, subdataset, row['image_path'])
            image_name = row['image_path'].split('.')[0]
            label = row['label'].lower()

            if label not in category_name_to_id:
                category_name_to_id[label] = len(category_name_to_id)
            if label not in category_name_to_count:
                category_name_to_count[label] = 0

            # Check that the image exists
            assert os.path.isfile(image_path)

            # Update counts
            category_name_to_count["all"] += 1
            category_name_to_count[label] += 1

            pil_im = visutils.open_image(image_path)
            image_w = pil_im.size[0]
            image_h = pil_im.size[1]

            annot = [category_name_to_id[label], 
                    row['xmin']/image_w,
                    row['ymin']/image_h,
                    (row['xmax']-row['xmin'])/image_w,
                    (row['ymax']-row['ymin'])/image_h]
                
            # Create subdataset folder
            subdataset_folder = os.path.join(dataset_folder, subdataset)

            if not os.path.exists(subdataset_folder):
                os.mkdir(subdataset_folder)
                os.mkdir(os.path.join(subdataset_folder, "image"))
                os.mkdir(os.path.join(subdataset_folder, "label"))

            # Create .txt label file with all detection boxes
            with open(os.path.join(subdataset_folder, "label", image_name+'.txt'), 'a') as f:
                line = '\t'.join(map(str, annot))
                # Write the line to the file
                f.write(line + '\n')
            
            if not os.path.exists(os.path.join(subdataset_folder, "image")+'/'+image_name+dataset_config["image_extension"]):
                shutil.copyfile(image_path,os.path.join(subdataset_folder, "image")+'/'+image_name+dataset_config["image_extension"])

        # for each annotation = detected bird

    # for each csv annotation file

    # Save metadata about annotations in this dataset
    with open(dataset_folder +'/metadata.json', 'w') as f:
        json.dump(category_name_to_count, f)

    return category_name_to_id


def from_classes_csv_to_labels(dataset_config, source_dataset_folder, dataset_folder, category_name_to_id):

    category_name_to_count = {}
    category_name_to_count["all"] = 0

    # Retrieve all csv annotations files
    csv_files = glob.glob(source_dataset_folder + '/**/*.csv', recursive=True)
    csv_files = [fn for fn in csv_files if 'annotations' in fn]

    for annotation_file in tqdm(csv_files):
    
        df = pd.read_csv(annotation_file,header=None)
        # path to sub-dataset
        subdataset = os.path.basename(os.path.dirname(annotation_file))

        for i_row,row in df.iterrows():

            image_path = os.path.join(source_dataset_folder, subdataset, row[0])
            image_name = row[0].split('.')[0]
            label = row[5]

            if label not in category_name_to_id:
                category_name_to_id[label] = len(category_name_to_id)
            if label not in category_name_to_count:
                category_name_to_count[label] = 0

            # Check that the image exists
            assert os.path.isfile(image_path)

            # Update counts
            category_name_to_count["all"] += 1
            category_name_to_count[label] += 1

            pil_im = visutils.open_image(image_path)
            image_w = pil_im.size[0]
            image_h = pil_im.size[1]

            annot = [category_name_to_id[label], 
                    row[1]/image_w,
                    row[2]/image_h,
                    (row[3]-row[1])/image_w,
                    (row[4]-row[2])/image_h]
                
            # Create subdataset folder
            subdataset_folder = os.path.join(dataset_folder, subdataset)

            if not os.path.exists(subdataset_folder):
                os.mkdir(subdataset_folder)
                os.mkdir(os.path.join(subdataset_folder, "image"))
                os.mkdir(os.path.join(subdataset_folder, "label"))
            
            # Create .txt label file with all detection boxes
            with open(os.path.join(subdataset_folder, "label", image_name+'.txt'), 'a') as f:
                line = '\t'.join(map(str, annot))
                # Write the line to the file
                f.write(line + '\n')
            
            if not os.path.exists(os.path.join(subdataset_folder, "image")+'/'+image_name+dataset_config["image_extension"]):
                shutil.copyfile(image_path,os.path.join(subdataset_folder, "image")+'/'+image_name+dataset_config["image_extension"])

        # for each annotation = detected bird

    # for each csv annotation file

    # Save metadata about annotations in this dataset
    with open(dataset_folder +'/metadata.json', 'w') as f:
        json.dump(category_name_to_count, f)

    return category_name_to_id
        
    
def from_global_json_to_csv(dataset_config, source_dataset_folder, dataset_folder, category_name_to_id):

    category_name_to_count = {}
    category_name_to_count["all"] = 0

    for subdataset in dataset_config["image_path"]:

        # Create subdataset folder
        subdataset_folder = os.path.join(dataset_folder, subdataset)
        if not os.path.exists(subdataset_folder):
            os.mkdir(subdataset_folder)
            os.mkdir(os.path.join(subdataset_folder, "image"))
            os.mkdir(os.path.join(subdataset_folder, "label"))

        available_images = os.listdir(os.path.join(source_dataset_folder, subdataset))
        available_images_files = [fn for fn in available_images if \
                   (fn.lower().endswith('.png') or fn.lower().endswith('.jpg'))]

        annotation_file = glob.glob(os.path.join(source_dataset_folder, subdataset) + '/*.json')

        with open(annotation_file[0], 'r') as f:
            annotations = json.load(f)
        

        for annotation in annotations:

            image_name = annotation['External ID']
            image_path = os.path.join(source_dataset_folder, subdataset, image_name)
            pil_im = visutils.open_image(image_path)
            image_w = pil_im.size[0]
            image_h = pil_im.size[1]

            labels = annotation['Label']
            if len(labels) > 0:
                objects = labels['objects']

                for obj in objects:
                    label = obj['value']
                    coords = obj['bbox']

                    if label not in category_name_to_id:
                        category_name_to_id[label] = len(category_name_to_id)
                    if label not in category_name_to_count:
                        category_name_to_count[label] = 0

                    category_name_to_count["all"] += 1
                    category_name_to_count[label] += 1

                    annot = [category_name_to_id[label], 
                                coords["left"]/image_w, 
                                coords["top"]/image_h,
                                coords["width"]/image_w,
                                coords["height"]/image_h]
                    
                    # Create .txt label file with all detection boxes
                    with open(os.path.join(subdataset_folder, "label", image_name.split(dataset_config["image_extension"])[0]+'.txt'), 'a') as f:
                        line = '\t'.join(map(str, annot))
                        # Write the line to the file
                        f.write(line + '\n')
                    
                    if not os.path.exists(os.path.join(subdataset_folder, "image")+'/'+image_name):
                        shutil.copyfile(image_path,os.path.join(subdataset_folder, "image")+'/'+image_name)

                # ...for each detected object
            
        # ...for each annotation dictionnary

    # ...for each subdataset

    # Save metadata about annotations in this dataset
    with open(dataset_folder +'/metadata.json', 'w') as f:
        json.dump(category_name_to_count, f)

    return category_name_to_id



def from_global_csv_for_tiff_to_labels(dataset_config, source_dataset_folder, dataset_folder, category_name_to_id):

    category_name_to_count = {}
    category_name_to_count["all"] = 0

    annotation_file = os.path.join(source_dataset_folder, dataset_config["annotation_path"])
    df = pd.read_csv(annotation_file)

    image_path = os.path.join(source_dataset_folder, dataset_config["image_path"])
    image_name = os.path.basename(image_path).split(dataset_config["image_extension"])[0]

    Image.MAX_IMAGE_PIXELS = None
    pil_im = visutils.open_image(image_path)
    draw = ImageDraw.Draw(pil_im)
    image_w = pil_im.size[0]
    image_h = pil_im.size[1]

    # From gdalinfo
    image_width_meters = 292.482 # 305.691
    image_height_meters = 305.691 # 292.482
    
    for i_row,row in df.iterrows():

        label = row["label"].lower()
        if label not in category_name_to_id:
            category_name_to_id[label] = len(category_name_to_id)
        if label not in category_name_to_count:
            category_name_to_count[label] = 0

        category_name_to_count["all"] += 1
        category_name_to_count[label] += 1

        ann_radius = 50
        x_meters = row['X']
        y_meters = row['Y']
        x_relative = x_meters / image_width_meters
        y_relative = y_meters / image_height_meters
        
        x_pixels = x_relative * pil_im.size[0]
        y_pixels = y_relative * pil_im.size[1]
        y_pixels = pil_im.size[1] - y_pixels

        annot = [category_name_to_id[label], 
                (x_pixels - ann_radius)/image_w,
                (y_pixels - ann_radius)/image_h,
                2*ann_radius/image_w,
                2*ann_radius/image_h]
            
        # Create .txt label file with all detection boxes
        with open(os.path.join(dataset_folder, "label", image_name+'.txt'), 'a') as f:
            line = '\t'.join(map(str, annot))
            # Write the line to the file
            f.write(line + '\n')
        
        if not os.path.exists(os.path.join(dataset_folder, "image")+'/'+image_name+dataset_config["image_extension"]):
            shutil.copyfile(image_path,os.path.join(dataset_folder, "image")+'/'+image_name+dataset_config["image_extension"])

    # for each annotation = detected bird

    # Save metadata about annotations in this dataset
    with open(dataset_folder +'/metadata.json', 'w') as f:
        json.dump(category_name_to_count, f)


    return category_name_to_id


def preview_few_images(dataset_config, source_dataset_folder, dataset_folder, category_name_to_id):

    to_search = '/**/*' + dataset_config["image_extension"]
    available_images = glob.glob(dataset_folder + to_search, recursive=True) # + dataset_config["image_extension"], recursive=True)
    selected_images = np.random.choice(available_images, 3)
    selected_images_names = [os.path.basename(img).split(dataset_config["image_extension"])[0] for img in selected_images]

    preview_folder = os.path.join(dataset_folder, 'preview')

    for i in range(3):

        # Open image
        pil_im = visutils.open_image(selected_images[i])
        draw = ImageDraw.Draw(pil_im)

        # Open corresponding labels
        selected_label = glob.glob(dataset_folder + '/**/' + selected_images_names[i] + '.txt', recursive=True)
        if len(selected_label) == 0:
            continue

        detection_boxes = []
        category_id_to_name = {v: k for k, v in category_name_to_id.items()}
        print(category_id_to_name)

        df = pd.read_csv(selected_label[0], sep='\t', header=None, index_col=False)
        for irow, row in df.iterrows():  
            det = {}
            det['conf'] = None
            det['category'] = row[0]
            det['bbox'] = [row[1], row[2], row[3], row[4]]
            detection_boxes.append(det)

        # Draw annotations
        output_file_annotated = preview_folder + selected_images_names[i] + '.JPG'
        visutils.draw_bounding_boxes_on_file(selected_images[i], output_file_annotated, detection_boxes,
                                     confidence_threshold=0.0,#detector_label_map=category_id_to_name,
                                     thickness=1,expansion=0)
        path_utils.open_file(output_file_annotated)


