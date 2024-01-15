''''
should get rid of this file
'''


import os
import pandas as pd
import shutil
import glob
import numpy as np
from pathlib import Path
import json
import math

from collections import defaultdict
from tqdm import tqdm
from PIL import ImageDraw, Image
import random
import yaml

# TODO: Am I allowed to use that ??
#from md_visualization import visualization_utils as visutils
import visualization_utils as visutils


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


def preview_few_images(dataset_config, dataset_folder, category_name_to_id, nb_display=3, saving_path=None):

    to_search = '/labels/**/*' + '.txt' #dataset_config["image_extension"]
    available_images = glob.glob(dataset_folder + to_search, recursive=True)
    selected_images = np.random.choice(available_images, nb_display)
    #selected_images_names = [os.path.basename(img).split(dataset_config["image_extension"])[0] for img in selected_images]
    selected_images_names = [os.path.basename(img).split('.txt')[0] for img in selected_images]

    preview_folder = os.path.join(dataset_folder, 'preview')
    if not saving_path:
        saving_path = preview_folder

    for i in range(nb_display):    
        preview_image(selected_images_names[i], dataset_folder, category_name_to_id, saving_path)
            


def preview_image(img_name, path, category_name_to_id, saving_img_path):

    # Open image
    img_path = os.path.join(path, "images", img_name + '.jpg')
    pil_im = visutils.open_image(img_path)
    draw = ImageDraw.Draw(pil_im)

    # Open corresponding labels
    label_path = os.path.join(path, "labels", img_name + '.txt')
    if not os.path.exists(label_path):
        return
        
    detection_boxes = []
    category_id_to_name = {v: k for k, v in category_name_to_id.items()}

    df = pd.read_csv(label_path, sep='\t', header=None, index_col=False)
    for irow, row in df.iterrows():  
        det = {}
        det['conf'] = None
        det['category'] = row[0]
        det['bbox'] = [row[1]-row[3]/2, row[2]-row[4]/2, row[3], row[4]]
        detection_boxes.append(det)
    
    output_file_annotated = saving_img_path + img_name + '.jpg'    
    visutils.draw_bounding_boxes_on_file(img_path, output_file_annotated, detection_boxes,
                                         confidence_threshold=0.0, #detector_label_map=category_id_to_name,
                                         thickness=1,expansion=0)





# TODO: get rid off it when every dataset pre-processed


def save_img(source_img_folder, saving_img_folder, current_img, resize, new_img_size=0):
    if resize == False:
        shutil.copyfile(os.path.join(source_img_folder, current_img),os.path.join(saving_img_folder, "images")+'/'+current_img)
    else:
        im = Image.open(os.path.join(source_img_folder, current_img))
        im_cropped = im.crop((0, 0, new_img_size, new_img_size))
        im_cropped.save(os.path.join(saving_img_folder, "images")+'/'+current_img)


def from_global_csv_to_labels(dataset_config, source_dataset_folder, dataset_folder, category_name_to_id):
                            
    category_name_to_count = {}
    category_name_to_count["all"] = 0
    resize = True

    metadata = open(dataset_folder +'/metadata.txt', 'a')

    if not os.path.exists(os.path.join(dataset_folder, "images")):
        os.mkdir(os.path.join(dataset_folder, "images"))
        os.mkdir(os.path.join(dataset_folder, "labels"))

    annotation_file = os.path.join(source_dataset_folder, dataset_config["annotation_path"])
    df = pd.read_csv(annotation_file)

    available_img = []
    available_img_names = []
    # Both positive & negative images
    for path in dataset_config["image_path"]:
        temp = os.listdir(os.path.join(source_dataset_folder, path))
        available_img_names.extend(set([os.path.splitext(s)[0] for s in available_img]))
        available_img.extend(set([os.path.join(source_dataset_folder, path, temp_i) for temp_i in temp]))
        #available_img.extend(os.listdir(os.path.join(source_dataset_folder, path)))
    #available_img_names = set([os.path.splitext(s)[0] for s in available_images])
    metadata.write("Nb of images: " + repr(len(available_img)) + "\n")

    # Check the size of the images:
    im = Image.open(os.path.join(source_dataset_folder, available_img[0]))
    image_w = im.size[0]
    image_h = im.size[1]
    new_img_size = 512
    metadata.write("Original images size: width=" + repr(image_w) + ", high=" + repr(image_h) + "\n")
    metadata.write("Need to resize images, \n New images size: width=" + repr(new_img_size) + ", high=" + repr(new_img_size) + " \n" )
    

    for img, img_name in zip(available_img, available_img_names):

        # Crop and save images
        im = Image.open(img)
        for new_img_i in range(math.ceil(image_w/new_img_size)):

            overlap = (new_img_size*math.ceil(image_w/new_img_size) - image_h)/(math.ceil(image_w/new_img_size)-1)
            im1 = im.crop((new_img_i*(new_img_size-overlap), 0, (new_img_i+1)*new_img_size, (new_img_i+1)*new_img_size))
            im1.save(os.path.join(dataset_folder, "images")+'/'+ img_name + '_cropped' + new_img_i + dataset_config['image_extension'])

        # Create, crop and save labels
        df_img_annotations = df[df['imageFilename'].split(dataset_config['image_extension'])[0] == img_name]

        with open(os.path.join(dataset_folder, "labels") + '/' + img_name+'.txt', 'a') as f:

            if len(df_img_annotations) == 0:
                f.write(' ' + '\n')
                continue

            for i_row, row in df_img_annotations.iterrows():

                    if not (row['imageFilename'].split(dataset_config['image_extension'])[0] == img_name):
                        print("ERROR !!!!")

                    label = 'bird'
                    if label not in category_name_to_id:
                        category_name_to_id[label] = len(category_name_to_id)
                    if label not in category_name_to_count:
                        category_name_to_count[label] = 0

                    annot = []
                    if resize==False:
                        print("Error !!!")

                    if resize == False:
                        annot.exend([category_name_to_id[label], 
                                     (row['x(column)']+row['width']/2)/image_w,
                                     (row['y(row)']+row['height']/2)/image_h,
                                     row['width']/image_w,
                                     row['height']/image_h])
                        # Update counts
                        category_name_to_count["all"] += 1
                        category_name_to_count[label] += 1
                    
                    else:
                        # Recrop the image to new_img_size*new_img_size, so keep only labels inside this window
                        if ((row['xmin']<new_img_size) and (row['ymin']<new_img_size)):
                            if (row['xmax']>new_img_size):
                                if (row['ymax']>new_img_size):
                                    annot.extend([category_name_to_id[label], 
                                            (row['xmin']+(row['xmax']-row['xmin'])/2)/new_img_size,
                                            (row['ymin']+(row['ymax']-row['ymin'])/2)/new_img_size,
                                            (new_img_size-row['xmin'])/new_img_size,
                                            (new_img_size-row['ymin'])/new_img_size])
                                else:
                                    annot.extend([category_name_to_id[label], 
                                            (row['xmin']+(row['xmax']-row['xmin'])/2)/new_img_size,
                                            (row['ymin']+(row['ymax']-row['ymin'])/2)/new_img_size,
                                            (new_img_size-row['xmin'])/new_img_size,
                                            (row['ymax']-row['ymin'])/new_img_size])
                            elif (row['ymax']>new_img_size):
                                annot.extend([category_name_to_id[label], 
                                            (row['xmin']+(row['xmax']-row['xmin'])/2)/new_img_size,
                                            (row['ymin']+(row['ymax']-row['ymin'])/2)/new_img_size,
                                            (row['xmax']-row['xmin'])/new_img_size,
                                            (new_img_size-row['ymin'])/new_img_size])
                            else:
                                annot.extend([category_name_to_id[label], 
                                            (row['xmin']+(row['xmax']-row['xmin'])/2)/new_img_size,
                                            (row['ymin']+(row['ymax']-row['ymin'])/2)/new_img_size,
                                            (row['xmax']-row['xmin'])/new_img_size,
                                            (row['ymax']-row['ymin'])/new_img_size])

                            # Update counts
                            category_name_to_count["all"] += 1
                            category_name_to_count[label] += 1
                    
                    #save annotation into label file
                    line = '\t'.join(map(str, annot))
                    # Write the line to the file
                    f.write(line + '\n')



    
    
    label = "bird"
    if label not in category_name_to_id:
        category_name_to_id[label] = len(category_name_to_id)
    if label not in category_name_to_count:
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




def from_multiple_global_csv_to_labels(dataset_config, source_dataset_folder, dataset_folder, category_name_to_id, general_bird=True):
    
    category_name_to_count = {}
    category_name_to_count["all"] = 0

    metadata = open(dataset_folder +'/metadata.txt', 'a')

    subdatasets = os.listdir(source_dataset_folder)

    for subdataset in subdatasets:

        source_subdataset_folder = os.path.join(source_dataset_folder, subdataset)
        resize = False
        # Create subdataset folder
        subdataset_folder = os.path.join(dataset_folder, subdataset)
        if not os.path.exists(subdataset_folder):
            os.mkdir(subdataset_folder)
            os.mkdir(os.path.join(subdataset_folder, "images"))
            os.mkdir(os.path.join(subdataset_folder, "labels"))

        # Retrieve all csv annotations files
        csv_files = glob.glob(source_dataset_folder + '/**/*.csv', recursive=True) # should be 1 or 2 (train+test)
        df = pd.DataFrame()

        for annotation_file in tqdm(csv_files):

            df_ = pd.read_csv(annotation_file)
            df = pd.concat([df, df_])
    
        metadata.write("Subdataset: " + repr(subdataset) + "\n")

        available_img = [fn for fn in os.listdir(source_subdataset_folder) if fn.endswith(dataset_config["image_extension"])]
        metadata.write("Nb of images: " + repr(len(available_img)) + "\n")

        im = Image.open(os.path.join(source_subdataset_folder, available_img[0]))
        image_w = im.size[0]
        image_h = im.size[1]
        metadata.write("Original images size: width=" + repr(image_w) + ", high=" + repr(image_h) + "\n")
        new_img_size = (image_w//32)*32
        if (image_w != new_img_size) or (image_h!=new_img_size):
            resize = True
            metadata.write("Need to resize images, \n New images size: width=" + repr(new_img_size) + ", high=" + repr(new_img_size) + " \n" )


        for img in available_img:
            
            # Save images
            save_img(source_subdataset_folder, subdataset_folder, img, resize, new_img_size)
            #if resize == False:
             #   shutil.copyfile(os.path.join(source_subdataset_folder, img),os.path.join(subdataset_folder, "images")+'/'+img)
            #else:
             #   im = Image.open(os.path.join(source_subdataset_folder, img))
              #  im_cropped = im.crop((0, 0, new_img_size, new_img_size))
               # im_cropped.save(os.path.join(subdataset_folder, "images")+'/'+img)

            # Create and save label file
            df_img_annotations = df[df['image_path'] == img]

            with open(os.path.join(subdataset_folder, "labels") + '/' + img.split(dataset_config['image_extension'])[0]+'.txt', 'a') as f:
                for i_row, row in df_img_annotations.iterrows():

                    if not (row['image_path'] == img):
                        print("ERRO !!!!")

                    label = row['label'].lower()
                    if label not in category_name_to_id:
                        category_name_to_id[label] = len(category_name_to_id)
                    if label not in category_name_to_count:
                        category_name_to_count[label] = 0

                    annot = []
                    if resize==False:
                        print("Error !!!")

                    if resize == False:
                        annot.extend([category_name_to_id[label], 
                                (row['xmin']+(row['xmax']-row['xmin'])/2)/image_w,
                                (row['ymin']+(row['ymax']-row['ymin'])/2)/image_h,
                                (row['xmax']-row['xmin'])/image_w,
                                (row['ymax']-row['ymin'])/image_h])

                        # Update counts
                        category_name_to_count["all"] += 1
                        category_name_to_count[label] += 1
                    
                    else:
                        # Recrop the image to new_img_size*new_img_size, so keep only labels inside this window
                        if ((row['xmin']<new_img_size) and (row['ymin']<new_img_size)):
                            if (row['xmax']>new_img_size):
                                if (row['ymax']>new_img_size):
                                    annot.extend([category_name_to_id[label], 
                                            (row['xmin']+(row['xmax']-row['xmin'])/2)/new_img_size,
                                            (row['ymin']+(row['ymax']-row['ymin'])/2)/new_img_size,
                                            (new_img_size-row['xmin'])/new_img_size,
                                            (new_img_size-row['ymin'])/new_img_size])
                                else:
                                    annot.extend([category_name_to_id[label], 
                                            (row['xmin']+(row['xmax']-row['xmin'])/2)/new_img_size,
                                            (row['ymin']+(row['ymax']-row['ymin'])/2)/new_img_size,
                                            (new_img_size-row['xmin'])/new_img_size,
                                            (row['ymax']-row['ymin'])/new_img_size])
                            elif (row['ymax']>new_img_size):
                                annot.extend([category_name_to_id[label], 
                                            (row['xmin']+(row['xmax']-row['xmin'])/2)/new_img_size,
                                            (row['ymin']+(row['ymax']-row['ymin'])/2)/new_img_size,
                                            (row['xmax']-row['xmin'])/new_img_size,
                                            (new_img_size-row['ymin'])/new_img_size])
                            else:
                                annot.extend([category_name_to_id[label], 
                                            (row['xmin']+(row['xmax']-row['xmin'])/2)/new_img_size,
                                            (row['ymin']+(row['ymax']-row['ymin'])/2)/new_img_size,
                                            (row['xmax']-row['xmin'])/new_img_size,
                                            (row['ymax']-row['ymin'])/new_img_size])

                            # Update counts
                            category_name_to_count["all"] += 1
                            category_name_to_count[label] += 1
                    
                    #save annotation into label file
                    line = '\t'.join(map(str, annot))
                    # Write the line to the file
                    f.write(line + '\n')

                # for each annotation = bird annotated
            # close the image label file

        # for each image in the subdataset

    metadata.write("Detections: " + repr(category_name_to_count) + "\n")
    # for each subdataset

    return category_name_to_id


def from_multiple_global_csv_to_labels_OLD(dataset_config, source_dataset_folder, dataset_folder, category_name_to_id, general_bird=True):
    
    category_name_to_count = {}
    category_name_to_count["all"] = 0

    metadata = open(dataset_folder +'/metadata.txt', 'w')

    # Retrieve all csv annotations files
    csv_files = glob.glob(source_dataset_folder + '/**/*.csv', recursive=True)

    for annotation_file in tqdm(csv_files):

        df = pd.read_csv(annotation_file)
        # sub-dataset
        subdataset = os.path.basename(os.path.dirname(annotation_file))
        source_subdataset_folder = os.path.join(source_dataset_folder, subdataset)
        metadata.write("Subdataset: " + repr(subdataset) + "\n")

        available_img = [fn for fn in os.listdir(source_subdataset_folder) if fn.endswith(dataset_config["image_extension"])]
        metadata.write("Nb of images: " + repr(len(available_img)) + "\n")

        for i_row,row in df.iterrows():
            annot = []

            image_path = os.path.join(source_dataset_folder, subdataset, row['image_path'])
            image_name = row['image_path'].split('.')[0]
            label = row['label'].lower()

            if label not in category_name_to_id:
                category_name_to_id[label] = len(category_name_to_id)
            if label not in category_name_to_count:
                category_name_to_count[label] = 0

            # Check that the image exists
            assert os.path.isfile(image_path)

            pil_im = visutils.open_image(image_path)
            image_w = pil_im.size[0]
            image_h = pil_im.size[1]
            new_img_size = image_w
                
            # Create subdataset folder
            subdataset_folder = os.path.join(dataset_folder, subdataset)
            if not os.path.exists(subdataset_folder):
                os.mkdir(subdataset_folder)
                os.mkdir(os.path.join(subdataset_folder, "image"))
                os.mkdir(os.path.join(subdataset_folder, "label"))
            
            if (image_w%32!=0) or (image_h%32!=0):
                new_img_size = (image_w//32)*32
                if i_row==0:
                    metadata.write("Old images size: width=" + repr(image_w) + ", high=" + repr(image_h) + "\n")
                    metadata.write("New images size: width=" + repr(new_img_size) + ", high=" + repr(new_img_size) + " \n" )

                # Recrop the image to new_img_size*new_img_size, so keep only labels inside this window
                if not ((row['xmin']>new_img_size) or (row['ymin']>new_img_size)):
                    if (row['xmax']>new_img_size):
                        if (row['ymax']>new_img_size):
                            annot.extend([category_name_to_id[label], 
                                    row['xmin']/new_img_size,
                                    row['ymin']/new_img_size,
                                    (new_img_size-row['xmin'])/new_img_size,
                                    (new_img_size-row['ymin'])/new_img_size])
                        else:
                            annot.extend([category_name_to_id[label], 
                                    row['xmin']/new_img_size,
                                    row['ymin']/new_img_size,
                                    (new_img_size-row['xmin'])/new_img_size,
                                    (row['ymax']-row['ymin'])/new_img_size])
                    elif (row['ymax']>new_img_size):
                        annot.extend([category_name_to_id[label], 
                                    row['xmin']/new_img_size,
                                    row['ymin']/new_img_size,
                                    (row['xmax']-row['xmin'])/new_img_size,
                                    (new_img_size-row['ymin'])/new_img_size])
                    else:
                        annot.extend([category_name_to_id[label], 
                                    row['xmin']/new_img_size,
                                    row['ymin']/new_img_size,
                                    (row['xmax']-row['xmin'])/new_img_size,
                                    (row['ymax']-row['ymin'])/new_img_size])

                    # Update counts
                    category_name_to_count["all"] += 1
                    category_name_to_count[label] += 1
                
            else:
                annot.extend([category_name_to_id[label], 
                              row['xmin']/image_w,
                              row['ymin']/image_h,
                              (row['xmax']-row['xmin'])/image_w,
                              (row['ymax']-row['ymin'])/image_h])
                
                # Update counts
                category_name_to_count["all"] += 1
                category_name_to_count[label] += 1
            
            if len(annot) != 0:
                # Create .txt label file with all detection boxes
                with open(os.path.join(subdataset_folder, "label", image_name+'.txt'), 'a') as f:
                    line = '\t'.join(map(str, annot))
                    # Write the line to the file
                    f.write(line + '\n')
            
#            if not os.path.exists(os.path.join(subdataset_folder, "image")+'/'+image_name+dataset_config["image_extension"]):
  #              # Resize images and labels before saving them
   #             im = Image.open(image_path)
    #            im_cropped = im.crop((0, 0, 480, 480))
     #           im_cropped.save(os.path.join(subdataset_folder, "image")+'/'+image_name+dataset_config["image_extension"])

        # for each annotation = detected bird

        available_img = [fn for fn in os.listdir(source_subdataset_folder) if fn.endswith(dataset_config["image_extension"])]
        metadata.write("Nb of images: " + repr(len(available_img)) + "\n")
        for img in available_img:
            if not os.path.exists(os.path.join(subdataset_folder, "image")+'/'+img):
                # Resize images and labels before saving them
                im = Image.open(image_path)

                if (im.size[0]%32!=0) or (im.size[1]%32!=0):
                    new_img_size = (image_w//32)*32
                    im_cropped = im.crop((0, 0, new_img_size, new_img_size))
                    im_cropped.save(os.path.join(subdataset_folder, "image")+'/'+img)

        # for each image in the subdataset folder

    # for each csv annotation file

    # Save metadata about annotations in this dataset
    metadata.write("Detections: " + repr(category_name_to_count) + "\n")
    #with open(dataset_folder +'/metadata.json', 'w') as f:
    #    json.dump(category_name_to_count, f)

    return category_name_to_id



def from_global_csv_to_labels_OLD(dataset_config, source_dataset_folder, dataset_folder, category_name_to_id):
                            
    category_name_to_count = {}
    category_name_to_count["all"] = 0

    if not os.path.exists(os.path.join(dataset_folder, "image")):
        os.mkdir(os.path.join(dataset_folder, "image"))
        os.mkdir(os.path.join(dataset_folder, "label"))

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

    if not os.path.exists(os.path.join(dataset_folder, "image")):
        os.mkdir(os.path.join(dataset_folder, "image"))
        os.mkdir(os.path.join(dataset_folder, "label"))
        
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


def from_ind_csv_to_labels(dataset_config, source_dataset_folder, dataset_folder, category_name_to_id, general_bird=False):

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
                
                if general_bird:
                    label = "bird"
                else:
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
        

