import os
import pandas as pd
import glob
import numpy as np
import math

from tqdm import tqdm
from PIL import ImageDraw
import yaml

# TODO: Am I allowed to use that ??
#from md_visualization import visualization_utils as visutils
import visualization_utils as visutils


def isnan(v):
    if not isinstance(v,float):
        return False
    return np.isnan(v)


def load_config(yaml_path):
    '''load sources config file'''
    with open(yaml_path, "r") as yamlfile:
        data = yaml.load(yamlfile, Loader=yaml.FullLoader)
        print("Datasets config read successful")
    return data


def extract_dataset_config(yaml_data, dataset_name):
    '''retrieve configuration specific to current source from global config dictionary'''
    return yaml_data.get(dataset_name)


def preview_few_images(dataset_config, dataset_folder, category_name_to_id, nb_display=3, saving_path=None):
    '''
    preview nb_display images of the current source, with bird annotations
    '''
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
    '''
    create visualization of image with annotation displayed on image
    '''
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



def get_imglabel_pair(img_file, current_folder):
    """
    Get corresponding pair image-label files for dataset processing
    Args:
        img_file (str): Name of the input image file
        current_foder (str): path of the dataset folder in which image-label are stored
    Returns:
        tuple: A tuple containing the image and ground truth bounding boxes
    """
    file_name =  Path(img_file).stem
    label_path = os.path.join(current_folder, "labels", file_name + '.txt')
    if os.path.exists(label_path):
        return file_name, open(label_path, "r").readlines()
    else:
        return file_name, None
