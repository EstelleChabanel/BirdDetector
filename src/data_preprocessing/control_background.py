import os
import shutil
import shutil
import yaml
from pathlib import Path
import math
import random

from preprocessing_utils import load_config, get_imglabel_pair


# ======= PARAMETERS =======

YAML_PATH = r'/home/eec42/BirdDetector/src/data_preprocessing/source_datasets_config.yaml'
config = load_config(YAML_PATH)

ORIGINAL_FOLDER = r'/gpfs/gibbs/project/jetz/eec42/data/formatted_data_'
SAVING_FOLDER = r'/gpfs/gibbs/project/jetz/eec42/data/formatted_data_10percent_background_'

DATABASE1_SOURCE = ['global_birds_poland', 'global_birds_palmyra', 'global_birds_penguins',
                    'global_birds_mckellar', 'global_birds_newmexico', 'global_birds_pfeifer',
                    'uav_thermal_waterfowl', 'hayes_albatross', 'terns_africa']

BACKGROUND_THRESHOLD = 1  
BACKGROUND_PERCENTAGE = 0.10


# ======= IMAGES SELECTION =======

if not os.path.exists(SAVING_FOLDER):
    os.mkdir(SAVING_FOLDER)

# Dictionnary to save dataset stats
data_temp = {}

# Treat dataset one by one
for dataset in DATABASE1_SOURCE:

    count_detections = 0
    nb_img_by_nb_birds = {}  # {0: 12, 1: 4} means 12 images contain 0 bird, 4 images contain 1 bird

    current_folder = os.path.join(ORIGINAL_FOLDER, dataset)
    os.makedirs(os.path.join(SAVING_FOLDER, dataset, "images"))
    os.makedirs(os.path.join(SAVING_FOLDER, dataset, "labels"))

    available_img = os.listdir(os.path.join(current_folder, "images"))
    saved_data = []
    background_data = []
    
    for img in available_img:
        image_name, detections_list = get_imglabel_pair(img, current_folder)
        if detections_list:
            temp_count = len(detections_list)
            if temp_count >= BACKGROUND_THRESHOLD:
                saved_data.append(img)
                count_detections += temp_count
                if temp_count not in nb_img_by_nb_birds:
                    nb_img_by_nb_birds[temp_count] = 0
                nb_img_by_nb_birds[temp_count] += 1
        else:
            background_data.append(img)
    
    if BACKGROUND_THRESHOLD>0 and BACKGROUND_PERCENTAGE>0:
        #nb_img = len(saved_data)
        #print(nb_img)
        #print(len(background_data))
        nb_background_desired = math.ceil( BACKGROUND_PERCENTAGE * len(saved_data)/(1-BACKGROUND_PERCENTAGE) ) #math.ceil( nb_img / (1/BACKGROUND_PERCENTAGE - 1) ) 
        if nb_background_desired>len(background_data):
            nb_background_desired = len(background_data)
        if dataset == 'global_birds_pfeifer':
            nb_background_desired = 0
        print(nb_background_desired)
        saved_data.extend(random.sample(background_data, nb_background_desired))
        nb_img_by_nb_birds[0] = nb_background_desired
        
    # Save dataset stats
    data_temp[dataset] = {"nb_img": len(saved_data), "nb_birds": count_detections, "birds_repartition": nb_img_by_nb_birds}

    for data in saved_data:
        shutil.copyfile(os.path.join(current_folder, "images", data), os.path.join(SAVING_FOLDER, dataset, "images", data))
        if os.path.exists(os.path.join(current_folder, "labels", Path(data).stem + '.txt')):
            shutil.copyfile(os.path.join(current_folder, "labels", Path(data).stem + '.txt'), os.path.join(SAVING_FOLDER, dataset, "labels", Path(data).stem + '.txt'))


fname = 'data_stats.yaml'
with open(os.path.join(SAVING_FOLDER, fname), 'w') as yaml_file:
    yaml_file.write( yaml.dump(data_temp, default_flow_style=False))