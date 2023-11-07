import os
import glob
import pandas as pd
import shutil
from PIL import Image
import shutil
import yaml
from pathlib import Path

from reformatting_utils import load_config, extract_dataset_config

yaml_path = r'/home/eec42/BirdDetector/src/data_preprocessing/source_datasets_config.yaml'
config = load_config(yaml_path)

original_folder = r'/gpfs/gibbs/project/jetz/eec42/data/formatted_data'
saving_folder = r'/gpfs/gibbs/project/jetz/eec42/data/formatted_data_no_background'

if not os.path.exists(saving_folder):
    os.mkdir(saving_folder)

database1_source = ['global_birds_poland', 'global_birds_palmyra', 'global_birds_penguins',
                    'global_birds_mckellar', 'global_birds_newmexico', 
                    'global_birds_pfeifer', 'uav_thermal_waterfowl']

split_sets = ["train", "val", "test"]
data_temp = {}

for dataset in database1_source:

    count_detections = 0
    nb_img_by_nb_birds = {}  # {0: 12, 1: 4} means 12 images contain 0 bird, 4 images contain 1 bird

    current_folder = os.path.join(original_folder, dataset)
    os.mkdir(os.path.join(saving_folder, dataset))
    os.mkdir(os.path.join(saving_folder, dataset, "images"))
    os.mkdir(os.path.join(saving_folder, dataset, "labels"))

    available_img = os.listdir(os.path.join(current_folder, "images"))
    saved_data = []
    
    for img in available_img:
        if os.path.exists(os.path.join(current_folder, "labels", Path(img).stem + '.txt')):
            f = open(os.path.join(current_folder, "labels", Path(img).stem + '.txt'), "r")
            temp_count = len(f.readlines())
            if temp_count >= 1:
                saved_data.append(img)
                count_detections += temp_count
                if temp_count not in nb_img_by_nb_birds:
                    nb_img_by_nb_birds[temp_count] = 0
                nb_img_by_nb_birds[temp_count] += 1
    
    file = open(os.path.join(saving_folder, dataset,'img_names.txt'),'w')
    file.writelines(saved_data)
    file.close()
    
    data_temp[dataset] = {"nb_img": len(saved_data), "nb_birds": count_detections, "birds_repartition": nb_img_by_nb_birds}


    for data in saved_data:
        shutil.copyfile(os.path.join(current_folder, "images", data), os.path.join(saving_folder, dataset, "images", data))
        shutil.copyfile(os.path.join(current_folder, "labels", Path(data).stem + '.txt'), os.path.join(saving_folder, dataset, "labels", Path(data).stem + '.txt'))


fname = 'data_stats.yaml'
with open(os.path.join(saving_folder, fname), 'w') as yaml_file:
    yaml_file.write( yaml.dump(data_temp, default_flow_style=False))