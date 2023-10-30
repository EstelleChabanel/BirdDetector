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

original_folder = r'/gpfs/gibbs/project/jetz/eec42/data/baseline1'
saving_folder = r'/gpfs/gibbs/project/jetz/eec42/data/baseline1_experiment_no_backgd'

if not os.path.exists(saving_folder):
    os.mkdir(saving_folder)
    os.mkdir(os.path.join(saving_folder, "train"))
    os.mkdir(os.path.join(saving_folder, "train", "images"))
    os.mkdir(os.path.join(saving_folder, "train", "labels"))
    os.mkdir(os.path.join(saving_folder, "val"))
    os.mkdir(os.path.join(saving_folder, "val", "images"))
    os.mkdir(os.path.join(saving_folder, "val", "labels"))
    os.mkdir(os.path.join(saving_folder, "test"))
    os.mkdir(os.path.join(saving_folder, "test", "images"))
    os.mkdir(os.path.join(saving_folder, "test", "labels"))

database1_source = ['global-bird-zenodo_poland', 'global-bird-zenodo_palmyra', 'global-bird-zenodo_penguins',
                    'global-bird-zenodo_mckellar', 'global-bird-zenodo_newmexico', 
                    'global-bird-zenodo_pfeifer', 'uav-waterfowl-thermal']

split_sets = ["train", "val", "test"]
data_temp = {}

for split_set in split_sets:

    count_detections = 0
    nb_img_by_nb_birds = {}  # {0: 12, 1: 4} means 12 images contain 0 bird, 4 images contain 1 bird

    current_folder = os.path.join(original_folder, split_set)
    available_img = os.listdir(os.path.join(current_folder, "images"))
    saved_data = []
    
    for img in available_img:
        if os.path.exists(os.path.join(current_folder, "labels", Path(img).stem + '.txt')):
            f = open(os.path.join(current_folder, "labels", Path(img).stem + '.txt'), "r")
            temp_count = len(f.readlines())
            print(temp_count)
            if temp_count >= 1:
                print("nb: ", temp_count)
                saved_data.append(img)
                print(temp_count)
                count_detections += temp_count
                if temp_count not in nb_img_by_nb_birds:
                    nb_img_by_nb_birds[temp_count] = 0
                nb_img_by_nb_birds[temp_count] += 1
    
    data_temp[split_set] = {"nb_img": len(saved_data), "nb_birds": count_detections, "birds_repartition": nb_img_by_nb_birds}


    for data in saved_data:
        shutil.copyfile(os.path.join(current_folder, "images", data), os.path.join(saving_folder, split_set, "images", data))
        shutil.copyfile(os.path.join(current_folder, "labels",  Path(data).stem + '.txt'), os.path.join(saving_folder, split_set, "labels", Path(data).stem + '.txt'))

    if split_set == "test":

        for subdataset in database1_source:

            dataset_config = extract_dataset_config(config, subdataset)
            current_folder = os.path.join(original_folder, split_set, dataset_config['name'])
            if not os.path.exists(os.path.join(saving_folder, split_set, dataset_config['name'])):
                os.mkdir(os.path.join(saving_folder, split_set, dataset_config['name']))
                os.mkdir(os.path.join(saving_folder, split_set, dataset_config['name'], "images"))
                os.mkdir(os.path.join(saving_folder, split_set, dataset_config['name'], "labels"))
            available_img = os.listdir(os.path.join(current_folder, "images"))
            saved_data = []
            
            for img in available_img:
                if os.path.exists(os.path.join(current_folder, "labels", Path(img).stem + '.txt')):
                    f = open(os.path.join(current_folder, "labels", Path(img).stem + '.txt'), "r")
                    if len(f.readlines()) >= 1:
                        saved_data.append(img)

            for data in saved_data:
                shutil.copyfile(os.path.join(current_folder, "images", data), os.path.join(saving_folder, split_set, dataset_config['name'], "images", data))
                shutil.copyfile(os.path.join(current_folder, "labels",  Path(data).stem + '.txt'), os.path.join(saving_folder, split_set, dataset_config['name'], "labels", Path(data).stem + '.txt'))


fname = 'data_stats.yaml'
with open(os.path.join(saving_folder, fname), 'w') as yaml_file:
    yaml_file.write( yaml.dump(data_temp, default_flow_style=False))