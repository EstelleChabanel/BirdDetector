import os
import glob
import pandas as pd
import shutil
from PIL import Image
import shutil
import yaml

from reformatting_utils import load_config, extract_dataset_config


# ======= PARAMETERS =======

ORIGINAL_FOLDER = r'/gpfs/gibbs/project/jetz/eec42/data/original'
SOURCE_FOLDER = r'/gpfs/gibbs/project/jetz/eec42/data/formatted_data_no_background'
SAVING_FOLDER = r'/gpfs/gibbs/project/jetz/eec42/data/baseline1_no_background'

YAML_PATH = r'/home/eec42/BirdDetector/src/data_preprocessing/source_datasets_config.yaml'

TRAIN_PERCENTAGE = 0.7
TEST_PERCENTAGE = 0.2
VAL_PERCENTAGE = 0.1

DATABASE1_SOURCE = ['global-bird-zenodo_poland', 'global-bird-zenodo_palmyra', 'global-bird-zenodo_penguins',
                    'global-bird-zenodo_mckellar', 'global-bird-zenodo_newmexico', 
                    'global-bird-zenodo_pfeifer', 'uav-waterfowl-thermal']


# ====== FUNCTIONS ======

def save_split_portion(split_set, split_set_img, dataset_folder, saving_folder, dataset_config):
    count_detections = 0
    nb_img_by_nb_birds = {}  # {0: 12, 1: 4} means 12 images contain 0 bird, 4 images contain 1 bird

    for img in split_set_img:
        img_name = img.split('.jpg')[0]
        img_path = os.path.join(dataset_folder, "images", img)
        label_path = os.path.join(dataset_folder, "labels", img_name + '.txt')
        save_name = dataset_config['name'] + '_' + img_name
        shutil.copyfile(img_path, os.path.join(saving_folder, split_set, "images", save_name + '.jpg'))
        if os.path.exists(label_path):
            shutil.copyfile(label_path, os.path.join(saving_folder, split_set, "labels", save_name + '.txt'))
            f = open(os.path.join(dataset_folder, "labels", img_name + '.txt'), "r")
            temp_count = len(f.readlines())
            count_detections += temp_count
            if temp_count not in nb_img_by_nb_birds:
                nb_img_by_nb_birds[temp_count] = 0
            nb_img_by_nb_birds[temp_count] += 1
        
        if split_set == 'test':
            if not os.path.exists(os.path.join(saving_folder, split_set, dataset_config['name'])):
                os.makedirs(os.path.join(saving_folder, split_set, dataset_config['name'], "images"))
                os.makedirs(os.path.join(saving_folder, split_set, dataset_config['name'], "labels"))
            shutil.copyfile(img_path, os.path.join(saving_folder, split_set, dataset_config['name'], "images", save_name + '.jpg'))
            if os.path.exists(label_path):
                shutil.copyfile(label_path, os.path.join(saving_folder, split_set, dataset_config['name'], "labels", save_name + '.txt'))
        
    return {"nb_img": len(test_img), "nb_birds": count_detections, "birds_repartition": nb_img_by_nb_birds} #count_detections, nb_img_by_nb_birds


# ====== TRAIN-TEST SPLIT ======

if not os.path.exists(SAVING_FOLDER):
    os.makedirs(os.path.join(SAVING_FOLDER, "train", "images"))
    os.makedirs(os.path.join(SAVING_FOLDER, "train", "labels"))
    os.makedirs(os.path.join(SAVING_FOLDER, "val", "images"))
    os.makedirs(os.path.join(SAVING_FOLDER, "val", "labels"))
    os.makedirs(os.path.join(SAVING_FOLDER, "test", "images"))
    os.makedirs(os.path.join(SAVING_FOLDER, "test", "labels"))

config = load_config(YAML_PATH)

# For dataset stats storing
data = {}

# Treat dataset one by one
for dataset in DATABASE1_SOURCE:

    # Extract specific dataset config
    dataset_config = extract_dataset_config(config, dataset)
    print("dataset: ", dataset)
    print(dataset_config)

    data_temp = {'name': dataset_config['name']}

    dataset_folder = os.path.join(SOURCE_FOLDER, dataset_config["name"])
    original_dataset_folder = os.path.join(ORIGINAL_FOLDER, dataset_config["name"])

    available_img = os.listdir(os.path.join(dataset_folder, "images"))
    nb_img = len(available_img)
    train_count = round(TRAIN_PERCENTAGE*nb_img)
    test_count = round(TEST_PERCENTAGE*nb_img)
    val_count = round(VAL_PERCENTAGE*nb_img)

    test_img = []
    val_img = []
    train_img = []

    if dataset_config['test_val_split']:
        #if dataset == 'uav-waterfowl-thermal':
         #   for path in dataset_config['test_val_split']['val']:
          #      test_img.extend(img for img in available_img if img.startswith(path))
           # for path in dataset_config['test_val_split']['test']:
            #    val_img.extend(img for img in available_img if img.startswith(path))
        #else:
        for path in dataset_config['test_val_split']['test']:
            test_img.extend(img for img in available_img if img.startswith(path))
        for path in dataset_config['test_val_split']['val']:
            val_img.extend(img for img in available_img if img.startswith(path))
        
        train_img = [img for img in available_img if img not in test_img and img not in val_img]

    else:
        if len(glob.glob(original_dataset_folder + '/**/*.csv', recursive=True)) == 2:
            df_test = pd.read_csv([fn for fn in glob.glob(original_dataset_folder + '/**/*.csv', recursive=True) if 'test' in fn][0])
            val_test_img = df_test[dataset_config['annotation_col_names'][0]].unique()
            val_img_temp = [os.path.splitext(img)[0] for img in val_test_img[0:round(len(val_test_img)/3)]]
            test_img_temp = [os.path.splitext(img)[0] for img in val_test_img[round(len(val_test_img)/3):]]

            val_img.extend([string_B for string_A in val_img_temp for string_B in available_img if string_B.startswith(string_A + '_patch_')])
            test_img.extend([string_B for string_A in test_img_temp for string_B in available_img if string_B.startswith(string_A + '_patch_')])
            train_img.extend([img for img in available_img if img not in test_img and img not in val_img])

        else:
            df_ = pd.read_csv(os.path.join(original_dataset_folder, glob.glob(original_dataset_folder + '/**/*.csv', recursive=True)[0]))
            all_old_img = df_[dataset_config['annotation_col_names'][0]].unique()
            train_img_temp = [os.path.splitext(img)[0] for img in all_old_img[0:round(TRAIN_PERCENTAGE*len(all_old_img))]]
            test_img_temp = [os.path.splitext(img)[0] for img in all_old_img[round(TRAIN_PERCENTAGE*len(all_old_img)):round(TRAIN_PERCENTAGE*len(all_old_img))+round(TEST_PERCENTAGE*len(all_old_img))]]
            val_img_temp = [os.path.splitext(img)[0] for img in all_old_img[round(TRAIN_PERCENTAGE*len(all_old_img))+round(TEST_PERCENTAGE*len(all_old_img)):]]

            train_img.extend([string_B for string_A in train_img_temp for string_B in available_img if string_B.startswith(string_A + '_patch_')])
            val_img.extend([string_B for string_A in val_img_temp for string_B in available_img if string_B.startswith(string_A + '_patch_')])
            test_img.extend([string_B for string_A in test_img_temp for string_B in available_img if string_B.startswith(string_A + '_patch_')])


    #count, nb_img_by_nb_birds = save_split_portion("test", test_img, dataset_folder, SAVING_FOLDER, dataset_config)
    data_temp["Test"] = save_split_portion("test", test_img, dataset_folder, SAVING_FOLDER, dataset_config)
    #data_temp["Test"] = {"nb_img": len(test_img), "nb_birds": count, "birds_repartition": nb_img_by_nb_birds}

    data_temp["Val"] = save_split_portion("val", val_img, dataset_folder, SAVING_FOLDER, dataset_config)
    #data_temp["Val"] = {"nb_img": len(val_img), "nb_birds": count, "birds_repartition": nb_img_by_nb_birds}

    data_temp["Train"] = save_split_portion("train", train_img, dataset_folder, SAVING_FOLDER, dataset_config)
    #data_temp["Train"] = {"nb_img": len(train_img), "nb_birds": count, "birds_repartition": nb_img_by_nb_birds}

    print("DONE, next dataset")
    data[dataset] = data_temp
    # for each dataset


# Store dataset stats
fname = 'data_stats.yaml'
with open(os.path.join(SAVING_FOLDER, fname), 'w') as yaml_file:
    yaml_file.write( yaml.dump(data, default_flow_style=False))