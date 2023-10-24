import os
import glob
import pandas as pd
import shutil
from PIL import Image
import shutil

from reformatting_utils import load_config, extract_dataset_config


def save_split_portion(split_set, split_set_img, dataset_folder, saving_folder, dataset_config):
    count_detections = 0

    for img in split_set_img:
        img_name = img.split(dataset_config["image_extension"])[0]
        shutil.copyfile(os.path.join(dataset_folder, "images", img), os.path.join(saving_folder, split_set, "images", img))
        if os.path.exists(os.path.join(dataset_folder, "labels", img_name + '.txt')):
            shutil.copyfile(os.path.join(dataset_folder, "labels", img_name + '.txt'), os.path.join(saving_folder, split_set, "labels", img_name + '.txt'))
            f = open(os.path.join(dataset_folder, "labels", img_name + '.txt'), "r")
            count_detections += len(f.readlines())     
    return count_detections

original_folder = r'/gpfs/gibbs/project/jetz/eec42/data/original'
source_folder = r'/gpfs/gibbs/project/jetz/eec42/data/formatted_data'
saving_folder = r'/gpfs/gibbs/project/jetz/eec42/data/baseline1'

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

yaml_path = r'/home/eec42/BirdDetector/src/data_preprocessing/source_datasets_config.yaml'
config = load_config(yaml_path)

train_percentage = 0.7
test_percentage = 0.2
val_percentage = 0.1

database1_source = ['global-bird-zenodo_poland', 'global-bird-zenodo_palmyra', 'global-bird-zenodo_penguins',
                    'global-bird-zenodo_mckellar', 'global-bird-zenodo_newmexico', 
                    'global-bird-zenodo_pfeifer', 'uav-waterfowl-thermal']

metadata = open(saving_folder +'/data_stats.txt', 'a')

for dataset in database1_source:

    # Extract specific dataset config
    dataset_config = extract_dataset_config(config, dataset)
    print("dataset: ", dataset)
    print(dataset_config)
    print("name: ", dataset_config['name'])

    metadata.write("Dataset: " + repr(dataset) + "\n \n")

    dataset_folder = os.path.join(source_folder, dataset_config["name"])
    original_dataset_folder = os.path.join(original_folder, dataset_config["name"])

    available_img = os.listdir(os.path.join(dataset_folder, "images"))
    nb_img = len(available_img)
    print("nb of images: ", nb_img)
    train_count = round(train_percentage*nb_img)
    test_count = round(test_percentage*nb_img)
    val_count = round(val_percentage*nb_img)

    test_img = []
    val_img = []
    train_img = []

    if dataset_config['test_val_split']:
        for path in dataset_config['test_val_split']['test']:
            test_img.extend(img for img in available_img if img.startswith(path))
        for path in dataset_config['test_val_split']['val']:
            val_img.extend(img for img in available_img if img.startswith(path))
        
        train_img = [img for img in available_img if img not in test_img and img not in val_img]

    else:

        if len(glob.glob(original_dataset_folder + '/**/*.csv', recursive=True)) == 2:
            df_test = pd.read_csv([fn for fn in glob.glob(original_dataset_folder + '/**/*.csv', recursive=True) if 'test' in fn][0])
            val_test_img = df_test[dataset_config['annotation_col_names'][0]]
            val_img_temp = [os.path.splitext(img)[0] for img in val_test_img[0:round(len(val_test_img)/3)]]
            test_img_temp = [os.path.splitext(img)[0] for img in val_test_img[round(len(val_test_img)/3):]]

            val_img.extend([string_B for string_A in val_img_temp for string_B in available_img if string_B.startswith(string_A)])
            test_img.extend([string_B for string_A in test_img_temp for string_B in available_img if string_B.startswith(string_A)])
            train_img.extend([img for img in available_img if img not in test_img and img not in val_img])

        else:
            df_ = pd.read_csv(os.path.join(original_dataset_folder, glob.glob(original_dataset_folder + '/**/*.csv', recursive=True)[0]))
            all_old_img = df_[dataset_config['annotation_col_names'][0]]
            train_img_temp = [os.path.splitext(img)[0] for img in all_old_img[0:round(train_percentage*len(all_old_img))]]
            test_img_temp = [os.path.splitext(img)[0] for img in all_old_img[round(train_percentage*len(all_old_img)):round(train_percentage*len(all_old_img))+round(test_percentage*len(all_old_img))]]
            val_img_temp = [os.path.splitext(img)[0] for img in all_old_img[round(train_percentage*len(all_old_img))+round(test_percentage*len(all_old_img)):]]

            train_img.extend([string_B for string_A in train_img_temp for string_B in available_img if string_B.startswith(string_A)])
            val_img.extend([string_B for string_A in val_img_temp for string_B in available_img if string_B.startswith(string_A)])
            test_img.extend([string_B for string_A in test_img_temp for string_B in available_img if string_B.startswith(string_A)])


    count = save_split_portion("test", test_img, dataset_folder, saving_folder, dataset_config)
    metadata.write("Testing set: " + repr(len(test_img)) + " images \n")
    metadata.write("                " + repr(count) + " birds annotated \n")

    count = save_split_portion("val", val_img, dataset_folder, saving_folder, dataset_config)
    metadata.write("Validation set: " + repr(len(val_img)) + " images \n")
    metadata.write("                " + repr(count) + " birds annotated \n")

    count = save_split_portion("train", train_img, dataset_folder, saving_folder, dataset_config)
    metadata.write("Testing set: " + repr(len(train_img)) + " images \n")
    metadata.write("                " + repr(count) + " birds annotated \n")
    
    metadata.write("\n \n")
    print("DONE, next dataset")
    # for each dataset
