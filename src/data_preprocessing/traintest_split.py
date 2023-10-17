import os
import glob
import pandas as pd
import shutil
from PIL import Image

from reformatting_utils import load_config, extract_dataset_config


def save_split_portion(split_set, split_set_img, subdataset_folder, saving_folder, dataset_config, resize=False):
    count_detections = 0

    for img in split_set_img:
        img_name = img.split(dataset_config["image_extension"])[0]
        if resize = False:
            shutil.copyfile(os.path.join(subdataset_folder, "image", img), os.path.join(saving_folder, split_set, "image", img))
            shutil.copyfile(os.path.join(subdataset_folder, "label", img_name + '.txt'), os.path.join(saving_folder, split_set, "label", img_name + '.txt'))
        else:
            #resize image and label before saving them
        f = open(os.path.join(subdataset_folder, "label", img_name + '.txt'), "r")
        count_detections += len(f.readlines())
    
    return count_detections


def resize_img():
    return


source_folder = r'/gpfs/gibbs/project/jetz/eec42/data/formatted_data'
saving_folder = r'/gpfs/gibbs/project/jetz/eec42/data/baseline1'

if not os.path.exists(saving_folder):
    os.mkdir(saving_folder)
    os.mkdir(os.path.join(saving_folder, "train"))
    os.mkdir(os.path.join(saving_folder, "train", "image"))
    os.mkdir(os.path.join(saving_folder, "train", "label"))
    os.mkdir(os.path.join(saving_folder, "val"))
    os.mkdir(os.path.join(saving_folder, "val", "image"))
    os.mkdir(os.path.join(saving_folder, "val", "label"))
    os.mkdir(os.path.join(saving_folder, "test"))
    os.mkdir(os.path.join(saving_folder, "test", "image"))
    os.mkdir(os.path.join(saving_folder, "test", "label"))

yaml_path = r'/home/eec42/BirdDetector/src/data_preprocessing/source_datasets_config.yaml'
config = load_config(yaml_path)

train_percentage = 0.7
test_percentage = 0.2
val_percentage = 0.1

database1_source = ['global-bird-zenodo']

metadata = open(saving_folder +'/data_stats.txt', 'a')
metadata.write("\n \n \n")

for dataset in database1_source:

    metadata.write("Dataset: " + repr(dataset) + "\n")

    # Extract specific dataset config
    dataset_config = extract_dataset_config(config, dataset)
    print(dataset_config)

    # Extract list of subdatasets
    dataset_folder = os.path.join(source_folder, dataset_config["name"])
    subdatasets = os.listdir(dataset_folder)
    subdatasets = [fn for fn in subdatasets if os.path.isdir(os.path.join(dataset_folder, fn))]

    for subdataset in subdatasets:

        metadata.write("Subdataset: " + repr(subdataset) + "\n")
        subdataset_folder = os.path.join(dataset_folder, subdataset)

        available_img = os.listdir(os.path.join(subdataset_folder, "image"))

        nb_img = len(available_img)
        train_count = round(train_percentage*nb_img)
        test_count = round(test_percentage*nb_img)
        val_count = round(val_percentage*nb_img)

        if dataset == "global-bird-zenodo":
            original_dataset = os.path.join(r'/gpfs/gibbs/project/jetz/eec42/data/original', dataset_config["name"], subdataset)
            print(original_dataset)

            if len(glob.glob(original_dataset + '/**/*.csv', recursive=True)) != 2:
                train_img = available_img[0:train_count]
                test_img = available_img[train_count+1:train_count+test_count+1]
                val_img = available_img[train_count+test_count+2:train_count+test_count+2+val_count]

            else:
                train_annotations = pd.read_csv([fn for fn in glob.glob(original_dataset + '/**/*.csv', recursive=True) if 'train' in fn][0])
                train_test_img = list(set(train_annotations["image_path"]))
                train_img = train_test_img[0:train_count]
                test_img = train_test_img[train_count+1: train_count+test_count+1]

                val_annotations = pd.read_csv([fn for fn in glob.glob(original_dataset + '/**/*.csv', recursive=True) if 'test' in fn][0])
                val_img = list(set(val_annotations["image_path"]))
            
            # Resize images&labels and save tem in correct folder
            if not ((Image.open(os.path.join(subdataset_folder, "image", available_img[0])).size[0] % 32 == 0) and (Image.open(os.path.join(subdataset_folder, "image", available_img[0])).size[1] % 32 == 0)):
                resize = True
            count = save_split_portion("train", train_img, subdataset_folder, saving_folder, dataset_config, resize)
            metadata.write("Training set: " + repr(len(train_img)) + " images \n")
            metadata.write("                " + repr(count) + " birds annotated \n")

            count = save_split_portion("test", test_img, subdataset_folder, saving_folder, dataset_config)
            metadata.write("Testing set: " + repr(len(test_img)) + " images \n")
            metadata.write("                " + repr(count) + " birds annotated \n")

            count = save_split_portion("val", val_img, subdataset_folder, saving_folder, dataset_config)
            metadata.write("Validation set: " + repr(len(val_img)) + " images \n")
            metadata.write("                " + repr(count) + " birds annotated \n \n")

        else:
            print("TBD")



