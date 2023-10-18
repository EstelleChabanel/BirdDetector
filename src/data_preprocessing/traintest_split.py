import os
import glob
import pandas as pd
import shutil
from PIL import Image
import shutil

from reformatting_utils import load_config, extract_dataset_config


def save_split_portion(split_set, split_set_img, subdataset_folder, saving_folder, dataset_config):
    count_detections = 0

    for img in split_set_img:
        img_name = img.split(dataset_config["image_extension"])[0]
        shutil.copyfile(os.path.join(subdataset_folder, "image", img), os.path.join(saving_folder, split_set, "images", img))
        if os.path.exists(os.path.join(subdataset_folder, "label", img_name + '.txt')):
            shutil.copyfile(os.path.join(subdataset_folder, "label", img_name + '.txt'), os.path.join(saving_folder, split_set, "labels", img_name + '.txt'))
            f = open(os.path.join(subdataset_folder, "label", img_name + '.txt'), "r")
            count_detections += len(f.readlines())     
    return count_detections

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

database1_source = ['global-bird-zenodo']

metadata = open(saving_folder +'/data_stats.txt', 'a')

for dataset in database1_source:

    metadata.write("Dataset: " + repr(dataset) + "\n \n")

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

        print(subdataset) 
        print(nb_img, " images")  
        print("desired train_count: ", train_count)
        print("desired test : ", test_count)
        print("desired val : ", val_count)

        if dataset == "global-bird-zenodo":
            original_dataset = os.path.join(r'/gpfs/gibbs/project/jetz/eec42/data/original', dataset_config["name"], subdataset)

            if len(glob.glob(original_dataset + '/**/*.csv', recursive=True)) != 2:
                train_img = available_img[0:train_count]
                test_img = available_img[train_count+1:train_count+test_count+1]
                val_img = available_img[train_count+test_count+2:train_count+test_count+2+val_count]

            else:
                original_test_annotations =  pd.read_csv([fn for fn in glob.glob(original_dataset + '/**/*.csv', recursive=True) if 'test' in fn][0])
                test_img = list(set(original_test_annotations["image_path"]))

                val_train_annotations = pd.read_csv([fn for fn in glob.glob(original_dataset + '/**/*.csv', recursive=True) if 'train' in fn][0])
                train_val_img = list(set(val_train_annotations["image_path"]))
                print("train_val_img list: ", len(train_val_img))

                if len(test_img) < test_count:
                    print("not enough test img")
                    test_img.extend(train_val_img[0:test_count-len(test_img)])
                    del train_val_img[0:test_count-len(test_img)]
                elif len(test_img) > test_count:
                    print("too much test img")
                    train_val_img.extend(test_img[test_count+1:])
                    test_img = test_img[0:test_count]
                else: 
                    print("prfect, go to train val")

                print("train_val_img list: ", len(train_val_img))
                train_img = train_val_img[0:train_count]
                val_img = train_val_img[train_count+1:]
                print(" final train count : ", len(train_img))
                print("final test : ", len(test_img))
                print("val : ", len(val_img))
            
            count = save_split_portion("train", train_img, subdataset_folder, saving_folder, dataset_config)
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

        metadata.write("\n \n")
        # for each subdataset

    metadata.write("\n \n \n")
    # for each dataset


