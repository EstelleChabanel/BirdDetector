import os
import glob
import pandas as pd
import shutil

from reformatting_utils import load_config, extract_dataset_config

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

metadata = open(saving_folder +'/data_stats.txt', 'a')
metadata.write("\n \n \n")

for dataset in config.keys():

    if not dataset=='global-bird-zenodo':
        continue

    metadata.write("Dataset: " + repr(dataset) + "\n")

    # Extract specific dataset config
    dataset_config = extract_dataset_config(config, dataset)
    print(dataset_config)

    dataset_folder = os.path.join(source_folder, dataset_config["name"])
    subdatasets = os.listdir(dataset_folder)
    subdatasets = [fn for fn in subdatasets if os.path.isdir(os.path.join(dataset_folder, fn))]

    for subdataset in subdatasets:

        metadata.write("Subdataset: " + repr(subdataset) + "\n")
        subdataset_folder = os.path.join(dataset_folder, subdataset)

        available_img = os.listdir(os.path.join(subdataset_folder, "image"))
        available_img_names = list(set([os.path.splitext(s)[0] for s in available_img]))

        nb_img = len(available_img_names)
        train_count = round(train_percentage*nb_img)
        test_count = round(test_percentage*nb_img)
        val_count = round(val_percentage*nb_img)

        if dataset == "global-bird-zenodo":
            original_dataset = os.path.join(r'/gpfs/gibbs/project/jetz/eec42/data/original', dataset_config["name"], subdataset)
            print(original_dataset)

            if len(glob.glob(original_dataset + '/**/*.csv', recursive=True)) != 2:
                count_detections = 0
                train_img = available_img[0:train_count]
                for img in train_img:
                    img_name = img.split(dataset_config["image_extension"])[0]
                    shutil.copyfile(os.path.join(subdataset_folder, "image", img), os.path.join(saving_folder, "train", "image", img))
                    shutil.copyfile(os.path.join(subdataset_folder, "label", img_name + '.txt'), os.path.join(saving_folder, "train", "label", img_name + '.txt'))
                    f = open(os.path.join(subdataset_folder, "label", img_name + '.txt'), "r")
                    count_detections += len(f.readlines())
                metadata.write("Training set: " + repr(len(train_img)) + " images \n")
                metadata.write("                " + repr(count_detections) + " birds annotated \n")

                count_detections = 0
                test_img = available_img[train_count+1:train_count+test_count+1]
                for img in test_img:
                    img_name = img.split(dataset_config["image_extension"])[0]
                    shutil.copyfile(os.path.join(subdataset_folder, "image", img), os.path.join(saving_folder, "test", "image", img))
                    shutil.copyfile(os.path.join(subdataset_folder, "label", img_name + '.txt'), os.path.join(saving_folder, "test", "label", img_name + '.txt'))
                    f = open(os.path.join(subdataset_folder, "label", img_name + '.txt'), "r")
                    count_detections += len(f.readlines())
                metadata.write("Training set: " + repr(len(test_img)) + " images \n")
                metadata.write("                " + repr(count_detections) + " birds annotated \n")

                count_detections = 0
                val_img = available_img[train_count+test_count+2:train_count+test_count+2+val_count]
                for img in val_img:
                    img_name = img.split(dataset_config["image_extension"])[0]
                    shutil.copyfile(os.path.join(subdataset_folder, "image", img), os.path.join(saving_folder, "val", "image", img))
                    shutil.copyfile(os.path.join(subdataset_folder, "label", img_name + '.txt'), os.path.join(saving_folder, "val", "label", img_name + '.txt'))
                    f = open(os.path.join(subdataset_folder, "label", img_name + '.txt'), "r")
                    count_detections += len(f.readlines())
                metadata.write("Training set: " + repr(len(val_img)) + " images \n")
                metadata.write("                " + repr(count_detections) + " birds annotated \n \n")


            else:
                train_annotations = pd.read_csv([fn for fn in glob.glob(original_dataset + '/**/*.csv', recursive=True) if 'train' in fn][0])

                train_test_img = list(set(train_annotations["image_path"]))
                train_img = train_test_img[0:train_count]
                test_img = train_test_img[train_count+1: train_count+test_count+1]

                count_detections = 0
                for img in train_img:
                    img_name = img.split(dataset_config["image_extension"])[0]
                    shutil.copyfile(os.path.join(subdataset_folder, "image", img), os.path.join(saving_folder, "train", "image", img))
                    shutil.copyfile(os.path.join(subdataset_folder, "label", img_name + '.txt'), os.path.join(saving_folder, "train", "label", img_name + '.txt'))
                    f = open(os.path.join(subdataset_folder, "label", img_name + '.txt'), "r")
                    count_detections += len(f.readlines())
                metadata.write("Training set: " + repr(len(train_img)) + " images \n")
                metadata.write("                " + repr(count_detections) + " birds annotated \n")

                count_detections = 0
                for img in test_img:
                    img_name = img.split(dataset_config["image_extension"])[0]
                    shutil.copyfile(os.path.join(subdataset_folder, "image", img), os.path.join(saving_folder, "test", "image", img))
                    shutil.copyfile(os.path.join(subdataset_folder, "label", img_name + '.txt'), os.path.join(saving_folder, "test", "label", img_name + '.txt'))
                    f = open(os.path.join(subdataset_folder, "label", img_name + '.txt'), "r")
                    count_detections += len(f.readlines())
                metadata.write("Testing set: " + repr(len(test_img)) + " images \n")
                metadata.write("                " + repr(count_detections) + " birds annotated \n")

                val_annotations = pd.read_csv([fn for fn in glob.glob(original_dataset + '/**/*.csv', recursive=True) if 'test' in fn][0])
                val_img = list(set(val_annotations["image_path"]))
                val_img_names = [img.split(dataset_config["image_extension"]) for img in val_img]

                for img in val_img:
                    img_name = img.split(dataset_config["image_extension"])[0]
                    shutil.copyfile(os.path.join(subdataset_folder, "image", img), os.path.join(saving_folder, "val", "image", img))
                    shutil.copyfile(os.path.join(subdataset_folder, "label", img_name + '.txt'), os.path.join(saving_folder, "val", "label", img_name + '.txt'))
                metadata.write("Validation set: " + repr(len(val_img)) + " images \n")
                metadata.write("                " + repr(len(val_annotations)) + " birds annotated \n \n")
                

        else:
            print("TBD")



