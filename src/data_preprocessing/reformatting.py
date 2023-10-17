import os
import json

from reformatting_utils import load_config, extract_dataset_config, from_ind_csv_to_labels, from_global_csv_to_labels, from_multiple_global_csv_to_labels, from_classes_csv_to_labels, from_global_json_to_csv, from_global_csv_for_tiff_to_labels, preview_few_images


data_folder = r'/gpfs/gibbs/project/jetz/eec42/data'
original_dataset_folder = 'original'
saving_dataset_folder = 'formatted_data'

if not os.path.exists(os.path.join(data_folder, saving_dataset_folder)):
    os.mkdir(os.path.join(data_folder, saving_dataset_folder))

yaml_path = r'/home/eec42/BirdDetector/src/data_preprocessing/source_datasets_config.yaml'

config = load_config(yaml_path)

# Dictionnary to store all classes and corresponding int id
category_name_to_id = {}


for dataset in config.keys():

    if not dataset=='global-bird-zenodo':
        continue

    # Extract specific dataset config
    dataset_config = extract_dataset_config(config, dataset)
    print(dataset_config)

    # Source dataset folder
    source_dataset_folder = os.path.join(
        data_folder, original_dataset_folder, dataset_config["name"])
    if not os.path.exists(source_dataset_folder):
        print("can't find the source dataset at following path", source_dataset_folder)
        continue

    # Create new folder to store current dataset
    dataset_folder = os.path.join(
        data_folder, saving_dataset_folder, dataset_config["name"])
    if not os.path.exists(dataset_folder):
        os.mkdir(dataset_folder)

    category_name_to_id = eval(dataset_config["formatting_method"])(
        dataset_config, source_dataset_folder, dataset_folder, category_name_to_id)
    preview_few_images(dataset_config, source_dataset_folder,
                       dataset_folder, category_name_to_id)

    print("DONE, next source")

# ...for each dataset=source

# Save metadata about annotations in this dataset
with open(data_folder + '/category_metadata.json', 'w') as f:
    json.dump(category_name_to_id, f)
