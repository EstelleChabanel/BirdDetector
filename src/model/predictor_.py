#import ultralytics
#ultralytics.checks()
#from ultralytics import YOLO

import yolo
from yolo import YOLO

import torch
import os
import yaml
import pandas as pd
import sys
import argparse
import json
import random

module_path = os.path.abspath(os.path.join('..'))
print(module_path)
module_path = module_path+'/data_preprocessing'
print(module_path)

if module_path not in sys.path:
    sys.path.append(module_path)

import src.data_preprocessing.visualization_utils as visutils
from src.model.constants import DATA_PATH, DATASETS_MAPPING, MODELS_PATH, NB_EPOCHS, BATCH_SIZE, PATIENCE, OPTIMIZER, TRAINING_IOU_THRESHOLD, CONF_THRESHOLD, NMS_IOU_THRESHOLD, DEFAULT_LOSS_GAIN, DEFAULT_PARAM_SET, EXAMPLES_IMG


module_path = os.path.abspath(os.path.join('..'))
module_path = module_path+'/data_preprocessing'

if module_path not in sys.path:
    sys.path.append(module_path)

#import src.data_preprocessing.visualization_utils as visutils

# Use GPU if available
device = "0" if torch.cuda.is_available() else "cpu"
if device == "0":
    torch.cuda.set_device(0) # Set to your desired GPU number


# ============ Parse Arguments ============ #
    
parser = argparse.ArgumentParser()
parser.add_argument("--model-name", type=str, required=True)
parser.add_argument("--subtask", type=str, required=True)
parser.add_argument("--dataset-name", type=str, required=True)
parser.add_argument("--iou", type=float, required=False)
args = parser.parse_args()

if args.iou:
    NMS_IOU_THRESHOLD = args.iou

# ============== INITIALIZE PARAMETERS ============== #
    

# Model specifications
MODEL_NAME = args.model_name #'DAN_pfpe_palm_Adam1e-3_dcLoss1' #'deepcoral_background_lscale16_epochs40_coralgain10' #'pfeifer_penguins_poland_palmyra_10percent_bckgd_yolov8m_120epochs'
SUBTASK = args.subtask #'domainclassifier' #Choose between: #'deepcoral_detect' #'detect'

# Data
DATASET_NAME = args.dataset_name 

eps = 1e-8

# ====== Load model & prepare evaluation ====== #

TASK = 'detect'
MODEL_PATH = MODELS_PATH + MODEL_NAME + '/weights/best.pt'
IMG_PATH = DATA_PATH + DATASET_NAME + '/test/'
SAVE_DIR = os.path.join('/vast/palmer/home.grace/eec42/BirdDetector/runs/detect', MODEL_NAME, 'better_predictions_onbest')
if not os.path.exists(SAVE_DIR):
    os.mkdir(SAVE_DIR)

print("model_path of model to load", MODEL_PATH)
model = YOLO(MODEL_PATH, task=TASK, subtask=SUBTASK ) #.load(MODEL_PATH)



def visualize_one_prediction(img, result, im_path, saving_path):
    detection_boxes = []
    # Retrieve detection predictions 
    for detect in range(len(result.boxes.cls)):
        det = {}
        det['conf'] = result.boxes.conf[detect].cpu()
        det['category'] = result.boxes.cls[detect].cpu()
        coords = result.boxes.xywhn[detect].cpu()
        det['bbox'] = [coords[0]-coords[2]/2, coords[1]-coords[3]/2, coords[2], coords[3]]
        detection_boxes.append(det)

    # Draw predictions on images   
    im_path_ = os.path.join(im_path + '/images/', img)
    save_path = saving_path + '/prediction_' + os.path.basename(result.path)  
    visutils.draw_bounding_boxes_on_file(im_path_, save_path, detection_boxes,
                                    confidence_threshold=0.0, detector_label_map=None,
                                    thickness=3,expansion=0, colormap=['Red'])

    # Retrieve detection groundtruths 
    selected_label = im_path + '/labels/' + os.path.basename(result.path).split('.jpg')[0] + '.txt'
    if os.path.exists(selected_label):
        detection_boxes = []

        df = pd.read_csv(selected_label, sep='\t', header=None, index_col=False)
        for irow, row in df.iterrows():  
            det = {}
            det['conf'] = None
            det['category'] = row[0]
            det['bbox'] = [row[1]-row[3]/2, row[2]-row[4]/2, row[3], row[4]]
            detection_boxes.append(det)
    
        # Draw groundtruths on images
        save_path2 = saving_path + '/' + os.path.basename(result.path)
        visutils.draw_bounding_boxes_on_file(save_path, save_path2, detection_boxes,
                                        confidence_threshold=0.0, detector_label_map=None,
                                        thickness=3, expansion=0, colormap=['SpringGreen'])
                                        
    # Remove predictions-only images
    os.remove(save_path)


def visualize_predictions(model, datasets, img_path_, saving_path, k=8):
    # Select randomly k images from the test dataset
    #selected_img = []
    for subdataset in datasets:
        img_path = os.path.join(img_path_, subdataset)
        #selected_img = (random.choices(os.listdir(img_path + '/images/'), k=5))
        if subdataset in EXAMPLES_IMG.keys():
            selected_img = EXAMPLES_IMG[subdataset]
            selected_img.extend((random.choices(os.listdir(img_path + '/images/'), k=2)))
        else:
            selected_img = []
            selected_img.extend((random.choices(os.listdir(img_path + '/images/'), k=2)))
        print(selected_img)


        # Predict results for randomly selected images
        results = model.predict(
                #model = 'runs/detect/pfeifer_yolov8n_70epoch_default_batch32_dropout0.3',
                source = [os.path.join(img_path + '/images/', img) for img in selected_img],
                conf = CONF_THRESHOLD, 
                iou = NMS_IOU_THRESHOLD,
                show = False,
                save = False
            )
        
        # Visualize predictions
        for img_, result_ in zip(selected_img, results):
            visualize_one_prediction(img_, result_, img_path, saving_path)
        
    print("Prediction visualizations saved")
    


# Predict on k images and visualize results
print(args.dataset_name)
results = visualize_predictions(model, DATASETS_MAPPING[args.dataset_name]['datasets'], IMG_PATH, SAVE_DIR, k=8)