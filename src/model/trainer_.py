#import ultralytics
#ultralytics.checks()
#from ultralytics import YOLO
import yolo
from yolo import YOLO
import torch
import yaml
import os
import random
import pandas as pd
import sys
import argparse

module_path = os.path.abspath(os.path.join('..'))
print(module_path)
module_path = module_path+'/data_preprocessing'
print(module_path)

if module_path not in sys.path:
    sys.path.append(module_path)

import src.data_preprocessing.visualization_utils as visutils
from constants import DATA_PATH, DATASETS_MAPPING, MODELS_PATH, NB_EPOCHS, BATCH_SIZE, PATIENCE, OPTIMIZER, TRAINING_IOU_THRESHOLD, CONF_THRESHOLD, IOU_THRESHOLD


device = "0" if torch.cuda.is_available() else "cpu"
if device == "0":
    torch.cuda.set_device(0) # Set to your desired GPU number


# ============ Parse Arguments ============ #
parser = argparse.ArgumentParser()
parser.add_argument("--model-name", type=str, required=True)
parser.add_argument("--subtask", type=str, required=True)
parser.add_argument("--dataset-name", type=str, required=True)
parser.add_argument("--lr", type=float)
parser.add_argument("--dcloss-gain", type=float)
args = parser.parse_args()


# ============ Initialize parameters ============ #

def upload_data_cfg(dataset_name):
    fname = "src/model/data.yaml"
    stream = open(fname, 'r')
    data = yaml.safe_load(stream)
    data['path'] = DATA_PATH + dataset_name
    with open(fname, 'w') as yaml_file:
        yaml_file.write(yaml.dump(data, default_flow_style=False))
    img_path = os.path.join(data['path'], data['test'])
    return img_path


# ============== Functions - to put in a utils ============== #

def train_model(model, args):
    model.train(
        data='src/model/data.yaml',
        #imgsz=480,  # we are trying with several img size so we do not precise the size -> will automatically resize all images to 640x640
        epochs=NB_EPOCHS,
        patience=PATIENCE,
        batch=BATCH_SIZE,
        device=0,
        optimizer=OPTIMIZER,
        verbose=True,
        val=True,
        #cos_lr=True,
        lr0=args.lr, # default=0.01, (i.e. SGD=1E-2, Adam=1E-3)
        lrf=0.01, # default=0.01, final learning rate (lr0 * lrf)
        #dropout=0.3,
        dc = args.dcloss_gain,
        iou=TRAINING_IOU_THRESHOLD,
        #augment=False,
        amp=True,
        single_cls=True,
        degrees=90, fliplr=0.5, flipud=0.5, scale=0.5, # augmentation parameters
        hsv_h=0.00, hsv_s=0.0, hsv_v=0.0, translate=0.0, shear=0.0, perspective=0.0, mosaic=0.0, mixup=0.0,
        name=args.model_name)
    return


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
    im_path_ = os.path.join(im_path + 'images/', img)
    save_path = saving_path + '/prediction_' + os.path.basename(result.path)  
    visutils.draw_bounding_boxes_on_file(im_path_, save_path, detection_boxes,
                                    confidence_threshold=0.0, detector_label_map=None,
                                    thickness=1,expansion=0, colormap=['Red'])

    # Retrieve detection groundtruths 
    selected_label = im_path + 'labels/' + os.path.basename(result.path).split('.jpg')[0] + '.txt'
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
                                        thickness=1, expansion=0, colormap=['SpringGreen'])
                                        
    # Remove predictions-only images
    os.remove(save_path)


def visualize_predictions(model, datasets, img_path, saving_path, k=5):
    # Select randomly k images from the test dataset
    selected_img = []
    for subdataset in datasets:
        selected_img.extend(random.choices(os.listdir(img_path + subdataset + '/images/'), k=5))

    # Predict results for randomly selected images
    results = model.predict(
            #model = 'runs/detect/pfeifer_yolov8n_70epoch_default_batch32_dropout0.3',
            source = [os.path.join(img_path + 'images/', img) for img in selected_img],
            conf = CONF_THRESHOLD, 
            iou = IOU_THRESHOLD,
            show = False,
            save = False
        )
    
    # Visualize predictions
    for img_, result_ in zip(selected_img, results):
        visualize_one_prediction(img_, result_, img_path, saving_path)
        
    print("Prediction visualizations saved")
    


# ============== Load & train model, Visualize predictions ==============
    
# Upload data config file
IMG_PATH = upload_data_cfg(args.dataset_name)

# Load model
model = YOLO('yolov8m_domainclassifier.yaml', task='detect', subtask=args.subtask).load("yolov8m.pt")
print(model.task, model.subtask)

# Train model
train_model(model, args)
torch.cuda.empty_cache()


# Create subfolder to store examples
SAVE_EXAMPLES_PATH = os.path.join(MODELS_PATH + args.model_name, 'predictions')
os.mkdir(SAVE_EXAMPLES_PATH)

# Predict on k images and visualize results
results = visualize_predictions(model, DATASETS_MAPPING[args.dataset_name], IMG_PATH, SAVE_EXAMPLES_PATH, k=5)
