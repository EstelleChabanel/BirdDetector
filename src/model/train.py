#import ultralytics
#ultralytics.checks()
#from ultralytics import YOLO

import yolo
from yolo import YOLO

from PIL import Image
import torch
import yaml

import os
import random
import pandas as pd
import sys

module_path = os.path.abspath(os.path.join('..'))
print(module_path)
module_path = module_path+'/data_preprocessing'
print(module_path)

if module_path not in sys.path:
    sys.path.append(module_path)

import src.data_preprocessing.visualization_utils as visutils


device = "0" if torch.cuda.is_available() else "cpu"
if device == "0":
    torch.cuda.set_device(0) # Set to your desired GPU number


# ============== PARAMETERS ==============

# TODO: can keep parameters in dictionary of corresponding parameters, + simple et - d'erreurs

# Pretrained model weights
PRETRAINED = False
PRETRAINED_MODEL_NAME = 'pfeifer_penguins_poland_palmyra_10percent_bckgd_yolov8m_100epochs'  #'pfeifer_penguins_poland_10percentbkgd_yolov8m_120epochs'
PRETRAINED_MODEL_PATH = 'runs/detect/' + PRETRAINED_MODEL_NAME + '/weights/best.pt' # 'runs/detect/' + PRETRAINED_MODEL_NAME + '/weights/best.pt' #

# Model specifications
SUBTASK = 'domainclassifier' # Choose between: 'detect', 'domainclassifier' 
MODEL_NAME = 'DAN_test_GRL' #'DAN_domainclassifier_test_GRL'
MODEL_PATH = 'runs/detect/' + MODEL_NAME + '/weights/best.pt'

# Data
DATASET_NAME = 'pfpepo_palmyra_10percentbkgd'
DATASET_PATH = '/gpfs/gibbs/project/jetz/eec42/data/' + DATASET_NAME
DATASETS = ['global_birds_pfeifer', 'global_birds_penguins', 'global_birds_poland', 'global_birds_palmyra'] #['source', 'target'] #['global_birds_pfeifer', 'global_birds_penguins', 'global_birds_poland', 'global_birds_palmyra']

# For training
NB_EPOCHS = 2
BATCH_SIZE = 32
PATIENCE = 30
OPTIMIZER = 'Adam' # choices=[SGD, Adam, Adamax, AdamW, NAdam, RAdam, RMSProp, auto]

# For predictions
IOU_THRESHOLD = 0.1
CONF_THRESHOLD = 0.1


# ============== Load model & TRAIN it ==============

# Upload config file
fname = "src/model/data.yaml"
stream = open(fname, 'r')
data = yaml.safe_load(stream)
data['path'] = DATASET_PATH
with open(fname, 'w') as yaml_file:
    yaml_file.write(yaml.dump(data, default_flow_style=False))
img_path = os.path.join(data['path'], data['test'])


# Load model

if PRETRAINED:
    model = YOLO(PRETRAINED_MODEL_PATH, task='detect', subtask=SUBTASK)
else:
    model = YOLO('yolov8m_domainclassifier.yaml', task='detect', subtask=SUBTASK).load("yolov8m.pt")

#model = YOLO('yolov8m.yaml', task='detect') #, subtask=SUBTASK).load("yolov8m.pt")
#model = YOLO('yolov8m_domainclassifier.yaml', task='detect', subtask=SUBTASK).load(MODEL_PATH)
print(model.task, model.subtask)


# Train model
results = model.train(
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
   lr0=0.001, # default=0.01, (i.e. SGD=1E-2, Adam=1E-3)
   lrf=0.01, # default=0.01, final learning rate (lr0 * lrf)
   #dropout=0.3,
   degrees=90, fliplr=0.5, flipud=0.5, scale=0.5, # augmentation parameters
   name=MODEL_NAME)


# ============== PREDICT on test set and visualize results ==============

# Create subfolder to store examples
SAVE_EXAMPLES_PATH = os.path.join('runs/detect/' + model.name, 'predictions')
os.mkdir(SAVE_EXAMPLES_PATH)

# Select randomly k images from the test dataset
selected_img = []
for subdataset in DATASETS:
    selected_img.extend(random.choices(os.listdir(img_path + subdataset + '/images/'), k=10))


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
for img, result in zip(selected_img, results):

    detection_boxes = []
    #save_path = '/vast/palmer/home.grace/eec42/BirdDetector/src/model/runs/detect/' + MODEL_NAME + '/prediction_' + os.path.basename(result.path).split('.jpg')[0] + '.jpg'
    #save_path = 'runs/detect/' + MODEL_NAME + '/prediction_' + os.path.basename(result.path).split('.jpg')[0] + '.jpg'        
    # Retrieve detection predictions 
    for detect in range(len(result.boxes.cls)):
        det = {}
        det['conf'] = result.boxes.conf[detect].cpu()
        det['category'] = result.boxes.cls[detect].cpu()
        coords = result.boxes.xywhn[detect].cpu()
        det['bbox'] = [coords[0]-coords[2]/2, coords[1]-coords[3]/2, coords[2], coords[3]]
        detection_boxes.append(det)

    # Draw predictions on images   
    im_path_ = os.path.join(img_path + 'images/', img)
    save_path = SAVE_EXAMPLES_PATH + '/prediction_' + os.path.basename(result.path)  
    visutils.draw_bounding_boxes_on_file(im_path_, save_path, detection_boxes,
                                    confidence_threshold=0.0, detector_label_map=None,
                                    thickness=1,expansion=0, colormap=['Red'])

    # Retrieve detection groundtruths 
    selected_label = img_path + 'labels/' + os.path.basename(result.path).split('.jpg')[0] + '.txt'
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
        #save_path2 = '/vast/palmer/home.grace/eec42/BirdDetector/src/model/runs/detect/' + MODEL_NAME + '/prediction_label_' + os.path.basename(result.path).split('.hpg')[0] + '.jpg'
        #save_path2 = 'runs/detect/' + MODEL_NAME + '/prediction_label_' + os.path.basename(result.path).split('.hpg')[0] + '.jpg'
        save_path2 = SAVE_EXAMPLES_PATH + '/' + os.path.basename(result.path)
        visutils.draw_bounding_boxes_on_file(save_path, save_path2, detection_boxes,
                                        confidence_threshold=0.0, detector_label_map=None,
                                        thickness=1,expansion=0, colormap=['SpringGreen'])
                                        
    # Remove predictions-only images
    os.remove(save_path)

print("Prediction visualizations saved")