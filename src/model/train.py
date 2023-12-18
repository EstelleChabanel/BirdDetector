import ultralytics
ultralytics.checks()
from ultralytics import YOLO
from PIL import Image

import torch
import yaml

import os
import random
from ultralytics.utils.plotting import plot_labels
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


# ======= PARAMETERS =======

PRETRAINED = True
PRETRAINED_MODEL_NAME = 'pfeifer_penguins_poland_10percentbckgd_yolov8m_120epoch'
PRETRAINED_MODEL_PATH = 'src/model/runs/detect/' + PRETRAINED_MODEL_NAME + '/weights/best.pt'

TASK = 'deepcoral_detect' # Choose between: 'deepcoral_detect' 'detect'
MODEL_NAME = 'deepcoral_background_lscale16_epochs40_coralgain10'
MODEL_PATH = 'src/model/runs/' + TASK + '/' + PRETRAINED_MODEL_NAME + '/weights/best.pt'

NB_EPOCHS = 40
BATCH_SIZE = 16

DATASETS = ['source', 'target']
#['source', 'target'] #['global_birds_pfeifer', 'global_birds_penguins', 'global_birds_poland', 'global_birds_palmyra']

# Dataset config file
fname = "src/model/data.yaml"
stream = open(fname, 'r')
data = yaml.safe_load(stream)
img_path = data['path'] + '/test/'


# ======= Load model from pretrained weights & train it =======

if PRETRAINED:
    model = YOLO(PRETRAINED_MODEL_PATH, TASK)
else:
    model = YOLO('yolov8m.pt', TASK)

print(model.task)

results = model.train(
   data='src/model/data.yaml',
   #imgsz=480,  # we are trying with several img size so we do not precise the size -> will automatically resize all images to 640x640
   epochs=NB_EPOCHS,
   batch=BATCH_SIZE, #32,
   #cos_lr=True,
   #dropout=0.3,
   #optimizer='Adam',
   patience=50,
   device=0,
   verbose=True,
   val=True,
   #lr0=0.001,
   #lrf=0.0001,
   degrees=90, fliplr=0.5, flipud=0.5, scale=0.5, # augmentation parameters
   name=MODEL_NAME)


# ======= Predict on test set and visualize results =======


if TASK == 'detect':

    # Select randomly k images from the test dataset
    selected_img = []
    for subdataset in DATASETS:
        selected_img.extend(random.choices(os.listdir(img_path + subdataset + '/images/'), k=10))

    results = model.predict(
            #model = 'runs/detect/pfeifer_yolov8n_70epoch_default_batch32_dropout0.3',
            source = [os.path.join(img_path + 'images/', img) for img in selected_img],
            conf = 0.2, 
            iou = 0.1,
            show=False,
            save=False
        )

    for img, result in zip(selected_img, results):

        detection_boxes = []
        #save_path = '/vast/palmer/home.grace/eec42/BirdDetector/src/model/runs/detect/' + MODEL_NAME + '/prediction_' + os.path.basename(result.path).split('.jpg')[0] + '.jpg'
        save_path = 'runs/' + TASK + '/' + MODEL_NAME + '/prediction_' + os.path.basename(result.path).split('.jpg')[0] + '.jpg'        
        for detect in range(len(result.boxes.cls)):
            det = {}
            det['conf'] = result.boxes.conf[detect].cpu()
            det['category'] = result.boxes.cls[detect].cpu()
            coords = result.boxes.xywhn[detect].cpu()
            det['bbox'] = [coords[0]-coords[2]/2, coords[1]-coords[3]/2, coords[2], coords[3]]
            detection_boxes.append(det)
            
        im_path_ = os.path.join(img_path + 'images/', img)
        print("IMG PATH", img_path)
        visutils.draw_bounding_boxes_on_file(im_path_, save_path, detection_boxes,
                                        confidence_threshold=0.0, detector_label_map=None,
                                        thickness=1,expansion=0, colormap=['Red'])

        selected_label = img_path + 'labels/' + os.path.basename(result.path).split('.jpg')[0] + '.txt'
        print("LABEL PATH", selected_label)
        if os.path.exists(selected_label):
            detection_boxes = []
            df = pd.read_csv(selected_label, sep='\t', header=None, index_col=False)
            for irow, row in df.iterrows():  
                det = {}
                det['conf'] = None
                det['category'] = row[0]
                det['bbox'] = [row[1]-row[3]/2, row[2]-row[4]/2, row[3], row[4]]
                detection_boxes.append(det)
        
            # Draw annotations
            #save_path2 = '/vast/palmer/home.grace/eec42/BirdDetector/src/model/runs/detect/' + MODEL_NAME + '/prediction_label_' + os.path.basename(result.path).split('.hpg')[0] + '.jpg'
            save_path2 = 'runs/' + TASK + '/' + MODEL_NAME + '/prediction_label_' + os.path.basename(result.path).split('.hpg')[0] + '.jpg'
            visutils.draw_bounding_boxes_on_file(save_path, save_path2, detection_boxes,
                                            confidence_threshold=0.0, detector_label_map=None,
                                            thickness=1,expansion=0, colormap=['SpringGreen'])
                                            
            # Remove predictions-only images
            os.remove(save_path)
    

elif TASK == 'deepcoral_detect':

    # Select randomly k images from the test dataset
    for subdataset in DATASETS:
        selected_img = random.choices(os.listdir(img_path + subdataset + '/images/'), k=12)

        results = model.predict(
                #model = 'runs/detect/pfeifer_yolov8n_70epoch_default_batch32_dropout0.3',
                source = [os.path.join(img_path, subdataset + '/images/', img) for img in selected_img],
                conf = 0.2, 
                iou = 0.1,
                show=False,
                save=False
            )
        
        for img, result in zip(selected_img, results):

            detection_boxes = []
            save_path = 'runs/' + TASK + '/' + MODEL_NAME + '/prediction_' + os.path.basename(result.path).split('.jpg')[0] + '.jpg'
            for detect in range(len(result.boxes.cls)):
                det = {}
                det['conf'] = result.boxes.conf[detect].cpu()
                det['category'] = result.boxes.cls[detect].cpu()
                coords = result.boxes.xywhn[detect].cpu()
                det['bbox'] = [coords[0]-coords[2]/2, coords[1]-coords[3]/2, coords[2], coords[3]]
                detection_boxes.append(det)
                
            im_path_ = os.path.join(img_path + subdataset + '/images/', img)
            visutils.draw_bounding_boxes_on_file(im_path_, save_path, detection_boxes,
                                            confidence_threshold=0.0, detector_label_map=None,
                                            thickness=1,expansion=0, colormap=['Red'])

            selected_label = img_path  + subdataset + '/labels/' + os.path.basename(result.path).split('.jpg')[0] + '.txt'
            if os.path.exists(selected_label):
                detection_boxes = []
                df = pd.read_csv(selected_label, sep='\t', header=None, index_col=False)
                for irow, row in df.iterrows():  
                    det = {}
                    det['conf'] = None
                    det['category'] = row[0]
                    det['bbox'] = [row[1]-row[3]/2, row[2]-row[4]/2, row[3], row[4]]
                    detection_boxes.append(det)
        
                # Draw annotations
                save_path2 = 'runs/' + TASK + '/' + MODEL_NAME + '/prediction_label_' + os.path.basename(result.path).split('.hpg')[0] + '.jpg'
                visutils.draw_bounding_boxes_on_file(save_path, save_path2, detection_boxes,
                                                confidence_threshold=0.0, detector_label_map=None,
                                                thickness=1,expansion=0, colormap=['SpringGreen'])
                
                # Remove predictions-only images
                os.remove(save_path)
