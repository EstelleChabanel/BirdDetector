import ultralytics
ultralytics.checks()
from ultralytics import YOLO

import os
import pandas as pd
import sys
import torch

module_path = os.path.abspath(os.path.join('..'))
print(module_path)
module_path = module_path+'/data_preprocessing'
print(module_path)

if module_path not in sys.path:
    sys.path.append(module_path)

device = "0" if torch.cuda.is_available() else "cpu"
if device == "0":
    torch.cuda.set_device(0) # Set to your desired GPU number


model = YOLO('yolov8m.pt')

# Tune hyperparameters on COCO8 for 30 epochs
model.tune(data='data.yaml', iterations=15, epochs=70, patience=100, optimizer='SGD', plots=False, save=False, val=True, workers=8) #,
           #space={"lr0": tune.uniform(1e-5, 1e-1)}, use_ray=True)