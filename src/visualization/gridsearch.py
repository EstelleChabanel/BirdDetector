import yolo
from yolo import YOLO

import torch
import os
import random
import json
import yaml
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

# Use GPU if available
device = "0" if torch.cuda.is_available() else "cpu"
if device == "0":
    torch.cuda.set_device(0) # Set to your desired GPU number


# ======= PARAMETERS =======

# Model specifications
MODEL_NAME_PREFIX = "pe_palm_lr_" #'deepcoral_background_lscale16_epochs40_coralgain10' #'pfeifer_penguins_poland_palmyra_10percent_bckgd_yolov8m_120epochs'
SUBTASK = 'domainclassifier' #Choose between: #'deepcoral_detect' #'detect'
MODELS_PATH = 'runs/detect/'
CSV_FILE = "results.csv"

# Data
DATASET_NAME = "pe_palmyra_10percentbkgd" #'pfpepo_palmyra_10percentbkgd' #'deepcoral_palmyraT__10percent_background' #'pfpepo_palmyra_10percentbkgd'
DATASETS_MAPPING = {'pe_palmyra_10percentbkgd': {'source': ['global_birds_penguins', 'global_birds_palmyra']}}
SUBDATASETS = DATASETS_MAPPING[DATASET_NAME] #{'source': ['global_birds_penguins', 'global_birds_pfeifer', 'global_birds_palmyra']} #   'global_birds_pfeifer',            'target': ['global_birds_palmyra']}

# Predictions parameters
IOU_THRESHOLD = 0.1
NB_CONF_THRESHOLDS = 50
CONF_THRESHOLDS = np.linspace(0, 1, NB_CONF_THRESHOLDS) # CAREFUL: if you change that, don't forget to change calls to plot_confusion_matrix function

eps = 1e-8


# ======= ANALYZE RESULTS =======

LRs = [0.0005,0.0001,0.001,0.005,0.01]
plotting_data = pd.DataFrame()

for lr in LRs:
    MODEL_NAME = MODEL_NAME_PREFIX + str(lr)
    MODEL_PATH = os.path.join(MODELS_PATH, MODEL_NAME)

    data = pd.read_csv(os.path.join(MODEL_PATH, CSV_FILE))

    data['train/tot_detect_loss'] = data['train/box_loss,'] + data['train/cls_loss,'] + data['train/dfl_loss,']
    data['val/tot_detect_loss'] = data['val/box_loss,'] + data['val/cls_loss,'] + data['val/dfl_loss,']

    plotting_data[f'{str(lr)}/epochs'] = data['epoch,']
    plotting_data[f'{str(lr)}/train/tot_detect_loss'] = data['train/tot_detect_loss']
    plotting_data[f'{str(lr)}/train/dc_loss'] = data['train/da_loss,']
    plotting_data[f'{str(lr)}/val/tot_detect_loss'] = data['val/tot_detect_loss']
    plotting_data[f'{str(lr)}/val/dc_loss'] = data['val/da_loss,']


fig, ax = plt.subplots(2, 2, figsize=(10, 6), tight_layout=True)
ax = ax.ravel()

for lr in LRs:
    #Training losses
    ax[0].plot(plotting_data[f'{str(lr)}/epochs'], plotting_data[f'{str(lr)}/train/tot_detect_loss'], marker='.', label="lr=str{lr}", linewidth=2, markersize=8)
    ax[1].plot(plotting_data[f'{str(lr)}/epochs'], plotting_data[f'{str(lr)}/train/dc_loss'], marker='.', label="lr=str{lr}", linewidth=2, markersize=8)
    #Validation losses
    ax[2].plot(plotting_data[f'{str(lr)}/epochs'], plotting_data[f'{str(lr)}/val/tot_detect_loss'], marker='.', label="lr=str{lr}", linewidth=2, markersize=8)
    ax[3].plot(plotting_data[f'{str(lr)}/epochs'], plotting_data[f'{str(lr)}/val/dc_loss'], marker='.', label="lr=str{lr}", linewidth=2, markersize=8)

                
    
