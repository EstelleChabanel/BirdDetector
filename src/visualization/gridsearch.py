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
MODEL_NAME_PREFIX = "pe_palm_" #'deepcoral_background_lscale16_epochs40_coralgain10' #'pfeifer_penguins_poland_palmyra_10percent_bckgd_yolov8m_120epochs'
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


# ======= ANALYZE RESULTS ======= #

def plot_grid_search_losses(param, param_set, saving_name):

    plotting_data = pd.DataFrame()
    fig1, ax1 = plt.subplots(1, 2, figsize=(12, 6), tight_layout=True)
    ax1 = ax1.ravel()
    fig2, ax2 = plt.subplots(1, 2, figsize=(12, 6), tight_layout=True)
    ax2 = ax2.ravel()

    for p in param_set:
        MODEL_NAME = MODEL_NAME_PREFIX + param + '_' + p
        MODEL_PATH = os.path.join(MODELS_PATH, MODEL_NAME)
        p_ = float(p)
        print("p=", p)

        data = pd.read_csv(os.path.join(MODEL_PATH, CSV_FILE))
        data.rename(str.strip, axis='columns', inplace=True)
        data = data.bfill(axis=1) #fillna(method="ffill", inplace=True)

        plotting_data[f'{p}/epochs'] = data['epoch']
        plotting_data[f'{p}/train/detect_losses'] = data['train/box_loss'] + data['train/cls_loss'] + data['train/dfl_loss']
        plotting_data[f'{p}/train/dc_loss'] = data['train/da_loss']
        plotting_data[f'{p}/train/total_loss'] = data['train/box_loss'] + data['train/cls_loss'] + data['train/dfl_loss'] + data['train/da_loss']
        plotting_data[f'{p}/val/detect_losses'] = data['val/box_loss'] + data['val/cls_loss'] + data['val/dfl_loss']
        plotting_data[f'{p}/val/dc_loss'] = data['val/da_loss']
        #plotting_data[f'{str(param)}/val/total_loss'] = data['val/box_loss'] + data['val/cls_loss'] + data['val/dfl_loss'] + data['val/da_loss']
        
        #Training losses
        ax1[0].plot(plotting_data[f'{(p)}/epochs'], plotting_data[f'{(p)}/train/detect_losses'], marker='.', label=f"param={str(format(p_, '.0e'))}", linewidth=2, markersize=8)
        ax1[1].plot(plotting_data[f'{(p)}/epochs'], plotting_data[f'{(p)}/train/dc_loss'], marker='.', label=f"param={str(format(p_, '.0e'))}", linewidth=2, markersize=8)
        #Validation losses
        ax2[0].plot(plotting_data[f'{(p)}/epochs'], plotting_data[f'{(p)}/val/detect_losses'], marker='.', label=f"param={str(format(p_, '.0e'))}", linewidth=2, markersize=8)
        ax2[1].plot(plotting_data[f'{(p)}/epochs'], plotting_data[f'{(p)}/val/dc_loss'], marker='.', label=f"param={str(format(p_, '.0e'))}", linewidth=2, markersize=8)
        
    ax1[0].legend()
    ax1[0].set_xlabel("epochs", fontsize=12)
    ax1[0].set_ylabel("detection loss", fontsize=12)
    ax1[1].legend()
    ax1[1].set_xlabel("epochs", fontsize=12)
    ax1[1].set_ylabel("domain classifier loss", fontsize=12)
    ax2[0].legend()
    ax2[0].set_xlabel("epochs", fontsize=12)
    ax2[0].set_ylabel("detection loss", fontsize=12)
    ax2[1].legend()
    ax2[1].set_xlabel("epochs", fontsize=12)
    ax2[1].set_ylabel("domain classifier loss", fontsize=12)

    fig1.suptitle("Training losses")
    fig2.suptitle("Validation losses")

    fname1 = os.path.join(MODELS_PATH, saving_name + '_train.png')
    fname2 = os.path.join(MODELS_PATH, saving_name + '_val.png')
    fig1.savefig(fname1, dpi=200)
    fig2.savefig(fname2, dpi=200)

    # Total training loss
    plt.figure(figsize=(10, 8))
    for p in param_set:
        p_ = float(p)
        plt.plot(plotting_data[f'{(p)}/epochs'], plotting_data[f'{(p)}/train/total_loss'], marker='.', label=f"{param}={str(format(p_, '.0e'))}", linewidth=2, markersize=8)
    plt.legend()
    plt.xlabel("epochs", fontsize=12)
    plt.ylabel("training loss", fontsize=12)
    fname = os.path.join(MODELS_PATH, saving_name + '_total_train.png')
    plt.savefig(fname, dpi=200)
    plt.close()

    return


def plot_map_hist(param, param_set, maps, saving_name):
    df_map = pd.DataFrame({param: param_set, "map": maps})
    ax = df_map.plot.bar(x=param, y='map', rot=0, legend=False)
    ax.bar_label(ax.containers[0])
    ax.set_xlabel(param)
    ax.set_ylabel("Average Precision")
    fname = os.path.join(MODELS_PATH, saving_name + '_AP.png')
    plt.savefig(fname, dpi=200)
    plt.close()


def plot_map_hist_s(param, param_set, maps, maps_source, maps_target, saving_name):
    df_map = pd.DataFrame({"source": maps_source, "target": maps_target, "both": maps},
                          index=param_set)
    ax = df_map.plot.bar(rot=0, legend=False)
    for container in ax.containers:
        ax.bar_label(container)
    ax.set_xlabel(param)
    ax.set_ylabel("Average Precision")
    fname = os.path.join(MODELS_PATH, saving_name + 'multiAP.png')
    plt.savefig(fname, dpi=200)
    plt.close()



# ==== LR Grid Search

LRs = ['0.00001','0.00005','0.0001','0.0005','0.001','0.005','0.01']
#plot_grid_search_losses("lr", LRs, 'dc_lr_grid_search')
map = [0.66, 0.91, 0.87, 0.91, 0.9, 0.9, 0.88]
#plot_map_hist("learning rate", LRs, map, 'dc_lr_grid_search')


# ==== DC_Loss Gain Grid Search

DCLoSS_GAINs = ["0.1","0.5","1.0","1.5","5","10"]
plot_grid_search_losses("dclossg", DCLoSS_GAINs, 'dc_gain_grid_search')
map = [0.89, 0.01, 0.9, 0.88, 0.92, 0.9]
map_target = [0.72, 0.0, 0.57, 0.45, 0.74, 0.51]
map_source = [0.91, 0.04, 0.92, 0.9, 0.94, 0.92]
plot_map_hist("domain classifier gain", DCLoSS_GAINs, map, 'dc_gain_grid_search')
plot_map_hist_s("domain classifier gain", DCLoSS_GAINs, map, map_source, map_target, 'dc_gain_grid_search')


"""
plotting_data = pd.DataFrame()

for lr in LRs:
    MODEL_NAME = MODEL_NAME_PREFIX + lr
    MODEL_PATH = os.path.join(MODELS_PATH, MODEL_NAME)

    data = pd.read_csv(os.path.join(MODEL_PATH, CSV_FILE))
    data.rename(str.strip, axis='columns', inplace=True)
    data = data.bfill(axis=1) #fillna(method="ffill", inplace=True)

    data['train/tot_detect_loss'] = data['train/box_loss'] + data['train/cls_loss'] + data['train/dfl_loss']
    data['val/tot_detect_loss'] = data['val/box_loss'] + data['val/cls_loss'] + data['val/dfl_loss']
    data['train/total_loss'] = data['train/box_loss'] + data['train/cls_loss'] + data['train/dfl_loss'] + data['train/da_loss']
    #data['val/total_loss'] = data['val/box_loss'] + data['val/cls_loss'] + data['val/dfl_loss'] + data['val/da_loss']
    
    plotting_data[f'{str(lr)}/epochs'] = data['epoch']
    plotting_data[f'{str(lr)}/train/tot_detect_loss'] = data['train/tot_detect_loss']
    plotting_data[f'{str(lr)}/train/dc_loss'] = data['train/da_loss']
    plotting_data[f'{str(lr)}/train/total_loss'] = data['train/total_loss']
    plotting_data[f'{str(lr)}/val/tot_detect_loss'] = data['val/tot_detect_loss']
    plotting_data[f'{str(lr)}/val/dc_loss'] = data['val/da_loss']

plotting_data = plotting_data.bfill(axis=1)
fig1, ax1 = plt.subplots(1, 2, figsize=(12, 6), tight_layout=True)
ax1 = ax1.ravel()
fig2, ax2 = plt.subplots(1, 2, figsize=(12, 6), tight_layout=True)
ax2 = ax2.ravel()

for lr in LRs:
    lr_ = float(lr)
    #Training losses
    ax1[0].plot(plotting_data[f'{(lr)}/epochs'], plotting_data[f'{(lr)}/train/tot_detect_loss'], marker='.', label=f"lr={str(format(lr_, '.0e'))}", linewidth=2, markersize=8)
    ax1[1].plot(plotting_data[f'{(lr)}/epochs'], plotting_data[f'{(lr)}/train/dc_loss'], marker='.', label=f"lr={str(format(lr_, '.0e'))}", linewidth=2, markersize=8)
    #Validation losses
    ax2[0].plot(plotting_data[f'{(lr)}/epochs'], plotting_data[f'{(lr)}/val/tot_detect_loss'], marker='.', label=f"lr={str(format(lr_, '.0e'))}", linewidth=2, markersize=8)
    ax2[1].plot(plotting_data[f'{(lr)}/epochs'], plotting_data[f'{(lr)}/val/dc_loss'], marker='.', label=f"lr={str(format(lr_, '.0e'))}", linewidth=2, markersize=8)
    
ax1[0].legend()
ax1[0].set_xlabel("epochs", fontsize=12)
ax1[0].set_ylabel("train/detect_loss", fontsize=12)
ax1[1].legend()
ax1[1].set_xlabel("epochs", fontsize=12)
ax1[1].set_ylabel("train/dc_loss", fontsize=12)
ax2[0].legend()
ax2[0].set_xlabel("epochs", fontsize=12)
ax2[0].set_ylabel("val/detect_loss", fontsize=12)
yticks = np.linspace(0, 2, 10)
#ax2[0].set_yticks(yticks)
ax2[1].legend()
ax2[1].set_xlabel("epochs", fontsize=12)
ax2[1].set_ylabel("val/dc_loss", fontsize=12)
yticks = np.linspace(33, 34, 10)
#ax2[1].set_yticks(yticks)
fname1 = os.path.join(MODELS_PATH, 'dc_lr_grid_search_train.png')
fname2 = os.path.join(MODELS_PATH, 'dc_lr_grid_search_val.png')
fig1.savefig(fname1, dpi=200)
fig2.savefig(fname2, dpi=200)

# Total training loss
figure = plt.figure(figsize=(10, 8))
for lr in LRs:
    lr_ = float(lr)
    plt.plot(plotting_data[f'{(lr)}/epochs'], plotting_data[f'{(lr)}/train/total_loss'], marker='.', label=f"lr={str(format(lr_, '.0e'))}", linewidth=2, markersize=8)
plt.legend()
plt.xlabel("epochs", fontsize=12)
plt.ylabel("training loss", fontsize=12)
fname = os.path.join(MODELS_PATH, 'dc_lr_grid_search.png')
plt.savefig(fname, dpi=200)
plt.close()
"""
