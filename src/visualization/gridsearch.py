import yolo
from yolo import YOLO

import torch
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Use GPU if available
device = "0" if torch.cuda.is_available() else "cpu"
if device == "0":
    torch.cuda.set_device(0) # Set to your desired GPU number

import src.data_preprocessing.visualization_utils as visutils
from src.model.constants import DATA_PATH, DATASETS_MAPPING, MODELS_PATH, NB_EPOCHS, BATCH_SIZE, PATIENCE, OPTIMIZER, TRAINING_IOU_THRESHOLD, CONF_THRESHOLD, NB_CONF_THRESHOLDS, IOU_THRESHOLD


# ======= ANALYZE RESULTS ======= #

def plot_grid_search_losses(param, param_set, saving_name):

    plotting_data = pd.DataFrame()
    fig1, ax1 = plt.subplots(1, 2, figsize=(12, 6), tight_layout=True)
    ax1 = ax1.ravel()
    fig2, ax2 = plt.subplots(1, 2, figsize=(12, 6), tight_layout=True)
    ax2 = ax2.ravel()

    for p in param_set:
        MODEL_NAME = MODEL_NAME_PREFIX + p
        MODEL_PATH = os.path.join(MODELS_PATH, MODEL_NAME)
        p_ = float(p)
        print("p=", p)

        data = pd.read_csv(os.path.join(MODEL_PATH, CSV_FILE))
        data.replace([np.inf, -np.inf], 0, inplace=True)
        data.rename(str.strip, axis='columns', inplace=True)
        data = data.bfill(axis=1) #fillna(method="ffill", inplace=True)

        plotting_data[f'{p}/epochs'] = data['epoch']
        plotting_data[f'{p}/train/detect_losses'] = data['train/box_loss'] + data['train/cls_loss'] + data['train/dfl_loss']
        plotting_data[f'{p}/train/dc_loss'] = data['train/da_loss']
        plotting_data[f'{p}/train/total_loss'] = data['train/box_loss'] + data['train/cls_loss'] + data['train/dfl_loss'] + data['train/da_loss']
        #plotting_data[f'{p}/val/detect_losses'] = data['val/box_loss'] + data['val/cls_loss'] + data['val/dfl_loss']
        plotting_data[f'{p}/val/dc_loss'] = data['val/da_loss']
        #plotting_data[f'{str(param)}/val/total_loss'] = data['val/box_loss'] + data['val/cls_loss'] + data['val/dfl_loss'] + data['val/da_loss']
        
        #Training losses
        ax1[0].plot(plotting_data[f'{(p)}/epochs'], plotting_data[f'{(p)}/train/detect_losses'], marker='.', label=f"{param}={str(format(p_, '.0e'))}", linewidth=2, markersize=8)
        ax1[1].plot(plotting_data[f'{(p)}/epochs'], plotting_data[f'{(p)}/train/dc_loss'], marker='.', label=f"{param}={str(format(p_, '.0e'))}", linewidth=2, markersize=8)
        #Validation losses
        #ax2[0].plot(plotting_data[f'{(p)}/epochs'], plotting_data[f'{(p)}/val/detect_losses'], marker='.', label=f"{param}={str(format(p_, '.0e'))}", linewidth=2, markersize=8)
        ax2[1].plot(plotting_data[f'{(p)}/epochs'], plotting_data[f'{(p)}/val/dc_loss'], marker='.', label=f"{param}={str(format(p_, '.0e'))}", linewidth=2, markersize=8)
        
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


def plot_grid_search_losses_multiDAN(param, param_set, saving_name):

    plotting_data = pd.DataFrame()
    fig1, ax1 = plt.subplots(1, 2, figsize=(12, 6), tight_layout=True)
    ax1 = ax1.ravel()
    fig2, ax2 = plt.subplots(1, 2, figsize=(12, 6), tight_layout=True)
    ax2 = ax2.ravel()

    for p in param_set:
        MODEL_NAME = MODEL_NAME_PREFIX + p
        MODEL_PATH = os.path.join(MODELS_PATH, MODEL_NAME)
        p_ = float(p)
        print("p=", p)

        data = pd.read_csv(os.path.join(MODEL_PATH, CSV_FILE))
        data.rename(str.strip, axis='columns', inplace=True)
        data = data.bfill(axis=1) #fillna(method="ffill", inplace=True)

        plotting_data[f'{p}/epochs'] = data['epoch']
        plotting_data[f'{p}/train/detect_losses'] = data['train/box_loss'] + data['train/cls_loss'] + data['train/dfl_loss']
        plotting_data[f'{p}/train/dc_loss'] = data['train/da_loss_s'] + data['train/da_loss_m'] + data['train/da_loss_l']
        plotting_data[f'{p}/train/total_loss'] = data['train/box_loss'] + data['train/cls_loss'] + data['train/dfl_loss'] + data['train/da_loss_s'] + data['train/da_loss_m'] + data['train/da_loss_l']
        plotting_data[f'{p}/val/detect_losses'] = data['val/box_loss'] + data['val/cls_loss'] + data['val/dfl_loss']
        plotting_data[f'{p}/val/dc_loss'] = data['val/da_loss_s'] + data['val/da_loss_m'] + data['val/da_loss_l']
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

def plot_grid_search_losses_YOLO(param, param_set, saving_name):

    plotting_data = pd.DataFrame()
    fig1, ax1 = plt.subplots(1, 2, figsize=(12, 6), tight_layout=True)
    ax1 = ax1.ravel()
    fig2, ax2 = plt.subplots(1, 2, figsize=(12, 6), tight_layout=True)
    ax2 = ax2.ravel()

    for p in param_set:
        MODEL_NAME = MODEL_NAME_PREFIX + p
        MODEL_PATH = os.path.join(MODELS_PATH, MODEL_NAME)
        p_ = float(p)
        print("p=", p)

        data = pd.read_csv(os.path.join(MODEL_PATH, CSV_FILE))
        data.rename(str.strip, axis='columns', inplace=True)
        data = data.bfill(axis=1) #fillna(method="ffill", inplace=True)

        plotting_data[f'{p}/epochs'] = data['epoch']
        plotting_data[f'{p}/train/detect_losses'] = data['train/box_loss'] + data['train/cls_loss'] + data['train/dfl_loss']
        #plotting_data[f'{p}/train/dc_loss'] = data['train/da_loss_s'] + data['train/da_loss_m'] + data['train/da_loss_l']
        plotting_data[f'{p}/train/total_loss'] = data['train/box_loss'] + data['train/cls_loss'] + data['train/dfl_loss'] #+ data['train/da_loss_s'] + data['train/da_loss_m'] + data['train/da_loss_l']
        #plotting_data[f'{p}/val/detect_losses'] = data['val/box_loss'] + data['val/cls_loss'] + data['val/dfl_loss']
        plotting_data[f'{p}/val/total_loss'] = data['val/box_loss'] + data['val/cls_loss'] + data['val/dfl_loss']
        #plotting_data[f'{str(param)}/val/total_loss'] = data['val/box_loss'] + data['val/cls_loss'] + data['val/dfl_loss'] + data['val/da_loss']
        
        
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

    # Total validation loss
    plt.figure(figsize=(10, 8))
    for p in param_set:
        p_ = float(p)
        plt.plot(plotting_data[f'{(p)}/epochs'], plotting_data[f'{(p)}/val/total_loss'], marker='.', label=f"{param}={str(format(p_, '.0e'))}", linewidth=2, markersize=8)
    plt.legend()
    plt.xlabel("epochs", fontsize=12)
    plt.ylabel("training loss", fontsize=12)
    fname = os.path.join(MODELS_PATH, saving_name + '_total_val.png')
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
    print(param_set)
    df_map = pd.DataFrame({"source": maps_source, "target": maps_target, "both": maps},
                          index=param_set)
    ax = df_map.plot.bar(rot=0, legend=True, color=['green','red','blue'])
    for container in ax.containers:
        ax.bar_label(container)
    ax.set_xlabel(param)
    ax.set_ylabel("Average Precision")
    ax.legend(loc='lower right')
    fname = os.path.join(MODELS_PATH, saving_name + 'multiAP.png')
    plt.savefig(fname, dpi=200)
    plt.close()



# ==== LR Grid Search

'''
maps = {'pe_palmyra_10percentbkgd': {'YOLO': {'map': [0.82, 0.83, 0.87, 0.55, 0.91, 0.9, 0.88, 0.89],
                                             'map_target': [0.04, 0.05, 0.51, 0.0, 0.58, 0.74, 0.59, 0.59],
                                             'map_source': [0.89, 0.9, 0.89, 0.59, 0.93, 0.91, 0.89, 0.9]},
                                    'DAN': {'map': [0.87, 0.88, 0.9, 0.89, 0.88, 0.85],
                                             'map_target': [0.52, 0.27, 0.48, 0.62, 0.34, 0.32],
                                             'map_source': [0.9, 0.91, 0.92, 0.91, 0.93, 0.89]},
                                    'multiDAN': {'map': [0.84, 0.84, 0.9, 0.91, 0.9, 0.87],
                                                  'map_target': [0.11, 0.22, 0.65, 0.68, 0.63, 0.67],
                                                  'map_source': [0.9, 0.88, 0.91, 0.93, 0.92, 0.88]},}
        }


CSV_FILE = "results.csv"

# Predictions parameters
eps = 1e-8

for dataset in maps.keys():
    DATASET_NAME = dataset
    SUBDATASETS = DATASETS_MAPPING[DATASET_NAME]['datasets']
    MODEL_NAME_PREFIX_ = DATASET_NAME + '_'

    # YOLO - original
    MODEL_NAME_PREFIX = "YOLO_" + MODEL_NAME_PREFIX_
    LRs = ['0.00005','0.0001','0.0005','0.001','0.005','0.01','0.05','0.1']
    plot_grid_search_losses_YOLO("lr", LRs, 'YOLO_lr_grid_search')
    map = maps[dataset]['YOLO']['map']
    map_target = maps[dataset]['YOLO']['map_target']
    map_source = maps[dataset]['YOLO']['map_source']
    plot_map_hist("learning rate", LRs, map, 'YOLO_lr_grid_search')
    plot_map_hist_s("learning rate", LRs, map, map_source, map_target, 'YOLO_lr_grid_search')

    
    # DAN
    MODEL_NAME_PREFIX = "DAN_" + MODEL_NAME_PREFIX_
    LRs = ['0.0005','0.001','0.005','0.01','0.05','0.1']
    plot_grid_search_losses("lr", LRs, 'DAN_lr_grid_search')
    map = maps[dataset]['DAN']['map']
    map_target = maps[dataset]['DAN']['map_target']
    map_source = maps[dataset]['DAN']['map_source']
    plot_map_hist("learning rate", LRs, map, 'DAN_lr_grid_search')
    plot_map_hist_s("learning rate", LRs, map, map_source, map_target, 'DAN_lr_grid_search')


    # multiDAN
    MODEL_NAME_PREFIX = 'multiDAN_' + MODEL_NAME_PREFIX_
    LRs = ['0.00005','0.0001','0.0005','0.001','0.005','0.01']
    plot_grid_search_losses_multiDAN("lr", LRs, 'multiDAN_lr_grid_search')
    map = maps[dataset]['multiDAN']['map']
    map_target = maps[dataset]['multiDAN']['map_target']
    map_source = maps[dataset]['multiDAN']['map_source']
    plot_map_hist("learning rate", LRs, map, 'multiDAN_lr_grid_search')
    plot_map_hist_s("learning rate", LRs, map, map_source, map_target, 'multiDAN_lr_grid_search')
    '''



# ==== Loss gain Grid Search

maps = {'pe_palmyra_10percentbkgd': {'DAN': {'map_target': [0.74, 0.57, 0.76, 0.7, 0.49, 0.62, 0.59],
                                             'map_source': [0.92, 0.94, 0.94, 0.92, 0.93, 0.91, 0.91],
                                             'map': [0.91, 0.92, 0.93, 0.91, 0.9, 0.9, 0.89]},
                                    'multiDAN': {'map_target': [0.8, 0.65, 0.53, 0.63, 0.68, 0.6, 0.67, 0.52],
                                                  'map_source': [0.92, 0.92, 0.92, 0.94, 0.93, 0.94, 0.92, 0.91],
                                                  'map': [0.91, 0.91, 0.9, 0.92, 0.91, 0.92, 0.9, 0.88],},}
        }


CSV_FILE = "results.csv"

# Predictions parameters
eps = 1e-8
#plt.rcParams['text.usetex'] = True

for dataset in maps.keys():
    DATASET_NAME = dataset
    SUBDATASETS = DATASETS_MAPPING[DATASET_NAME]['datasets']
    MODEL_NAME_PREFIX_ = DATASET_NAME + '_'

    # DAN
    MODEL_NAME_PREFIX = "DAN_" + MODEL_NAME_PREFIX_
    GAINs = ['0.75','1.0','1.5','2','3','5','10']
    plot_grid_search_losses("gain", GAINs, 'DAN_gain_grid_search')
    map = maps[dataset]['DAN']['map']
    map_target = maps[dataset]['DAN']['map_target']
    map_source = maps[dataset]['DAN']['map_source']
    plot_map_hist('gain', GAINs, map, 'DAN_gain_grid_search')
    plot_map_hist_s('gain', GAINs, map, map_source, map_target, 'DAN_gain_grid_search')

    '''
    # multiDAN
    MODEL_NAME_PREFIX = 'multiDAN_' + MODEL_NAME_PREFIX_
    GAINs = ['0.5','0.75','1.0','1.5','2','3','5','10']
    plot_grid_search_losses_multiDAN("$\alpha$", GAINs, 'multiDAN_gain_grid_search')
    map = maps[dataset]['multiDAN']['map']
    map_target = maps[dataset]['multiDAN']['map_target']
    map_source = maps[dataset]['multiDAN']['map_source']
    plot_map_hist("$\alpha$", GAINs, map, 'multiDAN_gain_grid_search')
    plot_map_hist_s("$\alpha$", GAINs, map, map_source, map_target, 'multiDAN_gain_grid_search')
    '''
