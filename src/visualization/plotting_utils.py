import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_grid_search_losses(task, models_folder, model_name, param, param_set):

    plotting_data = pd.DataFrame()
    fig1, ax1 = plt.subplots(1, 2, figsize=(12, 6), tight_layout=True)
    ax1 = ax1.ravel()
    fig2, ax2 = plt.subplots(1, 2, figsize=(12, 6), tight_layout=True)
    ax2 = ax2.ravel()

    for p in param_set:
        MODEL_NAME = model_name + p
        MODEL_PATH = os.path.join(models_folder, MODEL_NAME)
        p_ = float(p)
        print("p=", p)

        data = pd.read_csv(os.path.join(MODEL_PATH, "results.csv"))
        data.replace([np.inf, -np.inf], 0, inplace=True)
        data.rename(str.strip, axis='columns', inplace=True)
        data = data.bfill(axis=1) #fillna(method="ffill", inplace=True)

        plotting_data[f'{p}/epochs'] = data['epoch']
        plotting_data[f'{p}/train/detect_losses'] = data['train/box_loss'] + data['train/cls_loss'] + data['train/dfl_loss']
        plotting_data[f'{p}/val/detect_losses'] = data['val/box_loss'] + data['val/cls_loss'] + data['val/dfl_loss']

        if task=='DAN':
            plotting_data[f'{p}/train/extra_loss'] = data['train/da_loss']
            plotting_data[f'{p}/val/extra_loss'] = data['val/da_loss']
            plotting_data[f'{p}/train/total_loss'] = data['train/box_loss'] + data['train/cls_loss'] + data['train/dfl_loss'] + data['train/da_loss']
            plotting_data[f'{p}/val/total_loss'] = data['val/box_loss'] + data['val/cls_loss'] + data['val/dfl_loss'] + data['val/da_loss']
        elif task=='multiDAN':
            plotting_data[f'{p}/train/extra_loss'] = data['train/da_loss_s'] + data['train/da_loss_m'] + data['train/da_loss_l']
            plotting_data[f'{p}/val/extra_loss'] = data['val/da_loss_s'] + data['val/da_loss_m'] + data['val/da_loss_l']
            plotting_data[f'{p}/train/total_loss'] = data['train/box_loss'] + data['train/cls_loss'] + data['train/dfl_loss'] + data['train/da_loss_s'] + data['train/da_loss_m'] + data['train/da_loss_l']
            plotting_data[f'{p}/val/total_loss'] = data['val/box_loss'] + data['val/cls_loss'] + data['val/dfl_loss'] + data['val/da_loss_s'] + data['val/da_loss_m'] + data['val/da_loss_l']
        elif task=='featdist':
            plotting_data[f'{p}/train/extra_loss'] = data['train/feat_norm']
            plotting_data[f'{p}/val/extra_loss'] = data['val/feat_norm']
            plotting_data[f'{p}/train/total_loss'] = data['train/box_loss'] + data['train/cls_loss'] + data['train/dfl_loss'] + data['train/feat_norm'] 
            plotting_data[f'{p}/val/total_loss'] = data['val/box_loss'] + data['val/cls_loss'] + data['val/dfl_loss'] + data['val/feat_norm'] 
        elif task=='YOLO':
            plotting_data[f'{p}/train/total_loss'] = data['train/box_loss'] + data['train/cls_loss'] + data['train/dfl_loss']
            plotting_data[f'{p}/val/total_loss'] = data['val/box_loss'] + data['val/cls_loss'] + data['val/dfl_loss']

        if not task=='YOLO':
            #Training losses
            ax1[0].plot(plotting_data[f'{(p)}/epochs'], plotting_data[f'{(p)}/train/detect_losses'], marker='.', label=f"{param}={p_}", linewidth=2, markersize=8)
            ax1[1].plot(plotting_data[f'{(p)}/epochs'], plotting_data[f'{(p)}/train/extra_loss'], marker='.', label=f"{param}={p_}", linewidth=2, markersize=8)
            #Validation losses
            ax2[0].plot(plotting_data[f'{(p)}/epochs'], plotting_data[f'{(p)}/val/detect_losses'], marker='.', label=f"{param}={p_}", linewidth=2, markersize=8)
            ax2[1].plot(plotting_data[f'{(p)}/epochs'], plotting_data[f'{(p)}/val/extra_loss'], marker='.', label=f"{param}={p_}", linewidth=2, markersize=8)

    if not task=='YOLO':
        ax1[0].legend()
        ax1[0].set_xlabel("epochs", fontsize=12)
        ax1[0].set_ylabel("detection loss", fontsize=12)
        ax1[1].legend()
        ax1[1].set_xlabel("epochs", fontsize=12)
        ax1[1].set_ylabel("domain adaptation network loss", fontsize=12)
        ax2[0].legend()
        ax2[0].set_xlabel("epochs", fontsize=12)
        ax2[0].set_ylabel("detection loss", fontsize=12)
        ax2[1].legend()
        ax2[1].set_xlabel("epochs", fontsize=12)
        ax2[1].set_ylabel("domain adaptation network loss", fontsize=12)

        fig1.suptitle("Training losses")
        fig2.suptitle("Validation losses")

        fname1 = os.path.join(models_folder, model_name + '_' + param + 'search_traininglosses.png')
        fname2 = os.path.join(models_folder, model_name + '_' + param + 'search_vallosses.png')
        fig1.savefig(fname1, dpi=200)
        fig2.savefig(fname2, dpi=200)
        plt.close()

    # Total training&validation loss
    fig1, ax1 = plt.subplots(1, 2, figsize=(12, 6), tight_layout=True)
    ax1 = ax1.ravel()
    #plt.figure(figsize=(10, 8))
    for p in param_set:
        p_ = float(p)
        ax1[0].plot(plotting_data[f'{(p)}/epochs'], plotting_data[f'{(p)}/train/total_loss'], marker='.', label=f"{param}={p_}", linewidth=2, markersize=8)
        ax1[1].plot(plotting_data[f'{(p)}/epochs'], plotting_data[f'{(p)}/val/total_loss'], marker='.', label=f"{param}={p_}", linewidth=2, markersize=8)
    ax1[0].legend()
    ax1[0].set_xlabel("epochs", fontsize=12)
    ax1[0].set_ylabel("training loss", fontsize=12)
    ax1[1].legend()
    ax1[1].set_xlabel("epochs", fontsize=12)
    ax1[1].set_ylabel("validation loss", fontsize=12)
    fname = os.path.join(models_folder, model_name + '_' + param + 'search_totalloss.png')
    fig1.savefig(fname, dpi=200)
    plt.close()

    return



def plot_map_hist(models_folder, model_name, param, param_set, maps):
    df_map = pd.DataFrame({param: param_set, "map": maps})
    ax = df_map.plot.bar(x=param, y='map', rot=0, legend=False)
    ax.bar_label(ax.containers[0])
    ax.set_xlabel(param)
    ax.set_ylabel("Average Precision")
    fname = os.path.join(models_folder, model_name + '_' + param + 'search_AP.png')
    plt.savefig(fname, dpi=200)
    plt.close()


def plot_map_hist_s(models_folder, model_name, param, param_set, maps, maps_source, maps_target):
    print(param_set)
    df_map = pd.DataFrame({"source": maps_source, "target": maps_target, "both": maps},
                          index=param_set)
    ax = df_map.plot.bar(rot=0, legend=True) #, color=['green','red','blue'])
    for container in ax.containers:
        ax.bar_label(container)
    ax.set_xlabel(param)
    ax.set_ylabel("Average Precision")
    ax.legend(loc='lower right')
    fname = os.path.join(models_folder, model_name + '_' + param + 'search_multiAP.png')
    plt.savefig(fname, dpi=200)
    plt.close()