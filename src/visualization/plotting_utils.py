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



def plot_multiplearch_losses_bis(task_name, models_folder, dataset_name):

    fig1, (ax11, ax12) = plt.subplots(2, 1, sharex=True, figsize=(8, 6), tight_layout=True)
    fig1.subplots_adjust(hspace=0.05)  # adjust space between axes   
    fig2, (ax21, ax22) = plt.subplots(2, 1, sharex=True, figsize=(8, 6), tight_layout=True)
    fig2.subplots_adjust(hspace=0.05)  # adjust space between axes    
    fig3, (ax31) = plt.subplots(1, 1, sharex=True, figsize=(8, 6), tight_layout=True)
    fig3.subplots_adjust(hspace=0.05)  # adjust space between axes  
    fig4, (ax41, ax42) = plt.subplots(2, 1, sharex=True, figsize=(8, 6), tight_layout=True)
    fig4.subplots_adjust(hspace=0.05)  # adjust space between axes           
    plotting_data = pd.DataFrame()


    for task in task_name.keys():

        model_name = task + '_' + dataset_name
        MODEL_PATH = os.path.join(models_folder, model_name)
        print("task=", task)

        data = pd.read_csv(os.path.join(MODEL_PATH, "results.csv"))
        data.replace([np.inf, -np.inf], 0, inplace=True)
        data.rename(str.strip, axis='columns', inplace=True)
        data = data.bfill(axis=1) #fillna(method="ffill", inplace=True)

        plotting_data[f'{task}/epochs'] = data['epoch']
        plotting_data[f'{task}/train/detect_losses'] = data['train/box_loss'] + data['train/cls_loss'] + data['train/dfl_loss']
        plotting_data[f'{task}/val/detect_losses'] = data['val/box_loss'] + data['val/cls_loss'] + data['val/dfl_loss']

        if task=='DAN':
            plotting_data[f'{task}/train/extra_loss'] = data['train/da_loss']
            plotting_data[f'{task}/val/extra_loss'] = data['val/da_loss']
            plotting_data[f'{task}/train/total_loss'] = data['train/box_loss'] + data['train/cls_loss'] + data['train/dfl_loss'] + data['train/da_loss']
            plotting_data[f'{task}/val/total_loss'] = data['val/box_loss'] + data['val/cls_loss'] + data['val/dfl_loss'] + data['val/da_loss']
        elif task=='multiDAN':
            plotting_data[f'{task}/train/extra_loss'] = data['train/da_loss_s'] + data['train/da_loss_m'] + data['train/da_loss_l']
            plotting_data[f'{task}/val/extra_loss'] = data['val/da_loss_s'] + data['val/da_loss_m'] + data['val/da_loss_l']
            plotting_data[f'{task}/train/total_loss'] = data['train/box_loss'] + data['train/cls_loss'] + data['train/dfl_loss'] + data['train/da_loss_s'] + data['train/da_loss_m'] + data['train/da_loss_l']
            plotting_data[f'{task}/val/total_loss'] = data['val/box_loss'] + data['val/cls_loss'] + data['val/dfl_loss'] + data['val/da_loss_s'] + data['val/da_loss_m'] + data['val/da_loss_l']
        elif task=='featdist':
            plotting_data[f'{task}/train/extra_loss'] = data['train/feat_norm']
            plotting_data[f'{task}/val/extra_loss'] = data['val/feat_norm']
            plotting_data[f'{task}/train/total_loss'] = data['train/box_loss'] + data['train/cls_loss'] + data['train/dfl_loss'] + data['train/feat_norm'] 
            plotting_data[f'{task}/val/total_loss'] = data['val/box_loss'] + data['val/cls_loss'] + data['val/dfl_loss'] + data['val/feat_norm'] 
        elif task=='YOLO':
            plotting_data[f'{task}/train/total_loss'] = data['train/box_loss'] + data['train/cls_loss'] + data['train/dfl_loss']
            plotting_data[f'{task}/val/total_loss'] = data['val/box_loss'] + data['val/cls_loss'] + data['val/dfl_loss']

        if not task=='YOLO':
            #Training losses
            ax11.plot(plotting_data[f'{task}/epochs'], plotting_data[f'{task}/train/detect_losses'], marker='.', label=f"{task_name[task]}", linewidth=2, markersize=8)
            ax12.plot(plotting_data[f'{task}/epochs'], plotting_data[f'{task}/train/detect_losses'], marker='.', label=f"{task_name[task]}", linewidth=2, markersize=8)
            ax21.plot(plotting_data[f'{task}/epochs'], plotting_data[f'{task}/train/extra_loss'], marker='.', label=f"{task_name[task]}", linewidth=2, markersize=8)
            ax22.plot(plotting_data[f'{task}/epochs'], plotting_data[f'{task}/train/extra_loss'], marker='.', label=f"{task_name[task]}", linewidth=2, markersize=8)
            #Validation losses
            ax31.plot(plotting_data[f'{task}/epochs'], plotting_data[f'{task}/val/detect_losses'], marker='.', label=f"{task_name[task]}", linewidth=2, markersize=8)
#            ax32.plot(plotting_data[f'{task}/epochs'], plotting_data[f'{task}/val/detect_losses'], marker='.', label=f"{task_name[task]}", linewidth=2, markersize=8)
            ax41.plot(plotting_data[f'{task}/epochs'], plotting_data[f'{task}/val/extra_loss'], marker='.', label=f"{task_name[task]}", linewidth=2, markersize=8)
            ax42.plot(plotting_data[f'{task}/epochs'], plotting_data[f'{task}/val/extra_loss'], marker='.', label=f"{task_name[task]}", linewidth=2, markersize=8)

    ax11.set_ylim(11, 16.5)  # outliers only
    ax12.set_ylim(3.8, 8)  # most of the data
    ax21.set_ylim(7, 10.3)  # outliers only
    ax22.set_ylim(-0.7, 2)  # most of the data
    ax41.set_ylim(4, 5)  # outliers only
    ax42.set_ylim(-0.7, 1.5)  # most of the data    
    
    # hide the spines between ax and ax2
    ax11.spines.bottom.set_visible(False)
    ax12.spines.top.set_visible(False)
    ax11.xaxis.tick_top()
    ax11.tick_params(labeltop=False)  # don't put tick labels at the top
    ax12.xaxis.tick_bottom()
    ax21.spines.bottom.set_visible(False)
    ax22.spines.top.set_visible(False)
    ax21.xaxis.tick_top()
    ax21.tick_params(labeltop=False)  # don't put tick labels at the top
    ax22.xaxis.tick_bottom()
    ax41.spines.bottom.set_visible(False)
    ax42.spines.top.set_visible(False)
    ax41.xaxis.tick_top()
    ax41.tick_params(labeltop=False)  # don't put tick labels at the top
    ax42.xaxis.tick_bottom()

    ax12.legend()
    ax11.set_xlabel("epochs", fontsize=12)
    ax12.set_ylabel("detection loss", fontsize=12)

    d = .5  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    ax11.plot([0, 1], [0, 0], transform=ax11.transAxes, **kwargs)
    ax12.plot([0, 1], [1, 1], transform=ax12.transAxes, **kwargs)
    ''' 
    ax11.legend()
    ax11.set_xlabel("epochs", fontsize=12)
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
    '''
    fig1.suptitle("Training losses")
    fig2.suptitle("Validation losses")

    fname1 = os.path.join(models_folder, dataset_name + '_' + 'traininglosses.png')
    fname2 = os.path.join(models_folder, dataset_name + '_' + 'vallosses.png')
    fig1.savefig(fname1, dpi=200)
    fig2.savefig(fname2, dpi=200)
    plt.close()

    return



def plot_multiplearch_losses(task_name, models_folder, dataset_name):

    fig1, ax1 = plt.subplots(1, 2, sharex=True, figsize=(14, 6), tight_layout=True)
    fig1.subplots_adjust(hspace=0.05)  # adjust space between axes   
    fig2, ax2 = plt.subplots(1, 2, sharex=True, figsize=(14, 6), tight_layout=True)
    fig2.subplots_adjust(hspace=0.05)  # adjust space between axes          
    plotting_data = pd.DataFrame()

    for task in task_name.keys():

        model_name = task + '_' + dataset_name
        MODEL_PATH = os.path.join(models_folder, model_name)
        print("task=", task)

        data = pd.read_csv(os.path.join(MODEL_PATH, "results.csv"))
        data.replace([np.inf, -np.inf], 0, inplace=True)
        data.rename(str.strip, axis='columns', inplace=True)
        data = data.bfill(axis=1) #fillna(method="ffill", inplace=True)

        plotting_data[f'{task}/epochs'] = data['epoch']
        plotting_data[f'{task}/train/detect_losses'] = data['train/box_loss'] + data['train/cls_loss'] + data['train/dfl_loss']
        plotting_data[f'{task}/val/detect_losses'] = data['val/box_loss'] + data['val/cls_loss'] + data['val/dfl_loss']

        if task=='DAN':
            plotting_data[f'{task}/train/extra_loss'] = data['train/da_loss']
            plotting_data[f'{task}/val/extra_loss'] = data['val/da_loss']
            plotting_data[f'{task}/train/total_loss'] = data['train/box_loss'] + data['train/cls_loss'] + data['train/dfl_loss'] + data['train/da_loss']
            plotting_data[f'{task}/val/total_loss'] = data['val/box_loss'] + data['val/cls_loss'] + data['val/dfl_loss'] + data['val/da_loss']
        elif task=='multiDAN':
            plotting_data[f'{task}/train/extra_loss'] = data['train/da_loss_s'] + data['train/da_loss_m'] + data['train/da_loss_l']
            plotting_data[f'{task}/val/extra_loss'] = data['val/da_loss_s'] + data['val/da_loss_m'] + data['val/da_loss_l']
            plotting_data[f'{task}/train/total_loss'] = data['train/box_loss'] + data['train/cls_loss'] + data['train/dfl_loss'] + data['train/da_loss_s'] + data['train/da_loss_m'] + data['train/da_loss_l']
            plotting_data[f'{task}/val/total_loss'] = data['val/box_loss'] + data['val/cls_loss'] + data['val/dfl_loss'] + data['val/da_loss_s'] + data['val/da_loss_m'] + data['val/da_loss_l']
        elif task=='featdist':
            plotting_data[f'{task}/train/extra_loss'] = data['train/feat_norm']
            plotting_data[f'{task}/val/extra_loss'] = data['val/feat_norm']
            plotting_data[f'{task}/train/total_loss'] = data['train/box_loss'] + data['train/cls_loss'] + data['train/dfl_loss'] + data['train/feat_norm'] 
            plotting_data[f'{task}/val/total_loss'] = data['val/box_loss'] + data['val/cls_loss'] + data['val/dfl_loss'] + data['val/feat_norm'] 
        elif task=='YOLO':
            plotting_data[f'{task}/train/total_loss'] = data['train/box_loss'] + data['train/cls_loss'] + data['train/dfl_loss']
            plotting_data[f'{task}/val/total_loss'] = data['val/box_loss'] + data['val/cls_loss'] + data['val/dfl_loss']

        if not task=='YOLO':
            #Training losses
            ax1[0].plot(plotting_data[f'{task}/epochs'], plotting_data[f'{task}/train/detect_losses'], marker='.', label=f"{task_name[task]}", linewidth=2, markersize=8)
            ax1[1].plot(plotting_data[f'{task}/epochs'], plotting_data[f'{task}/train/extra_loss'], marker='.', label=f"{task_name[task]}", linewidth=2, markersize=8)
            #Validation losses
            ax2[0].plot(plotting_data[f'{task}/epochs'], plotting_data[f'{task}/val/detect_losses'], marker='.', label=f"{task_name[task]}", linewidth=2, markersize=8)
#            ax32.plot(plotting_data[f'{task}/epochs'], plotting_data[f'{task}/val/detect_losses'], marker='.', label=f"{task_name[task]}", linewidth=2, markersize=8)
            ax2[1].plot(plotting_data[f'{task}/epochs'], plotting_data[f'{task}/val/extra_loss'], marker='.', label=f"{task_name[task]}", linewidth=2, markersize=8)
        else:
            ax1[0].plot(plotting_data[f'{task}/epochs'], plotting_data[f'{task}/train/detect_losses'], marker='.', label=f"{task_name[task]}", linewidth=2, markersize=8)
            ax2[0].plot(plotting_data[f'{task}/epochs'], plotting_data[f'{task}/val/detect_losses'], marker='.', label=f"{task_name[task]}", linewidth=2, markersize=8)


    ax1[0].legend()
    ax1[1].legend()
    ax2[0].legend()
    ax2[1].legend()

    ax1[0].set_xlabel("epochs", fontsize=12)
    ax1[0].set_ylabel("detection loss", fontsize=12)    
    ax1[0].set_ylabel("detection loss", fontsize=12)

    ax1[1].set_xlabel("epochs", fontsize=12)
    ax1[1].set_ylabel("domain adaptation network loss", fontsize=12)

    ax2[0].set_xlabel("epochs", fontsize=12)
    ax2[0].set_ylabel("detection loss", fontsize=12)

    ax2[1].set_xlabel("epochs", fontsize=12)
    ax2[1].set_ylabel("domain adaptation network loss", fontsize=12)
    
    fig1.suptitle("Training losses")
    fig2.suptitle("Validation losses")

    fname1 = os.path.join(models_folder, dataset_name + '_multiDANarch_' + 'traininglosses.png')
    fname2 = os.path.join(models_folder, dataset_name + '_multiDANarch_' + 'vallosses.png')
    fig1.savefig(fname1, dpi=200)
    fig2.savefig(fname2, dpi=200)
    plt.close()

    return
