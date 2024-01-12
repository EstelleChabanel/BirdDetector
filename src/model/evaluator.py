#import ultralytics
#ultralytics.checks()
#from ultralytics import YOLO

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

module_path = os.path.abspath(os.path.join('..'))
module_path = module_path+'/data_preprocessing'

if module_path not in sys.path:
    sys.path.append(module_path)

#import src.data_preprocessing.visualization_utils as visutils

# Use GPU if available
device = "0" if torch.cuda.is_available() else "cpu"
if device == "0":
    torch.cuda.set_device(0) # Set to your desired GPU number



# ======= PARAMETERS =======

# Model specifications
MODEL_NAME = 'DAN_pfpe_palm_Adam1e-3_dcLoss1' #'deepcoral_background_lscale16_epochs40_coralgain10' #'pfeifer_penguins_poland_palmyra_10percent_bckgd_yolov8m_120epochs'
SUBTASK = 'domainclassifier' #Choose between: #'deepcoral_detect' #'detect'

# Data
DATASET_NAME = 'pfpe_palmyra_10percentbkgd' #'pfpepo_palmyra_10percentbkgd' #'deepcoral_palmyraT__10percent_background' #'pfpepo_palmyra_10percentbkgd'
SUBDATASETS = {'source': ['global_birds_penguins', 'global_birds_pfeifer', 'global_birds_palmyra']} #   'global_birds_pfeifer',            'target': ['global_birds_palmyra']}

# Predictions parameters
IOU_THRESHOLD = 0.1
NB_CONF_THRESHOLDS = 50
CONF_THRESHOLDS = np.linspace(0, 1, NB_CONF_THRESHOLDS) # CAREFUL: if you change that, don't forget to change calls to plot_confusion_matrix function

eps = 1e-8


# ====== Load model & prepare evaluation ======

TASK = 'detect'
MODEL_PATH = 'runs/detect/' + MODEL_NAME + '/weights/best.pt'
#MODEL_PATH = 'src/model/runs/' + TASK + '/' + MODEL_NAME + '/weights/best.pt'

model = YOLO('yolov8m_domainclassifier.yaml', task=TASK, subtask=SUBTASK ).load(MODEL_PATH)


IMG_PATH = '/gpfs/gibbs/project/jetz/eec42/data/' + DATASET_NAME + '/test/'
#SAVE_DIR = os.path.join('/vast/palmer/home.grace/eec42/BirdDetector/src/model/runs/', TASK, MODEL_NAME)
SAVE_DIR = os.path.join('/vast/palmer/home.grace/eec42/BirdDetector/runs/detect', MODEL_NAME)


# ====== FUNCTIONS FOR PREDICTIONS PROCESSING ======

def box_iou(box1, box2, eps=1e-7):
    """
    From Ultralytics
    Calculate intersection-over-union (IoU) of boxes. Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Based on https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py

    Args:
        box1 (torch.Tensor): A tensor of shape (N, 4) representing N bounding boxes.
        box2 (torch.Tensor): A tensor of shape (M, 4) representing M bounding boxes.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        (torch.Tensor): An NxM tensor containing the pairwise IoU values for every element in box1 and box2.
    """

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp_(0).prod(2)

    # IoU = inter / (area1 + area2 - inter)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)


# match_predictions method from YOLOv8 code - try to reuse it to simplify

def match_predictions(pred_classes, true_classes, iou, use_scipy=False):
    """
    From Ultralytics
    Matches predictions to ground truth objects (pred_classes, true_classes) using IoU.

    Args:
        pred_classes (torch.Tensor): Predicted class indices of shape(N,).
        true_classes (torch.Tensor): Target class indices of shape(M,).
        iou (torch.Tensor): An NxM tensor containing the pairwise IoU values for predictions and ground of truth
        use_scipy (bool): Whether to use scipy for matching (more precise).

    Returns:
        (torch.Tensor): Correct tensor of shape(N,1) for 1 IoU thresholds.
    """
    # Dx10 matrix, where D - detections, 10 - IoU thresholds
    correct = np.zeros((pred_classes.shape[0], 1)).astype(bool)
    # LxD matrix where L - labels (rows), D - detections (columns)
    correct_class = true_classes[:, None] == pred_classes
    iou = iou * correct_class  # zero out the wrong classes
    iou = iou.cpu().numpy()
    threshold = IOU_THRESHOLD
    #for i, threshold in enumerate(self.iouv.cpu().tolist()):
    if use_scipy:
        # WARNING: known issue that reduces mAP in https://github.com/ultralytics/ultralytics/pull/4708
        import scipy  # scope import to avoid importing for all commands
        cost_matrix = iou * (iou >= threshold)
        if cost_matrix.any():
            labels_idx, detections_idx = scipy.optimize.linear_sum_assignment(cost_matrix, maximize=True)
            valid = cost_matrix[labels_idx, detections_idx] > 0
            if valid.any():
                correct[detections_idx[valid]] = True
    else:
        matches = np.nonzero(iou >= threshold)  # IoU > threshold and classes match
        matches = np.array(matches).T
        if matches.shape[0]:
            if matches.shape[0] > 1:
                matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                # matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int)] = True
    return torch.tensor(correct, dtype=torch.bool, device=pred_classes.device)



# ====== FUNCTIONS FOR PLOTS ======

def plot_confusions_matrix(TP, FP, FN, TN, conf_threshold_i, dataset):
    # Confusion matrix at confidence_threshold = 0.102
    conf_threshold = round(CONF_THRESHOLDS[conf_threshold_i], 2)
    save_dir = os.path.join(SAVE_DIR, 'custom_confusion_matrix_' + dataset + '_conf_' + str(conf_threshold) + '.jpg')
    print(save_dir)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6), tight_layout=True)
    cf_matrix = [[TP[conf_threshold_i], FP[conf_threshold_i]],[FN[conf_threshold_i], TN[conf_threshold_i]]]
    sns.heatmap(cf_matrix, annot=True, cmap='Blues', xticklabels=['Bird', 'Background'], yticklabels=['Bird', 'Background'])
    plt.title(f'Confusion matrix on dataset {dataset}, at thresholds iou={IOU_THRESHOLD}, confidence={conf_threshold}')
    plt.xlabel('Groundtruths')
    plt.ylabel('Predictions')
    plt.show()
    fig.savefig(save_dir, dpi=250)


def plot_precision(precision, dataset):
    save_dir = os.path.join(SAVE_DIR, 'custom_P_curve_' + dataset + '.jpg')

    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    ax.plot(CONF_THRESHOLDS, precision, linewidth=1) #, label=f'{names[i]} {ap[i, 0]:.3f}')  # plot(recall, precision)
    ax.set_xlabel('Confidence')
    ax.set_ylabel('Precision')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    #ax.legend(bbox_to_anchor=(1.04, 1), loc='upper left')
    ax.set_title(f'Precision Curve on dataset {dataset}')
    fig.savefig(save_dir, dpi=250)
    plt.show()
    plt.close(fig)


def plot_recall(recall, dataset):
    save_dir = os.path.join(SAVE_DIR, 'custom_R_curve_' + dataset + '.jpg')

    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    ax.plot(CONF_THRESHOLDS, recall, linewidth=1) #, label=f'{names[i]} {ap[i, 0]:.3f}')  # plot(recall, precision)
    ax.set_xlabel('Confidence')
    ax.set_ylabel('Recall')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    #ax.legend(bbox_to_anchor=(1.04, 1), loc='upper left')
    ax.set_title(f'Recall Curve on dataset {dataset}')
    fig.savefig(save_dir, dpi=250)
    plt.show()
    plt.close(fig)

def plot_pr(precision, recall, dataset):
    save_dir = os.path.join(SAVE_DIR, 'custom_PR_curve_' + dataset + '.jpg')

    # Compute average Precision
    area = metrics.auc(y=precision, x=recall)
    
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    ax.plot(recall, precision, linewidth=1) #, label=f'{names[i]} {ap[i, 0]:.3f}')  # plot(recall, precision)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    #ax.legend(loc='upper left', bbox_to_anchor=(1.04, 1))
    ax.text(0.05, 0.95, f'Average Precision = {round(area, 2)}', transform=ax.transAxes, fontsize=14)
    ax.set_title(f'Precision-Recall Curve on dataset {dataset}')
    fig.savefig(save_dir, dpi=250)
    plt.show()
    plt.close(fig)


def plot_f1(f1_score, dataset):
    save_dir = os.path.join(SAVE_DIR, 'custom_F1score_curve_' + dataset + '.jpg')

    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    ax.plot(CONF_THRESHOLDS, f1_score, linewidth=1) #, label=f'{names[i]} {ap[i, 0]:.3f}')  # plot(recall, precision)
    ax.set_xlabel('Confidence')
    ax.set_ylabel('F1 score')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    #ax.legend(bbox_to_anchor=(1.04, 1), loc='upper left')
    ax.set_title(f'F1 score Curve on dataset {dataset}')
    fig.savefig(save_dir, dpi=250)
    plt.show()
    plt.close(fig)




# ====== EVALUATION ======

nb_subdataset = sum(len(lst) for lst in SUBDATASETS.values())
# tensors to store evaluation results: each line is a dataset, columns are confidence thresholds
final_TP = torch.zeros((nb_subdataset, NB_CONF_THRESHOLDS), dtype=torch.float32)
final_FN = torch.zeros((nb_subdataset, NB_CONF_THRESHOLDS), dtype=torch.float32)
final_FP = torch.zeros((nb_subdataset, NB_CONF_THRESHOLDS), dtype=torch.float32)
final_TN = torch.zeros((nb_subdataset, NB_CONF_THRESHOLDS), dtype=torch.float32)


# === Evaluation per dataset

for domain_i, domain in enumerate(SUBDATASETS.keys()):

    if TASK=='detect':
        img_path = IMG_PATH
    else:
        img_path = IMG_PATH + domain

    for dataset_i_, dataset in enumerate(SUBDATASETS[domain]):

        dataset_i = dataset_i_*(domain_i+1)
        print("DATASET : ", dataset, " dataset nb:", dataset_i)

        if TASK=='detect':
            img_list = os.listdir(img_path + dataset + '/images/')
        else:
            img_list = os.listdir(img_path + '/images/')
            img_list = [file for file in img_list if file.startswith(dataset)]
        #print("LEN OF IMG_LIST", len(img_list)) # For test

        TP = torch.zeros((NB_CONF_THRESHOLDS, len(img_list)), dtype=torch.float32)
        FN = torch.zeros((NB_CONF_THRESHOLDS, len(img_list)), dtype=torch.float32)
        FP = torch.zeros((NB_CONF_THRESHOLDS, len(img_list)), dtype=torch.float32)

        for conf_i, conf_threshold in enumerate(CONF_THRESHOLDS):

            for img_i, img in enumerate(img_list):

                # Apply model
                result = model.predict(
                    #model = 'runs/detect/pfeifer_yolov8n_70epoch_default_batch32_dropout0.3',
                    source = [os.path.join(img_path, 'images', img_) for img_ in [img]],
                    #source = [os.path.join(img_path, 'images', img)],
                    conf = conf_threshold, 
                    iou = IOU_THRESHOLD,
                    show=False,
                    save=False
                )
                result = result[0]

                """
                ## == For test
                detection_boxes = []
                save_path = SAVE_DIR + '/CUSTOM_TEST_' + os.path.basename(result.path).split('.jpg')[0] + '.jpg'
                for detect in range(len(result.boxes.cls)):
                    det = {}
                    det['conf'] = result.boxes.conf[detect].cpu()
                    det['category'] = result.boxes.cls[detect].cpu()
                    coords = result.boxes.xywhn[detect].cpu()
                    det['bbox'] = [coords[0]-coords[2]/2, coords[1]-coords[3]/2, coords[2], coords[3]]
                    detection_boxes.append(det)
                    
                im_path = os.path.join(img_path + '/images/', img)
                print("OTHE PATH TO IMG", im_path)
                visutils.draw_bounding_boxes_on_file(im_path, save_path, detection_boxes,
                                                confidence_threshold=0.0, detector_label_map=None,
                                                thickness=1,expansion=0, colormap=['Red'])

                selected_label = img_path + '/labels/' + os.path.basename(result.path).split('.jpg')[0] + '.txt'
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
                    save_path2 = SAVE_DIR + '/CUSTOM_TEST_label_' + os.path.basename(result.path).split('.hpg')[0] + '.jpg'
                    visutils.draw_bounding_boxes_on_file(save_path, save_path2, detection_boxes,
                                                    confidence_threshold=0.0, detector_label_map=None,
                                                    thickness=1,expansion=0, colormap=['SpringGreen'])
                                                    
                    # Remove predictions-only images
                    os.remove(save_path)
                
                ## =========== FIN TEST ========== #
                """

                pred_classes = result.boxes.cls.cpu()
                pred_bboxes = result.boxes.xyxyn.cpu()
                #print("NB of predictions: ", len(pred_bboxes)) # For test

                # TODO: see if there's no easier way to read true labels, maybe take a yolov8 method
                selected_label = img_path + '/labels/' + os.path.basename(result.path).split('.jpg')[0] + '.txt'
                true_bboxes = torch.tensor([], dtype=torch.float32)
                true_classes = torch.tensor([], dtype=torch.float32)
                if os.path.exists(selected_label):
                    df = pd.read_csv(selected_label, sep='\t', header=None, index_col=False)
                    for irow, row in df.iterrows():  
                        true_classes = torch.cat((true_classes, torch.tensor([row[0]], dtype=torch.float32)), 0)
                        true_bboxes = torch.cat((true_bboxes, torch.tensor([[row[1]-row[3]/2, row[2]-row[4]/2, row[1]+row[3]/2, row[2]+row[4]/2]], dtype=torch.float32)), 0)
                    #print("NB of labels: ", len(true_bboxes)) # For test
                
                    iou = box_iou(true_bboxes, pred_bboxes)
                else:
                    iou = torch.zeros((len(true_bboxes), len(pred_bboxes)), dtype=torch.float32)
                correct = match_predictions(pred_classes, true_classes, iou)  # what they call tp in the code !!!!

                img_TP = correct.sum()
                img_FN = len(true_bboxes) - img_TP
                img_FP = len(pred_bboxes) - img_TP

                """
                # == For test
                print("TP:", img_TP)
                print("FN:", img_FN)
                print("FP:", img_FP)
                """

                TP[conf_i, img_i] = (img_TP)
                FN[conf_i, img_i] = (img_FN)
                FP[conf_i, img_i] = (img_FP)
        
        # Retrieve TP, FP, FN, TN values
        final_TP[dataset_i, :] = torch.sum(TP, dim=1)
        final_FP[dataset_i, :] = torch.sum(FP, dim=1)
        final_FN[dataset_i, :] = torch.sum(FN, dim=1)

        # Compute Precision, Recall & F1-score
        precision = final_TP[dataset_i, :] / (final_TP[dataset_i, :] + final_FP[dataset_i, :] + eps)
        recall = final_TP[dataset_i, :] / (final_TP[dataset_i, :] + final_FN[dataset_i, :] + eps)
        f1_score = 2*(precision*recall)/(precision+recall)

        # === Plot Confusion Matrix at various confidence thresholds
        # Confusion matrix at confidence_threshold = 0.102
        plot_confusions_matrix(final_TP[dataset_i, :], final_FP[dataset_i, :], final_FN[dataset_i, :], final_TN[dataset_i, :], 5, dataset)

        # Confusion matrix at confidence_threshold = 0.204
        plot_confusions_matrix(final_TP[dataset_i, :], final_FP[dataset_i, :], final_FN[dataset_i, :], final_TN[dataset_i, :], 10, dataset)

        # Confusion matrix at confidence_threshold = 0.51
        plot_confusions_matrix(final_TP[dataset_i, :], final_FP[dataset_i, :], final_FN[dataset_i, :], final_TN[dataset_i, :], 25, dataset)


        # === Plot Precision, Recall, PR & F1 score curves
        plot_precision(precision, dataset)
        plot_recall(recall, dataset)
        plot_pr(precision, recall, dataset)
        plot_f1(f1_score, dataset)



# === Global evaluation (on entire dataset)

# Retrieve TP, FP, FN, TN values by summing over axis 0
global_TP = torch.sum(final_TP, dim=0)
global_FP = torch.sum(final_FP, dim=0)
global_FN = torch.sum(final_FN, dim=0)
global_TN = torch.zeros_like(global_TP)


# Compute Precision, Recall & F1-score
precision = global_TP / (global_TP + global_FP + eps)
recall = global_TP / (global_TP + global_FN + eps)
f1_score = 2*(precision*recall)/(precision+recall)

# === Plot Confusion Matrix at various confidence thresholds
dataset = "global"
# Confusion matrix at confidence_threshold = 0.102
plot_confusions_matrix(global_TP, global_FP, global_FN, global_TN, 5, dataset)

# Confusion matrix at confidence_threshold = 0.204
plot_confusions_matrix(global_TP, global_FP, global_FN, global_TN, 10, dataset)

# Confusion matrix at confidence_threshold = 0.51
plot_confusions_matrix(global_TP, global_FP, global_FN, global_TN, 25, dataset)


# === Plot Precision, Recall, PR & F1 score curves
plot_precision(precision, dataset)
plot_recall(recall, dataset)
plot_pr(precision, recall, dataset)
plot_f1(f1_score, dataset)



# === Store results in json dictionnary, for graphs reproducibility ===

eval = {"model": MODEL_NAME,
        "model_path": MODEL_PATH,
        "subtask": SUBTASK,
        "dataset_name": DATASET_NAME,
        "datasets": SUBDATASETS,
        "confidence_thresholds": CONF_THRESHOLDS.tolist() ,
        "iou_threshold": IOU_THRESHOLD,
        "results_metrics": {"TP": final_TP.tolist() ,
                            "FP": final_FP.tolist() ,
                            "FN": final_FN.tolist() ,
                            "TN": final_FN.tolist() },
        "eps": eps}

# Convert and write JSON object to file
fname = 'evaluation_results.json'
with open(os.path.join(SAVE_DIR, fname), 'w') as yaml_file:
    yaml_file.write(yaml.dump(eval, default_flow_style=False))