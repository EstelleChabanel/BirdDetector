import ultralytics
ultralytics.checks()
from ultralytics import YOLO

import torch
import os
import yaml
import pandas as pd
import sys
import argparse
import json
from sklearn import metrics
import numpy as np

from constants import MATCH_IOU_THRESHOLD, NMS_IOU_THRESHOLD, NB_CONF_THRESHOLDS, CONF_THRESHOLDS, EVAL_DATASETS_MAPPING, DATA_PATH, MODELS_PATH
from evaluation_utils import box_iou, match_predictions, plot_confusions_matrix, plot_precision, plot_recall, plot_pr, plot_f1

module_path = os.path.abspath(os.path.join('..'))
module_path = module_path+'/data_preprocessing'

if module_path not in sys.path:
    sys.path.append(module_path)

#import src.data_preprocessing.visualization_utils as visutils

# Use GPU if available
device = "0" if torch.cuda.is_available() else "cpu"
if device == "0":
    torch.cuda.set_device(0) # Set to your desired GPU number


# ============ Parse Arguments ============ #
    
def list_of_strings(arg):
    return arg.split(',')

parser = argparse.ArgumentParser()
parser.add_argument("--model-name", type=str, required=True)
parser.add_argument("--dataset-name", type=str, required=True)
parser.add_argument("--iou", type=float, required=False)
parser.add_argument("--param", type=str, required=True)
parser.add_argument("--params", type=list_of_strings, required=True)

args = parser.parse_args()

if args.iou:
    NMS_IOU_THRESHOLD = args.iou

# ============== INITIALIZE PARAMETERS ============== #
    
# Model specifications
MODEL_NAME = args.model_name #'DAN_pfpe_palm_Adam1e-3_dcLoss1' #'deepcoral_background_lscale16_epochs40_coralgain10' #'pfeifer_penguins_poland_palmyra_10percent_bckgd_yolov8m_120epochs'

# Data
DATASET_NAME = args.dataset_name 
SUBDATASETS = EVAL_DATASETS_MAPPING[DATASET_NAME]['source']
print(SUBDATASETS)
nb_subdataset = len(SUBDATASETS)

eps = 1e-8

APs = torch.zeros((len(args.params), nb_subdataset), dtype=torch.float32)
global_APs = torch.zeros((len(args.params)), dtype=torch.float32)


for p_i, p in enumerate(args.params):


    # ====== Load model & prepare evaluation ====== #

    TASK = 'detect'
    MODEL_PATH = MODELS_PATH + MODEL_NAME + p + '/weights/best.pt'
    IMG_PATH = DATA_PATH + DATASET_NAME + '/test/'
    SAVE_DIR = '/vast/palmer/home.grace/eec42/BirdDetector/runs/detect'
    if not os.path.exists(SAVE_DIR):
        os.mkdir(SAVE_DIR)

    print("model_path of model to load", MODEL_PATH)
    model = YOLO(MODEL_PATH, task=TASK) #.load(MODEL_PATH)

    # ================== EVALUATION ================== #

    # tensors to store evaluation results: each line is a dataset, columns are confidence thresholds
    final_TP = torch.zeros((nb_subdataset, NB_CONF_THRESHOLDS), dtype=torch.float32)
    final_FN = torch.zeros((nb_subdataset, NB_CONF_THRESHOLDS), dtype=torch.float32)
    final_FP = torch.zeros((nb_subdataset, NB_CONF_THRESHOLDS), dtype=torch.float32)
    final_TN = torch.zeros((nb_subdataset, NB_CONF_THRESHOLDS), dtype=torch.float32)

    # === Evaluation per dataset

    dataset_i = 0

    for dataset_i_, dataset in enumerate(SUBDATASETS):

        img_path = IMG_PATH + dataset

        print("DATASET : ", dataset, " dataset nb:", dataset_i)

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
                    iou = NMS_IOU_THRESHOLD,
                    show=False,
                    save=False
                )
                result = result[0]

        
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
        AP_ = metrics.auc(y=precision, x=recall)
        APs[p_i, dataset_i] = AP_

        dataset_i += 1


    # === Global evaluation (on entire dataset)

    # Retrieve TP, FP, FN, TN values by summing over axis 0
    global_TP = torch.sum(final_TP, dim=0)
    global_FP = torch.sum(final_FP, dim=0)
    global_FN = torch.sum(final_FN, dim=0)
    global_TN = torch.zeros_like(global_TP)

    # Compute Precision, Recall & F1-score
    precision = global_TP / (global_TP + global_FP + eps)
    recall = global_TP / (global_TP + global_FN + eps)
    AP_ = metrics.auc(y=precision, x=recall)
    global_APs[p_i] = AP_


APs_per_param = torch.sum(APs, dim=1).numpy()
best_param_indice = np.argmax(APs_per_param)
best_param = args.params[best_param_indice]

eval = {"model": MODEL_NAME,
        "dataset_name": DATASET_NAME,
        "datasets": SUBDATASETS,
        "param": args.param,
        "parameters": args.params,
        "iou_threshold_for_matching": MATCH_IOU_THRESHOLD,
        "iou_threshold_for_NMS": NMS_IOU_THRESHOLD, #args.iou_threshold,
        "Average_Precision_per_dataset": APs.tolist(),
        "global_Average_Precision": global_APs.tolist(),
        "best_param": float(best_param),
        "best_APs": APs[best_param_indice,:].tolist(),
        "eps": eps}


# Convert and write JSON object to file
fname = MODEL_NAME + '_gridsearch.json'
with open(os.path.join(SAVE_DIR, fname), 'w') as yaml_file:
    #yaml_file.write(yaml.dump(eval, default_flow_style=False))
    json.dump(eval, yaml_file)
