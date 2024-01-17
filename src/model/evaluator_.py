#import ultralytics
#ultralytics.checks()
#from ultralytics import YOLO

import yolo
from yolo import YOLO

import torch
import os
import yaml
import pandas as pd
import sys
import argparse

from constants import IOU_THRESHOLD, NB_CONF_THRESHOLDS, CONF_THRESHOLDS, EVAL_DATASETS_MAPPING, DATA_PATH, MODELS_PATH
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
    
parser = argparse.ArgumentParser()
parser.add_argument("--model-name", type=str, required=True)
parser.add_argument("--subtask", type=str, required=True)
parser.add_argument("--dataset-name", type=str, required=True)
args = parser.parse_args()


# ============== INITIALIZE PARAMETERS ============== #

# Model specifications
MODEL_NAME = args.model_name #'DAN_pfpe_palm_Adam1e-3_dcLoss1' #'deepcoral_background_lscale16_epochs40_coralgain10' #'pfeifer_penguins_poland_palmyra_10percent_bckgd_yolov8m_120epochs'
SUBTASK = args.subtask #'domainclassifier' #Choose between: #'deepcoral_detect' #'detect'

# Data
DATASET_NAME = args.dataset_name 
SUBDATASETS = EVAL_DATASETS_MAPPING[DATASET_NAME] 

eps = 1e-8


# ====== Load model & prepare evaluation ====== #

TASK = 'detect'
MODEL_PATH = MODELS_PATH + MODEL_NAME + '/weights/best.pt'
IMG_PATH = DATA_PATH + DATASET_NAME + '/test/'
SAVE_DIR = os.path.join('/vast/palmer/home.grace/eec42/BirdDetector/runs/detect', MODEL_NAME, 'eval')
os.mkdir(SAVE_DIR)

model = YOLO('yolov8m_domainclassifier.yaml', task=TASK, subtask=SUBTASK ).load(MODEL_PATH)


# ================== EVALUATION ================== #

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
        plot_confusions_matrix(final_TP[dataset_i, :], final_FP[dataset_i, :], final_FN[dataset_i, :], final_TN[dataset_i, :], 5, dataset, SAVE_DIR)

        # Confusion matrix at confidence_threshold = 0.204
        plot_confusions_matrix(final_TP[dataset_i, :], final_FP[dataset_i, :], final_FN[dataset_i, :], final_TN[dataset_i, :], 10, dataset, SAVE_DIR)

        # Confusion matrix at confidence_threshold = 0.51
        plot_confusions_matrix(final_TP[dataset_i, :], final_FP[dataset_i, :], final_FN[dataset_i, :], final_TN[dataset_i, :], 25, dataset, SAVE_DIR)


        # === Plot Precision, Recall, PR & F1 score curves
        plot_precision(precision, dataset, SAVE_DIR)
        plot_recall(recall, dataset, SAVE_DIR)
        plot_pr(precision, recall, dataset, SAVE_DIR)
        plot_f1(f1_score, dataset, SAVE_DIR)



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
plot_confusions_matrix(global_TP, global_FP, global_FN, global_TN, 5, dataset, SAVE_DIR)

# Confusion matrix at confidence_threshold = 0.204
plot_confusions_matrix(global_TP, global_FP, global_FN, global_TN, 10, dataset, SAVE_DIR)

# Confusion matrix at confidence_threshold = 0.51
plot_confusions_matrix(global_TP, global_FP, global_FN, global_TN, 25, dataset, SAVE_DIR)


# === Plot Precision, Recall, PR & F1 score curves
plot_precision(precision, dataset, SAVE_DIR)
plot_recall(recall, dataset, SAVE_DIR)
plot_pr(precision, recall, dataset, SAVE_DIR)
plot_f1(f1_score, dataset, SAVE_DIR)



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