import torch
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

from constants import MATCH_IOU_THRESHOLD, NB_CONF_THRESHOLDS, CONF_THRESHOLDS


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
    threshold = MATCH_IOU_THRESHOLD
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

def plot_confusions_matrix(TP, FP, FN, TN, conf_threshold_i, dataset, SAVE_DIR):
    # Confusion matrix at confidence_threshold = 0.102
    conf_threshold = round(CONF_THRESHOLDS[conf_threshold_i], 2)
    save_dir = os.path.join(SAVE_DIR, 'custom_confusion_matrix_' + dataset + '_conf_' + str(conf_threshold) + '.jpg')
    print(save_dir)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6), tight_layout=True)
    cf_matrix = [[TP[conf_threshold_i], FP[conf_threshold_i]],[FN[conf_threshold_i], TN[conf_threshold_i]]]
    sns.heatmap(cf_matrix, annot=True, cmap='Blues', xticklabels=['Bird', 'Background'], yticklabels=['Bird', 'Background'])
    #plt.title(f'Confusion matrix on dataset {dataset}, at thresholds iou={NMS_IOU_THRESHOLD}, confidence={conf_threshold}')
    plt.title(f'Confusion matrix on dataset {dataset}, at threshold confidence={conf_threshold}')
    plt.xlabel('Groundtruths')
    plt.ylabel('Predictions')
    plt.show()
    fig.savefig(save_dir, dpi=250)


def plot_precision(precision, dataset, SAVE_DIR):
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


def plot_recall(recall, dataset, SAVE_DIR):
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

def plot_pr(precision, recall, dataset, SAVE_DIR):
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


def plot_f1(f1_score, dataset, SAVE_DIR):
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



