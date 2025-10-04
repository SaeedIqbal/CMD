import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve
import torch
import torch.nn.functional as F

def compute_nap50(pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels, iou_threshold=0.5):
    """
    Compute novel-class Average Precision at IoU=0.5 (nAP50).
    
    Args:
        pred_boxes (list of np.ndarray): [N, 4] predicted boxes per image (xyxy)
        pred_scores (list of np.ndarray): [N] confidence scores per image
        pred_labels (list of np.ndarray): [N] predicted labels per image (0=normal, 1=defect)
        gt_boxes (list of np.ndarray): [M, 4] ground truth boxes per image (xyxy)
        gt_labels (list of np.ndarray): [M] ground truth labels per image (1=defect)
        iou_threshold (float): IoU threshold for matching (default: 0.5)
    
    Returns:
        float: nAP50 score
    """
    # Flatten all predictions and ground truths
    all_pred_boxes = []
    all_pred_scores = []
    all_pred_labels = []
    all_gt_boxes = []
    all_gt_labels = []
    
    for i in range(len(pred_boxes)):
        # Only consider defect predictions (label=1)
        defect_pred_mask = pred_labels[i] == 1
        if np.any(defect_pred_mask):
            all_pred_boxes.append(pred_boxes[i][defect_pred_mask])
            all_pred_scores.append(pred_scores[i][defect_pred_mask])
            all_pred_labels.append(pred_labels[i][defect_pred_mask])
        else:
            all_pred_boxes.append(np.empty((0, 4)))
            all_pred_scores.append(np.empty(0))
            all_pred_labels.append(np.empty(0))
        
        # Only consider defect ground truths (label=1)
        defect_gt_mask = gt_labels[i] == 1
        if np.any(defect_gt_mask):
            all_gt_boxes.append(gt_boxes[i][defect_gt_mask])
            all_gt_labels.append(gt_labels[i][defect_gt_mask])
        else:
            all_gt_boxes.append(np.empty((0, 4)))
            all_gt_labels.append(np.empty(0))
    
    # Compute AP for defect class (label=1)
    ap = _compute_ap(
        all_pred_boxes,
        all_pred_scores,
        all_gt_boxes,
        iou_threshold
    )
    return ap

def _compute_ap(pred_boxes_list, pred_scores_list, gt_boxes_list, iou_threshold=0.5):
    """
    Compute Average Precision for a single class.
    """
    # Flatten predictions and ground truths
    pred_boxes = np.concatenate(pred_boxes_list, axis=0) if pred_boxes_list else np.empty((0, 4))
    pred_scores = np.concatenate(pred_scores_list, axis=0) if pred_scores_list else np.empty(0)
    gt_boxes = np.concatenate(gt_boxes_list, axis=0) if gt_boxes_list else np.empty((0, 4))
    
    if len(pred_boxes) == 0 and len(gt_boxes) == 0:
        return 1.0
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return 0.0
    
    # Sort predictions by score
    sorted_indices = np.argsort(-pred_scores)
    pred_boxes = pred_boxes[sorted_indices]
    pred_scores = pred_scores[sorted_indices]
    
    # Compute IoUs
    ious = _compute_iou(pred_boxes, gt_boxes)  # [N_pred, N_gt]
    
    # Match predictions to ground truths
    gt_matched = np.zeros(len(gt_boxes), dtype=bool)
    tp = np.zeros(len(pred_boxes), dtype=bool)
    fp = np.zeros(len(pred_boxes), dtype=bool)
    
    for i in range(len(pred_boxes)):
        # Find best match
        if len(gt_boxes) > 0:
            max_iou = np.max(ious[i])
            max_idx = np.argmax(ious[i])
            if max_iou >= iou_threshold and not gt_matched[max_idx]:
                tp[i] = True
                gt_matched[max_idx] = True
            else:
                fp[i] = True
        else:
            fp[i] = True
    
    # Compute precision and recall
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    recall = tp_cumsum / len(gt_boxes) if len(gt_boxes) > 0 else tp_cumsum
    precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)
    
    # Compute AP using 11-point interpolation
    ap = 0.0
    for t in np.arange(0.0, 1.1, 0.1):
        precisions = precision[recall >= t]
        p = np.max(precisions) if len(precisions) > 0 else 0.0
        ap += p / 11.0
    
    return ap

def _compute_iou(boxes1, boxes2):
    """
    Compute IoU between two sets of boxes.
    
    Args:
        boxes1 (np.ndarray): [N, 4] (x1, y1, x2, y2)
        boxes2 (np.ndarray): [M, 4] (x1, y1, x2, y2)
    
    Returns:
        np.ndarray: [N, M] IoU matrix
    """
    # Intersection
    x1 = np.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
    y1 = np.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
    x2 = np.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
    y2 = np.minimum(boxes1[:, None, 3], boxes2[None, :, 3])
    
    inter_w = np.maximum(0, x2 - x1)
    inter_h = np.maximum(0, y2 - y1)
    inter_area = inter_w * inter_h
    
    # Areas
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    # IoU
    union = area1[:, None] + area2[None, :] - inter_area
    iou = inter_area / (union + 1e-8)
    return iou

def compute_pixel_auroc(pred_masks, gt_masks):
    """
    Compute Pixel-level AUROC.
    
    Args:
        pred_masks (np.ndarray): [N, H, W] predicted anomaly scores (0-1)
        gt_masks (np.ndarray): [N, H, W] ground truth masks (0=normal, 1=anomaly)
    
    Returns:
        float: AUROC score
    """
    # Flatten
    y_true = gt_masks.flatten()
    y_score = pred_masks.flatten()
    
    # Remove pixels where ground truth is undefined (if any)
    valid_mask = (y_true >= 0)
    y_true = y_true[valid_mask]
    y_score = y_score[valid_mask]
    
    if len(np.unique(y_true)) < 2:
        return 1.0 if y_true.sum() == 0 else 0.0
    
    return roc_auc_score(y_true, y_score)

def compute_localization_f1(pred_masks, gt_masks, threshold=None):
    """
    Compute Localization F1-Score.
    
    Args:
        pred_masks (np.ndarray): [N, H, W] predicted anomaly scores (0-1)
        gt_masks (np.ndarray): [N, H, W] ground truth masks (0=normal, 1=anomaly)
        threshold (float, optional): Binarization threshold. If None, use optimal threshold.
    
    Returns:
        float: F1-Score
    """
    # Flatten
    y_true = gt_masks.flatten()
    y_score = pred_masks.flatten()
    
    # Remove undefined pixels
    valid_mask = (y_true >= 0)
    y_true = y_true[valid_mask]
    y_score = y_score[valid_mask]
    
    if len(np.unique(y_true)) < 2:
        return 1.0 if y_true.sum() == 0 else 0.0
    
    if threshold is None:
        # Optimal threshold via precision-recall curve
        precision, recall, thresholds = precision_recall_curve(y_true, y_score)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        threshold = thresholds[np.argmax(f1_scores)]
    
    y_pred = (y_score >= threshold).astype(int)
    return f1_score(y_true, y_pred)

# Convenience functions for PyTorch tensors
def tensor_to_numpy(tensor):
    """Convert torch tensor to numpy array."""
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return tensor

def compute_metrics_torch(pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels, 
                         pred_masks=None, gt_masks=None):
    """
    Compute all metrics from PyTorch tensors.
    
    Args:
        pred_boxes (list of torch.Tensor): Predicted boxes
        pred_scores (list of torch.Tensor): Prediction scores
        pred_labels (list of torch.Tensor): Predicted labels
        gt_boxes (list of torch.Tensor): Ground truth boxes
        gt_labels (list of torch.Tensor): Ground truth labels
        pred_masks (torch.Tensor, optional): [N, H, W] predicted masks
        gt_masks (torch.Tensor, optional): [N, H, W] ground truth masks
    
    Returns:
        dict: {
            'nap50': float,
            'pixel_auroc': float (if masks provided),
            'f1_score': float (if masks provided)
        }
    """
    # Convert to numpy
    pred_boxes_np = [tensor_to_numpy(b) for b in pred_boxes]
    pred_scores_np = [tensor_to_numpy(s) for s in pred_scores]
    pred_labels_np = [tensor_to_numpy(l) for l in pred_labels]
    gt_boxes_np = [tensor_to_numpy(b) for b in gt_boxes]
    gt_labels_np = [tensor_to_numpy(l) for l in gt_labels]
    
    metrics = {}
    
    # nAP50
    metrics['nap50'] = compute_nap50(
        pred_boxes_np, pred_scores_np, pred_labels_np,
        gt_boxes_np, gt_labels_np
    )
    
    # Pixel AUROC and F1 (if masks provided)
    if pred_masks is not None and gt_masks is not None:
        pred_masks_np = tensor_to_numpy(pred_masks)
        gt_masks_np = tensor_to_numpy(gt_masks)
        
        # Ensure masks are [N, H, W]
        if pred_masks_np.ndim == 3 and gt_masks_np.ndim == 3:
            metrics['pixel_auroc'] = compute_pixel_auroc(pred_masks_np, gt_masks_np)
            metrics['f1_score'] = compute_localization_f1(pred_masks_np, gt_masks_np)
    
    return metrics