import tensorflow as tf
import numpy as np

def IOU(boxes, labels):
    '''
    Returns IOU (Intersection over Union) scores between array of predicted boxes and array of box labels
    '''
    xA = max(boxes[0], labels[0])
    yA = max(boxes[1], labels[1])
    xB = min(boxes[2], labels[2])
    yB = min(boxes[3], labels[3])

    intersection = max(0, xB - xA) * max(0, yB - yA)

    boxesArea = (boxes[2] - boxes[0]) * (boxes[3] - boxes[1])
    labelsArea = (labels[2] - labels[0]) * (labels[3] - labels[1])

    try:
        iou = intersection / float(boxesArea + labelsArea - intersection)
    except ZeroDivisionError:
        iou = 0

    return iou

def confFilter(boxes, labels, db, conf_thresh):
    x_pred, y_pred, w_pred, h_pred, conf_pred, classes_pred = db.seperate_labels(boxes)
    x_true, y_true, w_true, h_true, conf_true, classes_true = db.seperate_labels(labels)

    boxes = []
    labels = []

    for x in range(db.sx):
        for y in range(db.sy):
            for i in range(db.B):
                if conf_pred[x][y][i] > conf_thresh:
                    bounds = db.xywh_to_p1p2([x_pred[x][y][i], y_pred[x][y][i], w_pred[x][y][i], h_pred[x][y][i]], x, y)
                    bounds.append(classes_pred[x][y][i*db.NUM_CLASSES:i*db.NUM_CLASSES+4])
                    boxes.append(bounds)
                if conf_true[x][y][i] > conf_thresh:
                    bounds = db.xywh_to_p1p2([x_true[x][y][i], y_true[x][y][i], w_true[x][y][i], h_true[x][y][i]], x, y)
                    bounds.append(classes_true[x][y][i*db.NUM_CLASSES:i*db.NUM_CLASSES+4])
                    labels.append(bounds)
    return boxes, labels

def stats(boxes, labels, db, iou_thresh = 0.5, conf_thresh = 0.7):
    '''
    Returns number of True Positives (TP), False Positives (FP), and False Negatives (FN)
    given prediction boxes and labels. True Negatives (TN) are not recorded
    '''

    TP, FP, FN = 0., 0., 0.

    boxes, labels = confFilter(boxes, labels, db, conf_thresh)
    #boxes = non_max_suppression(boxes, iou_thresh, db)

    for label_box in labels:
        for pred_box in boxes:
            check = False

            iou = IOU(label_box, pred_box)
            if iou > iou_thresh:
                TP += 1
                check = True
            else:
                FP += 1
                check = True
        if not check:
            FN += 1

    try:
        precision = TP / (TP + FP)
    except ZeroDivisionError:
        precision = 0

    try:
        recall = TP / (TP + FN)
    except ZeroDivisionError:
        recall = 0

    try:
        f1 = 2. * ((precision * recall) / (precision + recall))
    except ZeroDivisionError:
        f1 = 0

    return TP, FP, FN, precision, recall, f1

# def non_max_suppression(boxes, iou_thresh, db):
#
#     for c in range(db.NUM_CLASSES):
#         conf = boxes[:, 4 + c]
#         final_detections = np.array([detections[0]])
#         for bbox in detections:
#             iou = intersection_over_union(bbox, final_detections)
#             assert (iou >= 0).all() and (iou <= 1).all()
#             overlap_idxs = np.where(iou > 0.5)
#             if overlap_idxs[0].size == 0:
#                 final_detections = np.vstack((final_detections, bbox))
#         return final_detections

# def multIOU(box, targ):
#     src_box_left = box[0]
#     src_box_top = box[1]
#     src_box_right = box[0] + box[2]
#     src_box_bottom = box[1] + box[3]
#
#     targ_left = targ[:,0]
#     targ_top = targ[:,1]
#     targ_right = targ[:,0] + targ[:,2]
#     targ_bottom = targ[:,1] + targ[:,3]
#
#     intersect_width = np.maximum(0, np.minimum(targ_right, src_box_right) - np.maximum(targ_left, src_box_left))
#     intersect_height = np.maximum(0, np.minimum(targ_bottom, src_box_bottom) - np.maximum(targ_top, src_box_top))
#     intersection = intersect_width * intersect_height
#
#     area_src = box[2] * box[3]
#     area_target = targ[:,2] * targ[:,3]
#     union = area_src + area_target - intersection
#
#     return np.divide(intersection, union)
