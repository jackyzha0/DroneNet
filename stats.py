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

    iou = intersection / float(boxesArea + labelsArea - intersection + 1e-8)

    # return the intersection over union value
    return iou

def confFilter(boxes, labels, db, conf_thresh):
    x_pred, y_pred, w_pred, h_pred, conf_pred, classes_pred = db.seperate_labels(boxes)
    x_true, y_true, w_true, h_true, conf_true, classes_true = db.seperate_labels(labels)

    boxes = []
    labels = []

    for x in range(db.sx):
        for y in range(db.sy):
            for B in range(db.B):
                if conf_pred[x][y][i] > conf_thresh:
                    bounds = db.xywh_to_p1p2([x_pred[x][y][i], y_pred[x][y][i], w_pred[x][y][i], h_pred[x][y][i]], x, y)
                    bounds.append(conf_pred[x][y][i])
                    bounds.append(classes_pred[x][y][i*db.NUM_CLASSES:i*db.NUM_CLASSES+4])
                    boxes.append(bounds)
                if conf_true[x][y][i] > conf_thresh:
                    bounds = db.xywh_to_p1p2([x_true[x][y][i], y_true[x][y][i], w_true[x][y][i], h_true[x][y][i]], x, y)
                    bounds.append(conf_true[x][y][i])
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

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2. * ((precision * recall) / (precision + recall))

    return TP, FP, FN, precision, recall, f1
