import tensorflow as tf
import numpy as np

def reformboxes(boxes, labels):
    '''
    Takes boxes in form [5, 5, B(C+5)] and reshapes to form [25, 27]
    '''

    # Convert pred to [25, 27]

    x_, y_, w_, h_, conf_, prob_ = tf.split(net, [B, B, B, B, B, B * C], axis=3)
    # Convert label to [25, 34]
    # Slice label to [25,27]
    # tf.split to x1y1x2p2 form
    # x = [25, 3]

    #pred_x1, pred_y1, pred_x2, pred_y2 = tf.split(boxes, 4, axis = 1)
    #true_x1, true_y1, true_x2, true_y2 = tf.split(labels, 4, axis = 1)


def IOU(boxes, labels):
    '''
    Returns IOU (Intersection over Union) scores between array of predicted boxes and array of box labels
    '''
    xA = max(boxes[0], labels[0])
    yA = max(boxes[1], labels[1])
    xB = min(boxes[2], labels[2])
    yB = min(boxes[3], labels[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxesArea = (boxes[2] - boxes[0]) * (boxes[3] - boxes[1])
    labelsArea = (labels[2] - labels[0]) * (labels[3] - labels[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxesArea + labelsArea - interArea + 1e-8)

    # return the intersection over union value
    return iou

def non_max_suppression(boxes, conf_scores, iou_thresh = 0.5, conf_thresh = 0.7):
    '''
    Return at most max_box_out non-max suppressed bounding boxes. Eliminates all boxes with
    confidence < conf_thresh and IOU > 0.5 with another box of the same class
    '''

    indices = tf.image.non_max_suppression(boxes, conf_scores, max_output_size = max_box_out, iou_threshold = iou_thresh, score_threshold = conf_thresh)
    boxes = tf.gather(boxes, indices)
    return boxes

def get_stats(boxes, labels, iou_thresh = 0.5):
    '''
    Returns number of True Positives (TP), False Positives (FP), and False Negatives (FN)
    given prediction boxes and labels. True Negatives (TN) are not recorded
    '''

    TP, FP, FN = 0., 0., 0.

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

    return TP, FP, FN

def get_all(boxes,labels):
    '''
    Takes all predicted boxes [bn, sx, sy, B(C+4)] and all labels [bn, sx, sy, B(C+7)+1] and returns mAP, precision, recall,
    F1 score, and number of TP / FP / FN
    '''

    #Crop dims
