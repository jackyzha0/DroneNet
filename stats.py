import tensorflow as tf
import numpy as np

def IOU(boxes, labels):
    '''
    Returns IOU (Intersection over Union) scores between array of predicted boxes [] and array of box labels
    '''
    pred_x1, pred_y1, pred_x2, pred_y2 = tf.split(boxes, 4, axis = 1)
    true_x1, true_y1, true_x2, true_y2 = tf.split(labels, 4, axis = 1)

    xmax = tf.maximum(pred_x1, tf.transpose(true_x1))
    xmin = tf.minimum(pred_x2, tf.transpose(true_x2))
    ymin = tf.minimum(pred_y1, tf.transpose(true_y1))
    ymax = tf.maximum(pred_y2, tf.transpose(true_y2))

    intersection = tf.maximum((xmin - xmax), 0) * tf.maximum((ymin - ymax), 0)

    pred_boxes_area = (pred_x2 - pred_x1) * (pred_y1 - pred_y2)
    labels_boxes_area = (pred_x2 - pred_x1) * (pred_y1 - pred_y2)

    union = (pred_boxes_area + tf.transpose(labels_boxes_area)) - intersection

    return intesection / (union + 1e-4)

def non_max_suppression(boxes, conf_scores, max_box_out = 12, iou_thresh = 0.5, conf_thresh = 0.7):
    '''
    Return at most max_box_out non-max suppressed bounding boxes. Eliminates all boxes with
    confidence < conf_thresh and IOU > 0.5 with another box of the same class
    '''

def get_stats(boxes, labels):
    '''
    Returns number of True Positives (TP), True Negatives (TN), False Positives (FP), and
    False Negatives (FN) given prediction boxes and labels
    '''
