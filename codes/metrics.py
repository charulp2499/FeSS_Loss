import os
import numpy as np
import tensorflow as tf
from glob import glob

def precision(y_true, y_pred):
    true_positives = tf.reduce_sum(tf.cast(y_true * tf.round(tf.clip_by_value(y_pred, 0, 1)), tf.float32))
    predicted_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + tf.keras.backend.epsilon())
    return precision

def recall(y_true, y_pred):
    true_positives = tf.reduce_sum(tf.cast(y_true * tf.round(tf.clip_by_value(y_pred, 0, 1)), tf.float32))
    actual_positives = tf.reduce_sum(y_true)
    recall = true_positives / (actual_positives + tf.keras.backend.epsilon())
    return recall

def specificity(y_true, y_pred):
    true_negatives = tf.reduce_sum(tf.cast((1 - y_true) * tf.round(tf.clip_by_value(1 - y_pred, 0, 1)), tf.float32))
    actual_negatives = tf.reduce_sum(1 - y_true)
    specificity = true_negatives / (actual_negatives + tf.keras.backend.epsilon())
    return specificity

def dice_coef(y_true, y_pred):
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    dice = (2.0 * intersection) / (union)
    return dice

def iou(y_true, y_pred):
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    iou_score = (intersection + tf.keras.backend.epsilon()) / (union + tf.keras.backend.epsilon())
    return iou_score

def sensitivity(y_true, y_pred):
    true_positives = tf.reduce_sum(tf.cast(y_true * tf.round(tf.clip_by_value(y_pred, 0, 1)), tf.float32))
    actual_positives = tf.reduce_sum(y_true)
    sensitivity = true_positives / (actual_positives + tf.keras.backend.epsilon())
    return sensitivity

