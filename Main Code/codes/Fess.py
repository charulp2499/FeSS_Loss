import os
import numpy as np
import tensorflow as tf
from glob import glob

def contrastive_loss(prev, pres, temperature=0.5, lr=0.0001):
    prev = tf.math.l2_normalize(prev, axis=[1, 2, 3])
    pres = tf.math.l2_normalize(pres, axis=[1, 2, 3])

    similarity = tf.reduce_sum(prev * pres, axis=[1, 2, 3])
    numerator = tf.exp(similarity / temperature)
    
    denominator = tf.reduce_sum(tf.exp(similarity / temperature))
    loss_con = -tf.math.log(numerator / denominator)
    loss_con = tf.reduce_mean(loss_con)
    loss_con *= lr
    
    return loss_con

def dice_loss2(y_true, y_pred):
    smooth = 1e-5
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice

def combined_loss(y_true, y_pred, prev, pres, temperature=0.5, dice_weight=0.5):
    dice_loss_value = dice_loss2(y_true, y_pred)
    con_loss_value = contrastive_loss(prev, pres, temperature=temperature)
    total_loss = dice_weight * dice_loss_value + (1-dice_weight) * con_loss_value
    return total_loss