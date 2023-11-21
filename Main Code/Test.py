import os
import numpy as np
import tensorflow as tf
from tqdm.auto import tqdm
from glob import glob
import matplotlib.pyplot as plt
from codes.model import UNET3D
from codes.data_load import *
from codes.metrics import *
from codes.Fess import *

images_file = os.listdir("/content/drive/MyDrive/VDI/processed_64/images/")
masks_file = os.listdir("/content/drive/MyDrive/VDI/processed_64/masks/")

npy_images_path = [os.path.join("/content/drive/MyDrive/VDI/processed_64/images/", img_file) for img_file in images_file]
npy_masks_path = [os.path.join("/content/drive/MyDrive/VDI/processed_64/masks/", mas_file) for mas_file in masks_file]


test_data_gen = mini_batches_(npy_images_path[369:420], npy_masks_path[369:420],5)

model = UNET3D(classes=1, dropout_rate=0.2, l2_reg=0.01).call()
model.load_weights('/content/drive/MyDrive/Bratsxyz.h5')

dice_test = []
pre_test = []
batch_loss_test=[]
se_test=[]
spe_test = []
io_test = []
model_test = model
for batch_idx, (images, masks) in enumerate(tqdm(test_data_gen)):
    predictions, logits = model(images)

    loss = dice_loss2(masks,logits)
    batch_loss_test.append(loss)

    dice_test.append(dice_coef(masks, logits))
    pre_test.append(precision(masks, logits))
    se_test.append(sensitivity(masks, logits))
    spe_test.append(specificity(masks, logits))
    io_test.append(iou(masks, logits))

batch_loss_test = np.array(batch_loss_test).mean()
dice_test = np.array(dice_test).mean()
pre_test =np.array(pre_test).mean()
se_test =  np.array(se_test).mean()
spe_test =  np.array(spe_test).mean()
io_test =  np.array(io_test).mean()

print("------------------------>Test<------------------------")
print(" Loss: {} , Dice Coeff: {}\n\n, Precision: {} Sensitivity: {} \n\n Specificity: {} , IOU: {}".format(batch_loss_test,dice_test,pre_test,se_test,spe_test,io_test))