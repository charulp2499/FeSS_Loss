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

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Paths for images and masks
images_file = os.listdir("/content/drive/MyDrive/VDI/processed_64/images/")
masks_file = os.listdir("/content/drive/MyDrive/VDI/processed_64/masks/")
npy_images_path = [os.path.join("/content/drive/MyDrive/VDI/processed_64/images/", img_file) for img_file in images_file]
npy_masks_path = [os.path.join("/content/drive/MyDrive/VDI/processed_64/masks/", mas_file) for mas_file in masks_file]

# Set hyperparameters
LR = 0.00001
batch_size = 5
epochs = 150

# Data preparation
data = {}
data["original_data"] = mini_batches_(npy_images_path[:100], npy_masks_path[:100], batch_size)
data["model"] = UNET3D(classes=1, dropout_rate=0.2, l2_reg=0.01).call()
data["optimizer"] = tf.keras.optimizers.Adam(LR)

test_data_gen = mini_batches_(npy_images_path[200:241], npy_masks_path[240:241], batch_size)

# Training loop
plot_dice_losses = []
plot_total_loss = []
plot_dice_co = []
plot_test = []

for epoch in range(epochs):
    print(epoch + 1, '/', epochs)
    model = data['model']
    optimizer = data["optimizer"]
    model.set_weights(data["model"].get_weights())

    dice = []
    pre = []
    batch_loss = []
    se = []
    spe = []
    io = []
    dice_losses = []
    contrastive_losses = []

    for batch_idx, (images, masks) in enumerate(tqdm(data['original_data'])):
        pro2, _ = data['model'](images)
        with tf.GradientTape() as tape:
            pro1, logits = model(images)
            loss_combined = combined_loss(y_true=masks, y_pred=logits, prev=pro2, pres=pro1, temperature=0.5,
                                          dice_weight=0.5)
            loss = loss_combined
            batch_loss.append(loss)

            dice_loss_value = dice_loss2(masks, logits)
            contrastive_loss_value = contrastive_loss(prev=pro2, pres=pro1, temperature=0.5)
            dice_losses.append(dice_loss_value)
            contrastive_losses.append(contrastive_loss_value)

            # Metrics
            dice.append(dice_coef(masks, logits))
            pre.append(precision(masks, logits))
            se.append(sensitivity(masks, logits))
            spe.append(specificity(masks, logits))
            io.append(iou(masks, logits))

        grads = tape.gradient(loss, model.trainable_variables)
        gradients, _ = tf.clip_by_global_norm(grads, clip_norm=1.0)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    batch_loss = np.array(batch_loss).mean()
    dice = np.array(dice).mean()
    pre = np.array(pre).mean()
    se = np.array(se).mean()
    spe = np.array(spe).mean()
    io = np.array(io).mean()

    mean_dice_loss = np.mean(dice_losses)
    mean_contrastive_loss = np.mean(contrastive_losses)
    print(f"\nEpoch {epoch}: Dice Loss = {mean_dice_loss}, Contrastive Loss = {mean_contrastive_loss}")

    print("\nWeighted Combined Loss: {} | Dice Coeff: {}  |\n\n Precision: {} Sensitivity: {} | \n\n Specificity: {} , IOU: {} | \n\n".format(
        batch_loss, dice, pre, se, spe, io))

    plot_dice_losses.append(mean_dice_loss)
    plot_total_loss.append(batch_loss)
    plot_dice_co.append(dice)
    
    if epoch % 10 == 0 and epoch != 0:
        dice_test = []
        pre_test = []
        batch_loss_test = []
        se_test = []
        spe_test = []
        io_test = []
        model_test = data["model"]
        for batch_idx, (images, masks) in enumerate(tqdm(test_data_gen)):
            predictions, logits = model(images)

            loss = dice_loss2(masks, logits)
            batch_loss_test.append(loss)

            dice_test.append(dice_coef(masks, logits))
            pre_test.append(precision(masks, logits))
            se_test.append(sensitivity(masks, logits))
            spe_test.append(specificity(masks, logits))
            io_test.append(iou(masks, logits))

        batch_loss_test = np.array(batch_loss_test).mean()
        dice_test = np.array(dice_test).mean()
        pre_test = np.array(pre_test).mean()
        se_test = np.array(se_test).mean()
        spe_test = np.array(spe_test).mean()
        io_test = np.array(io_test).mean()
        plot_test.append(dice_test)

        print("------------------------>Test<------------------------------------------------")
        print("Epoch: {} , Loss: {} , Dice Coeff: {}\n\n, Precision: {} Sensitivity: {} \n\n Specificity: {} , IOU: {}".format(
            epoch + 1, batch_loss_test, dice_test, pre_test, se_test, spe_test, io_test))
        print("------------------------>Trining Conti<---------------------------------------")

# Save the final model
model.save("/content/drive/MyDrive/" + "brats00xyz.h5")

# Plot Dice Coefficient, Dice loss And Test Dice Coefficient
epochs_with_values = np.arange(0, epochs-batch_size+1, batch_size)  # Adjusted to have the same length as plot_test

assert len(epochs_with_values) == len(plot_test), "Mismatch in the lengths of epochs_with_values and plot_test"
interpolated_values = np.interp(np.arange(0, epochs+1, 1), epochs_with_values, plot_test)

plt.figure(figsize=(10, 5))
plt.plot(plot_dice_losses, label='Train Dice Loss', color='blue')
plt.plot(plot_dice_co, label='Train Dice Coefficient', color='green')
plt.plot(np.arange(0, 151, 1), interpolated_values, label='Test Dice Coefficient')
plt.title('Dice Loss and Coefficient Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Metric Value')
plt.legend()
plt.show()