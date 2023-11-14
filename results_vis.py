import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from codes.model import UNET3D


image_paths = [f"/content/drive/MyDrive/new_brain/images/img{i}.npy" for i in range(0, 11)]
mask_paths = [f"/content/drive/MyDrive/new_brain/masks/mask{i}.npy" for i in range(0, 11)]

# Load models
model_paths = [
    '/content/drive/MyDrive/bratsxyz.h5',
    '/content/drive/MyDrive/dice_bratsxyz.h5',
    '/content/drive/MyDrive/simCLBratsxyz.h5',
    '/content/drive/MyDrive/infoCLBratsxyz.h5'
]

model_names = ["Our Predicated Mask", "Vanilla Dice", "simCLR", "infoNCE"]

models = [UNET3D(classes=1, dropout_rate=0.2, l2_reg=0.01).call() for _ in range(len(model_paths))]
for i, model_path in enumerate(model_paths):
    models[i].load_weights(model_path)

# Visualize the results for each image and model
z_slice = 64

for image_path, mask_path in zip(image_paths, mask_paths):
    numpy_image = np.load(image_path)
    ground_truth_mask = np.load(mask_path)

    plt.figure(figsize=(20, 4))

    # Plot the original image
    plt.subplot(1, 6, 1)
    plt.title('Original Image')
    plt.imshow(numpy_image[:, :, z_slice, 0], cmap='gray')
    plt.axis('off')

    # Plot the ground truth mask on the original image
    plt.subplot(1, 6, 2)
    plt.title('Ground Truth Mask')
    plt.imshow(numpy_image[:, :, z_slice, 0], cmap='gray')
    plt.imshow(ground_truth_mask[:, :, z_slice], alpha=0.3, cmap='hot')
    plt.axis('off')

    # Plot the segmentation images for each model
    for i, model in enumerate(models, start=0):
        test_img_input = np.expand_dims(numpy_image, axis=0)
        bridge, test_prediction_slices = model.predict(test_img_input)
        test_prediction = test_prediction_slices[0, :, :, :, 0]
        binary_prediction = np.where(test_prediction > 0.5, 1, 0)

        true_positive = np.logical_and(binary_prediction == 1, ground_truth_mask[:, :, :, 0] == 1)
        false_positive = np.logical_and(binary_prediction == 1, ground_truth_mask[:, :, :, 0] == 0)
        false_negative = np.logical_and(binary_prediction == 0, ground_truth_mask[:, :, :, 0] == 1)
        uncertain_mask = np.logical_not(np.logical_or.reduce([true_positive, false_positive, false_negative]))

        plt.subplot(1, 6, i + 3)
        plt.title(f'{model_names[i]} Segmentation')
        # plt.title(f'{model_names[i]} Segmentation')
        plt.imshow(numpy_image[:, :, z_slice, 0], cmap='viridis')
        plt.imshow(binary_prediction[:, :, z_slice], alpha=0.3, cmap='bone')
        plt.imshow(false_negative[:, :, z_slice], alpha=0.3, cmap='autumn')
        plt.imshow(true_positive[:, :, z_slice], alpha=0.3, cmap='winter')
        plt.imshow(false_positive[:, :, z_slice], alpha=0.3, cmap='cool')

        plt.axis('off')

    plt.tight_layout()
    plt.show()