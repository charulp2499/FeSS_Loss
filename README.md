# FESS Loss: Feature-Enhanced Spatial Segmentation Loss
<div style="text-align: justify">

## Introduction
This repository contains the implementation of the Feature-Enhanced Spatial Segmentation Loss (FESS Loss) proposed in the research paper titled "FESS Loss: Feature-Enhanced Spatial Segmentation Loss for Optimizing Medical Image Analysis." FESS Loss is designed to improve the accuracy and precision of medical image segmentation by combining contrastive learning with the spatial accuracy provided by the Dice loss.

## Abstract
Medical image segmentation is a critical process in the field of medical imaging, playing a pivotal role in diagnosis, treatment, and research. Conventional methods often struggle with balancing spatial precision and comprehensive feature representation. FESS Loss addresses this challenge by integrating the benefits of contrastive learning with the spatial accuracy inherent in the Dice loss. This README provides an overview of the implementation and usage of FESS Loss.

## Features
- Integration of contrastive learning for improved feature representation
- Enhanced spatial accuracy through Dice loss
- Superior performance in limited annotated data scenarios

## Getting Started

### Prerequisites
- Python 3.12.0
- TensorFlow 2.12.0

### Installation
Clone the repository and install the dependencies:

```bash
git clone https://github.com/charulp2499/FeSS_Loss.git
pip install -r requirements.txt
```
## Datasets

We utilized the high-quality multi-modal MRI scans from the [BraTs 2016](https://www.smir.ch/BRATS/Start2016) and [BraTs 2017](https://www.med.upenn.edu/sbia/brats2017/data.html) datasets, comprising 274 and 285 patients, respectively. With a focus on the FLAIR modality, our dataset encompasses a variety of tumor shapes, sizes, and locations. Furthermore, to evaluate the generalizability of our approach, we incorporated the [AbdomenCT-1K](https://github.com/JunMa11/AbdomenCT-1K) dataset, which consists of over a thousand abdominal CT scans. your work.


## Usage

### Code File Structure

The code is organized into a structured file hierarchy to enhance clarity and maintainability. Here's a breakdown of the main code file structure:

```
│
├───Main Code
│   │───main.py
│   │───results_vis.py
│   │───Test.py
│   └───codes
│       │───data_load.py
│       │───Fess.py
│       │───metrics.py
│       └───model.py
```

### Description:

- **Main Code:**
  - **main.py:** The primary script for training the model using our approach. Running this file initiates the training process.
  - **Test.py:** Script for generating test results using pretrained model weights. Requires input of pretrained model weights.
  - **results_vis.py:** Script for visualizing test results, providing insights into model performance against different baseline models.
  

- **codes:**
  - **data_load.py:** Python script for loading data and splitting data into mini batches.
  - **Fess.py:** **Python script containing the implementation of the Fess loss function, a main component of our approach.**
  - **metrics.py:** Python script for calculating evaluation metrics relevant to the project.
  - **model.py:** Python script defining the main U-Net 3D model used in our approach.

### How to Use:

1. **Training the Model:**
   - Execute `main.py` to initiate the training process using our approach.

2. **Generating Test Results:**
   - Run `Test.py` by providing pretrained model weights as input to generate test results.

## Flow diagram of our approach

<!-- <img src="Readme_Supply\flowchart.svg" alt="Flowchart" /> -->
<img src="https://raw.githubusercontent.com/charulp2499/FeSS_Loss/main/Readme_Supply/Flowchart.svg" alt="Flowchart" />

## Experiments and Results

We used a batch size (N) of 5 to find a suitable equilibrium between computational efficiency and optimal model performance. Additionally, a learning rate (η) of 1e-5 was chosen to facilitate effective model convergence. Other crucial hyperparameters, such as ε (set at 1e-5 to ensure smoothness in dice loss) and Temperature Δ (established at 0.5 for contrastive loss to regulate probability spread and concentration), were meticulously selected to bolster the overall robustness of the model.

For a more in-depth understanding of our methodology, you can explore our codebase. The comparison table below showcases the results in contrast to baselines. Furthermore, we provide an additional diagram illustrating the performance of our approach compared to baseline performance across various training sample sizes.

### Test Results for Segmentation Performance with FESS Loss and Different loss Functions on Medical Image Dataset

| **Dataset**       | **Loss Function** | **DICE coefficient** | **IoU @0.5** | **Precision** | **Specificity** | **Sensitivity** |
|-------------------|-------------------|----------------------|--------------|---------------|------------------|------------------|
| **BraTs 2016**    | Our Loss          | 0.85 ± 0.02          | 0.75 ± 0.03   | 0.80 ± 0.02   | 0.98 ± 0.001    | 0.91 ± 0.01      |
|                   | Vanilla Dice      | 0.69 ± 0.01          | 0.57 ± 0.03   | 0.70 ± 0.04   | 0.97 ± 0.003    | 0.72 ± 0.02      |
|                   | simCLR            | 0.70 ± 0.02          | 0.57 ± 0.01   | 0.80 ± 0.03   | 0.99 ± 0.001    | 0.67 ± 0.02      |
|                   | infoNCE           | 0.71 ± 0.02          | 0.58 ± 0.02   | 0.76 ± 0.03   | 0.99 ± 0.001    | 0.69 ± 0.02      |
| **BraTs 2017**    | Our Loss          | 0.82 ± 0.03          | 0.71 ± 0.02   | 0.75 ± 0.04   | 0.97 ± 0.001    | 0.93 ± 0.01      |
|                   | Vanilla Dice      | 0.76 ± 0.02          | 0.63 ± 0.02   | 0.71 ± 0.04   | 0.97 ± 0.001    | 0.85 ± 0.02      |
|                   | simCLR            | 0.77 ± 0.01          | 0.62 ± 0.03   | 0.80 ± 0.04   | 0.98 ± 0.002    | 0.77 ± 0.01      |
|                   | infoNCE           | 0.77 ± 0.02          | 0.64 ± 0.02   | 0.80 ± 0.03   | 0.98 ± 0.002    | 0.78 ± 0.04      |
| **Combined BraTs**| Our Loss          | 0.83 ± 0.03          | 0.69 ± 0.03   | 0.86 ± 0.04   | 0.99 ± 0.001    | 0.77 ± 0.02      |
|                   | Vanilla Dice      | 0.73 ± 0.02          | 0.59 ± 0.02   | 0.76 ± 0.06   | 0.99 ± 0.002    | 0.74 ± 0.03      |
|                   | simCLR            | 0.71 ± 0.01          | 0.58 ± 0.02   | 0.94 ± 0.03   | 0.99 ± 0.002    | 0.62 ± 0.03      |
|                   | infoNCE           | 0.81 ± 0.01          | 0.68 ± 0.01   | 0.92 ± 0.03   | 0.99 ± 0.001    | 0.74 ± 0.02      |
| **AbdomenCT-1K**  | Our Loss          | 0.63 ± 0.01          | 0.46 ± 0.01   | 0.60 ± 0.02   | 0.96 ± 0.003    | 0.68 ± 0.04      |
|                   | Vanilla Dice      | 0.60 ± 0.04          | 0.43 ± 0.04   | 0.55 ± 0.05   | 0.95 ± 0.004    | 0.67 ± 0.03      |
|                   | simCLR            | 0.61 ± 0.01          | 0.44 ± 0.01   | 0.60 ± 0.04   | 0.96 ± 0.004    | 0.63 ± 0.04      |
|                   | infoNCE           | 0.61 ± 0.01          | 0.44 ± 0.01   | 0.55 ± 0.03   | 0.95 ± 0.009    | 0.69 ± 0.04      |

### mean and standard error across varied sample sizes plot
<img src="Readme_Supply\SEM_final.png" alt="SEM Diagram" />


## Contributors
Charulkumar Chodvadiya, Kinshuk Gaurav Singh, Navyansh Mahla, Dr Kshitij Jadhav

## License
This project is licensed under the [MIT license](LICENSE).



