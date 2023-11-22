# FESS Loss: Feature-Enhanced Spatial Segmentation Loss
<div style="text-align: justify">

## Introduction
This repository contains the implementation of the Feature-Enhanced Spatial Segmentation Loss (FESS Loss) proposed in the research paper titled "FESS Loss: Feature-Enhanced Spatial Segmentation Loss for Optimizing Medical Image Analysis." FESS Loss is designed to improve the accuracy and precision of medical image segmentation by combining contrastive learning with the spatial accuracy provided by the Dice loss.

<!-- <img src="Readme_Supply\flowchart.svg" alt="Flowchart" /> -->
<img src="https://raw.githubusercontent.com/charulp2499/FeSS_Loss/main/Readme_Supply/Flowchart.svg" alt="Flowchart" />


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


<!-- Feel free to explore each module and adapt the code to fit your project's requirements. Refer to individual script comments for more detailed information. -->


## Experiments and Results


## Contributors
Charulkumar Chodvadiya, Navyansh Mahla, Kinshuk Gaurav Singh, Kshitij Sharad Jadhav

## License
This project is licensed under the [MIT](LICENSE).



