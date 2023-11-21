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

### File Structure:

### Step-by-step intruction for running files:

## Experiments and Results


## Contributors
Charulkumar Chodvadiya, Navyansh Mahla, Kinshuk Gaurav Singh, Kshitij Sharad Jadhav

## License
<!-- This project is licensed under the [](LICENSE). -->


