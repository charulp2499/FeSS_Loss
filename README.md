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

### WOrking Flow Diagram
<!-- <img src="Readme_Supply\flowchart.svg" alt="Flowchart" /> -->
<img src="https://raw.githubusercontent.com/charulp2499/FeSS_Loss/main/Readme_Supply/Flowchart.svg" alt="Flowchart" />


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


## Experiments and Results

### Table: Segmentation Performance
```latex

\begin{table}[!h]
\centering
\footnotesize
\hspace{0.1cm}
\begin{tabularx}{\textwidth}{| p{2.5cm} | X | X | X | X | p{2.3cm} | X |} \hline
\textbf{Dataset} & \textbf{Loss Function} & \textbf{DICE cofficient} &  \textbf{IoU @0.5} & \textbf{Precision} & \textbf{Specificity} & \textbf{Sensitivity}\\
\hline
\hline

\multirow{\textbf{BraTs 2016}} & \textbf{Our Loss} & \textbf{0.85} \pm \textbf{0.02} & \textbf{0.75} \pm \textbf{0.03} & \textbf{0.80} \pm \textbf{0.02} & 0.98 \pm 0.001 & \textbf{0.91} \pm \textbf{0.01}\\
\cline{2-7}
& \textbf{Vanilla Dice} & 0.69 \pm 0.01 & 0.57 \pm 0.03 & 0.70 \pm 0.04 & 0.97 \pm 0.003 & 0.72 \pm 0.02 \\
\cline{2-7}
& \textbf{simCLR} & 0.70 \pm 0.02 & 0.57 \pm 0.01 & \textbf{0.80} \pm \textbf{0.03} & \textbf{0.99} \pm \textbf{0.001} & 0.67 \pm 0.02\\
\cline{2-7}
& \textbf{infoNCE} & 0.71\pm 0.02 & 0.58 \pm 0.02 & 0.76 \pm 0.03 & \textbf{0.99} \pm \textbf{0.001} & 0.69 \pm 0.02\\
% \cline{2-7}
\hline
\hline

\multirow{\textbf{BraTs 2017}} & \textbf{Our Loss} & \textbf{0.82} \pm \textbf{0.03} & \textbf{0.71} \pm \textbf{0.02} & 0.75 \pm 0.04 & 0.97 \pm 0.001 & \textbf{0.93} \pm \textbf{0.01}\\
\cline{2-7}
& \textbf{Vanilla Dice} & 0.76 \pm 0.02 & 0.63 \pm 0.02 & 0.71 \pm 0.04 & 0.97 \pm 0.001 & 0.85 \pm 0.02\\
\cline{2-7}
& \textbf{simCLR} & 0.77 \pm 0.01 & 0.62 \pm 0.03 & \textbf{0.80} \pm \textbf{0.04} & \textbf{0.98} \pm \textbf{0.002} & 0.77 \pm 0.01 \\
\cline{2-7}
& \textbf{infoNCE}& 0.77 \pm 0.02 & 0.64 \pm 0.02 & \textbf{0.80} \pm \textbf{0.03} & \textbf{0.98} \pm \textbf{0.002} & 0.78 \pm 0.04\\
% \cline{2-7}
\hline
\hline

\multirow{\textbf{Combined BraTs}} & \textbf{Our Loss} & \textbf{0.83} \pm \textbf{0.03} & \textbf{0.69 }\pm \textbf{0.03} & 0.86 \pm 0.04 & \textbf{0.99} \pm \textbf{0.001} & \textbf{0.77} \pm \textbf{0.02} \\
\cline{2-7}
& \textbf{Vanilla Dice} & 0.73 \pm 0.02 & 0.59 \pm 0.02 & 0.76 \pm 0.06 & \textbf{0.99} \pm \textbf{0.002} & 0.74 \pm 0.03 \\
\cline{2-7}
& \textbf{simCLR}& 0.71 \pm 0.01 & 0.58 \pm 0.02 & \textbf{0.94} \pm \textbf{0.03} & \textbf{0.99} \pm \textbf{0.002} & 0.62 \pm 0.03\\
\cline{2-7}
& \textbf{infoNCE} & 0.81 \pm 0.01 & 0.68 \pm 0.01 & 0.92 \pm 0.03 & \textbf{0.99} \pm \textbf{0.001} & 0.74 \pm 0.02 \\
\hline
\hline

\multirow{\textbf{AbdomenCT-1K}} & \textbf{Our Loss} & \textbf{0.63} \pm \textbf{0.01}  & \textbf{0.46} \pm \textbf{0.01} & \textbf{0.60}  \pm \textbf{0.02}  & \textbf{0.96 } \pm \textbf{0.003}  & 0.68 \pm 0.04\\
\cline{2-7}
& \textbf{Vanilla Dice}& 0.60  \pm 0.04 & 0.43  \pm 0.04 & 0.55  \pm 0.05 & 0.95  \pm 0.004 & 0.67  \pm 0.03\\
\cline{2-7}
& \textbf{simCLR} & 0.61  \pm 0.01 & 0.44  \pm 0.01 & \textbf{0.60}  \pm \textbf{0.04}  & \textbf{0.96}  \pm \textbf{0.004}  & 0.63  \pm 0.04\\
\cline{2-7}
& \textbf{infoNCE} & 0.61  \pm 0.01  & 0.44  \pm 0.01 & 0.55  \pm 0.03 & 0.95  \pm 0.009  & \textbf{0.69}  \pm \textbf{0.04} \\
\hline
    
\end{tabularx}
\label{tab:comp}
\caption{\small Quantitative Test Results for Segmentation Performance with FESS Loss and Different Functions on Medical Image Datasets}
\end{table}


```


## Contributors
Charulkumar Chodvadiya, Kinshuk Gaurav Singh, Navyansh Mahla, Dr Kshitij Jadhav

## License
This project is licensed under the [MIT license](LICENSE).



