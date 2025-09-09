# Deep Regression for Predicting Annotation Variability in Skin Lesion Segmentation - VARISkin

This project investigates deep learning regression models for predicting annotation variability in skin lesion segmentation, a challenge caused by inter-annotator disagreement when outlining lesion boundaries. Using a dataset of 2,261 dermoscopic images and transfer learning from pretrained architectures (ResNet-18, EfficientNet-B0, and ViT-B/32), the models were trained to estimate disagreement scores (based on Dice coefficients). Results showed that while ResNet-18 achieved slightly lower error rates, differences across models were not statistically significant. The study also explored interpretability with Grad-CAM and performed statistical tests and metadata correlation analysis, ultimately demonstrating the feasibility of real-time variability prediction as a foundation for quality control in medical image annotation pipelines.

## Report
[**Read about our findings here!**](https://github.com/Hooyar-Foroughi/VariSkin/blob/main/Report.pdf)
## Demo Video
[**Watch our demo video here!**](https://drive.google.com/file/d/1Rb_n_naBlO2E6h_Nkhtk3LNm3vEgj-x8/view?usp=sharing)

## Table of Contents:
1. [Demo](#demo)

2. [Installation](#installation)

3. [Reproducing this project](#repro)

<br>

## 1. Example demo

A minimal example to showcase the main training and evaluation flow of this project:

Within ***src/*** you wil find ***run.py***, which is the main entry point. Make sure your [**config.yaml**](#config) is properly set, and you have carefully followed the [**Installation**](#installation) steps prior to running the pipeline.
```bash
python3 run.py
```

If you want to enable or disable specific features (such as Grad-CAM, scatter plots, or subset evaluations), [update the **config.yaml** file accordingly.](#config)

### What to find where

A brief overview of the repository structure:

```bash
repository
├── src/                         ## All source code lives here
│   ├── run.py                   ## Main script for training, evaluation, visualization
│   ├── config.yaml              ## Central config file to toggle features and paths
│   ├── utils/                   ## Helper modules: Grad-CAM, data loaders, etc.
│   └── analysis/                ## Evaluation, plots, generated visuals, subset stats
│
├── requirements.txt             ## Pip dependencies
├── .gitignore                   ## Git exclusions (e.g., __pycache__, results)
├── README.md                    ## You are here
└── LICENSE                      ## Licensing information
```

<br>

<a name="installation"></a>

## 2. Installation

>**Disclaimer:** *This program was built and tested specifically on Ubuntu 22.04.5 LTS, running on Simon Fraser University (SFU) CSIL lab machines. Compatibility or performance on other operating systems or environments is not guaranteed.*

### Basic Execution Instructions and Guidance
The goal of this guidance is to allow everyone to simply execute the code without needing to debug installation issues. It aims to address the "it works on my computer" problem.

### What is the requirements.txt?
This file contains the local libraries required to execute the program. It was created by doing a `pip freeze --local > requirements.txt` . If you install a new dependency, please update this file.

### Execution instructions for Linux with a virtual environment
You can read more about virtual environments [**here.**](https://docs.python.org/3/library/venv.html)

A virtual environment creates an isolated library path for your project, ensuring that it uses its own dependencies without interfering with globally installed Python packages. To activate and use a virtual environment, follow these steps:

>*Note: Some instructions differ when using SFU CSIL machines. Please follow the steps below accordingly.*

>*Note: Please ensure that you are working within the `VariSkin/` project directory before executing the instructions outlined below.*


**1. Create the virtual environment:**

- On local machine:
    `python3 -m venv venv`
- On CSIL:
    `python3 -m venv --copies venv`

**2. Activate the virtual environment:**

&nbsp;&nbsp;&nbsp; On local machine:
- `chmod u+x venv/bin/activate`
- `./venv/bin/activate` 

&nbsp;&nbsp;&nbsp; **OR** instead (works on **CSIL**):
- `source venv/bin/activate`

**3. Install dependencies from the requirements.txt:** 

- `pip install -r requirements.txt`

**4. Execute the program:** (in `src/` directory)

- All model types, toggles, and file paths are controlled via the config.yaml file.
- `python3 run.py`

<br>

<a name="config"></a>
### Breakdown of config.yaml

**Your `config.yaml` file (located inside the `src/` directory) controls what components of the project will run and how. Below is a summary of its structure:**

```yaml
# What to run
enable_training: true                    # Train the model from scratch

# Model & training
model_name: resnet18                     # Options: resnet18, efficientnet_b0, vit_b_32
num_epochs: 10                           # Number of training epochs
batch_size: 32                           # Batch size for dataloaders

# Analysis output options
enable_scatter_plot: true               # Visualize ground truth vs. predictions
enable_error_visuals: true              # Show images with highest/lowest prediction errors
enable_subset_eval: true                # Evaluate model on benign vs. malignant subsets
enable_correlation_analysis: true       # Correlate disagreement with diagnosis type
enable_mae_normalization: true          # Normalize and statistically compare MAE values
enable_mse_normalization: true          # Normalize and statistically compare MSE values
enable_gradcam: true                    # Generate Grad-CAM heatmaps for selected images
enable_dice_histograms: true            # Plot histograms of Dice score distribution

# Images to use for heatmaps
gradcam_images:                         # Filenames of ISIC images to use for Grad-CAM
  - ISIC_0000088.jpg
  - ISIC_0000047.jpg
  - ISIC_0000013.jpg

# File and directory paths              (will be downloaded during inital execution)
metrics_file: metrics.csv               # CSV file with image stats (should be in src/)
image_archive_dir: ISIC_Archive         # Directory with image files (should be in src/)
benign_json: analysis/benign_names.json # JSON list of benign images (should be in src/analysis/)
malignant_json: analysis/malignant_names.json # JSON list of malignant images (should be in src/analysis/)
```

This file gives you full control over which functionalities to enable, what data to use, and how to run the model.

<br>

<a name="repro"></a>

## 3. Reproduction

To reproduce the core results and visualizations of this project, follow the steps below.

### Prerequisites

Ensure that:
- The virtual environment is activated (see [Installation](#installation))
- You are working inside the `VariSkin/` project directory
- All dependencies are installed (`pip install -r requirements.txt`)

### Steps

```bash
# Step into the main project directory (if not already there)
cd VariSkin/

# Activate the virtual environment
source venv/bin/activate

# Run the main program
python3 src/run.py
```

### The execution will:

- Train the model (if enable_training: true is set in config.yaml)
- Generate prediction outputs and evaluation results
- Produce all selected visualizations (Grad-CAM, scatter plots, statistical plots, etc.)
- Print detailed progress and results to the terminal, such as:
    - Training progress and loss per epoch
    - Final test performance (MSE, MAE)
    - Messages for any enabled analysis modules (e.g. Grad-CAM, correlation plots, error visualizations)

### Data:

- All necessary data (images, metrics, and metadata) will be automatically downloaded through the script.

### Output:

Generated outputs will be saved inside appropriate subfolders under src/analysis/, such as:
- gradcam_results/ for Grad-CAM heatmaps
- experiment/ for statistical result plots (MAE, MSE)
- diagnosis_correlation/ – correlation plots and metadata analysis
- histograms/, scatter/, and others for their respective visualizations
