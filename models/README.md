# Models Directory

This directory contains the trained model files for the Live Video Anomaly Detection project.

## Model Files

Due to GitHub's file size limitations (100MB), the actual model files are not stored in this repository. The following models are part of this project:

- `alexnet_model.h5` - AlexNet model for anomaly detection (343.70 MB)
- `densenet_model.h5` - DenseNet121 transfer learning model
- `mobilenet_anomaly_model_final.h5` - MobileNet V2 lightweight model

## How to Get the Models

### Option 1: Train Yourself (Recommended)
Run the `model_training.ipynb` notebook to train all models from scratch. This will:
- Download the UCF-Crime dataset
- Train DenseNet121 and AlexNet models
- Save the trained models in this directory

### Option 2: Download Pre-trained Models
If pre-trained models are available in the GitHub releases section, you can download them directly.

### Option 3: Use Alternative Storage
For sharing large model files, consider:
- Google Drive
- Dropbox
- Git LFS (Large File Storage)
- Hugging Face Model Hub

## Model Performance

The models are trained on the UCF-Crime dataset with 14 activity classes including various anomalous activities like assault, robbery, vandalism, etc.

Refer to the main README.md for detailed information about model architectures and performance metrics.
