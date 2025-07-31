# Live Video Anomaly Detection 🎥🔍

A comprehensive machine learning project for real-time and pre-recorded video anomaly detection using deep learning models and computer vision techniques.

## 📋 Overview

This project implements multiple approaches for video anomaly detection, including real-time detection using webcam input and analysis of pre-recorded videos. The system can detect various types of anomalous activities including violence, theft, accidents, and other suspicious behaviors.

## 🎯 Features

- **Real-time Anomaly Detection**: Live video analysis using webcam input
- **Pre-recorded Video Analysis**: Batch processing of video files
- **Multiple Model Support**: AlexNet, DenseNet121, and MobileNet implementations
- **YOLO Integration**: Object detection combined with anomaly classification
- **Multi-class Classification**: Detection of 14 different activity types
- **Data Visualization**: Training metrics and ROC curves
- **Transfer Learning**: Utilizing pre-trained models for improved performance

## 🛠️ Technologies Used

### Deep Learning Frameworks
- **TensorFlow/Keras**: Primary framework for model development
- **PyTorch**: Supporting framework for YOLO integration

### Computer Vision
- **OpenCV**: Video processing and real-time capture
- **YOLO v8**: Object detection and tracking
- **PIL/Pillow**: Image preprocessing

### Machine Learning Libraries
- **NumPy**: Numerical computations
- **Pandas**: Data manipulation and analysis
- **Scikit-learn**: Model evaluation metrics
- **Matplotlib/Seaborn**: Data visualization
- **Plotly**: Interactive visualizations

### Pre-trained Models
- **DenseNet121**: Transfer learning for feature extraction
- **AlexNet**: Classic CNN architecture
- **MobileNet V2**: Lightweight model for mobile deployment
- **YOLOv8n**: Real-time object detection

### Data Processing
- **ImageDataGenerator**: Data augmentation and preprocessing
- **KaggleHub**: Dataset downloading and management

## 📁 Project Structure

```
LiveVideoAnomalyDetection/
├── model_training.ipynb              # Complete model training pipeline
├── model_run_with_pre-recorded_video.ipynb  # Pre-recorded video analysis
├── model_run_with_live_video_yolo.py        # Real-time detection with YOLO
├── yolov8n.pt                        # YOLO model weights
├── models/                           # Trained model files
│   ├── alexnet_model.h5
│   ├── densenet_model.h5
│   └── mobilenet_anomaly_model_final.h5
├── videos/                           # Sample video files
│   ├── video1.mp4
│   ├── video2.mp4
│   └── video3.mp4
├── ucf-crime-dataset1/               # Dataset directory
│   ├── Train/                        # Training data
│   └── Test/                         # Testing data
└── README.md
```

## 🎭 Supported Activity Classes

The system can detect and classify the following 14 activity types:

1. **Abuse** - Physical abuse scenarios
2. **Arrest** - Law enforcement activities
3. **Arson** - Fire-related incidents
4. **Assault** - Physical assault cases
5. **Burglary** - Breaking and entering
6. **Explosion** - Explosive incidents
7. **Fighting** - Physical altercations
8. **Normal Videos** - Regular, non-anomalous activities
9. **Road Accidents** - Traffic-related incidents
10. **Robbery** - Theft with force or threat
11. **Shooting** - Firearm-related incidents
12. **Shoplifting** - Retail theft
13. **Stealing** - General theft activities
14. **Vandalism** - Property damage

## 🚀 Getting Started

### Prerequisites

```bash
pip install tensorflow
pip install opencv-python
pip install ultralytics
pip install numpy
pip install pandas
pip install matplotlib
pip install seaborn
pip install plotly
pip install scikit-learn
pip install kagglehub
pip install torch
pip install pillow
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/geeky-brahma/LiveVideoAnomalyDetection.git
cd LiveVideoAnomalyDetection
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Download the dataset (handled automatically in the training notebook)

### Usage

#### Training Models
1. Open `model_training.ipynb` in Jupyter Notebook or VS Code
2. Run all cells to:
   - Download and preprocess the UCF-Crime dataset
   - Train DenseNet121 and AlexNet models
   - Generate performance visualizations
   - Save trained models

#### Real-time Detection
```bash
python model_run_with_live_video_yolo.py
```

#### Pre-recorded Video Analysis
1. Open `model_run_with_pre-recorded_video.ipynb`
2. Run cells to analyze videos in the `videos/` directory

## 📊 Model Performance

The project implements multiple architectures with the following characteristics:

- **DenseNet121**: Transfer learning with ImageNet weights
- **AlexNet**: Custom implementation adapted for 64x64 input
- **MobileNet V2**: Lightweight model for real-time applications

Performance metrics include:
- Multi-class ROC-AUC curves
- Accuracy measurements
- Training/validation loss tracking

## 🗃️ Dataset

The project uses the **UCF-Crime Dataset**, which contains:
- Surveillance camera footage
- 14 different crime and normal activity categories
- Training and testing splits
- Preprocessed frames for model training

## 🔧 Configuration

Key hyperparameters (configurable in `model_training.ipynb`):
- Image dimensions: 64x64 pixels
- Batch size: 64
- Learning rate: 0.00003
- Number of classes: 14
- Training epochs: Configurable

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- UCF Crime Dataset creators
- TensorFlow and PyTorch communities
- YOLO development team
- OpenCV contributors

## 📞 Contact

For questions or collaboration opportunities, please reach out through GitHub issues.

---

**Note**: This project is for educational and research purposes. Ensure compliance with local laws and regulations when deploying video surveillance systems.
