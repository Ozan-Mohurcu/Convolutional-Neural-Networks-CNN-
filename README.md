# 🧠 Convolutional Neural Networks (CNN) Projects

<div align="center">
  <img src="https://miro.medium.com/v2/resize:fit:1400/1*vkQ0hXDaQv57sALXAJquxA.jpeg" width="600" alt="CNN Banner"/>
  
  <h3>🔍 Deep Learning for Computer Vision & Image Classification</h3>
  
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow"/>
  <img src="https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white" alt="Keras"/>
  <img src="https://img.shields.io/badge/OpenCV-27338e?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV"/>
</div>

---

## 📋 Projects Overview

| Project | Type | Accuracy | Dataset | Domain |
|---------|------|----------|---------|--------|
| 🐱🐶 **Cat & Dog Classifier** | Binary Classification | 94.5% | 25,000 images | Computer Vision |
| 🥔 **Potato Health Detection** | Multi-class Classification | 92.3% | Plant Disease | Agriculture AI |

---

## 🚀 Featured Projects

### 🐱🐶 Cat & Dog Classification
**Advanced binary image classification using CNN architecture**

- **🎯 Objective**: Distinguish between cats and dogs in images
- **📊 Dataset**: 25,000 high-resolution pet images
- **🏗️ Architecture**: Custom CNN with data augmentation
- **📈 Performance**: 94.5% validation accuracy
- **🔧 Features**: Real-time prediction, batch processing

### 🥔 Potato Health Detection
**Agricultural AI for crop disease identification**

- **🎯 Objective**: Detect healthy vs diseased potato plants
- **📊 Dataset**: Agricultural plant disease dataset
- **🏗️ Architecture**: Transfer Learning + Fine-tuning
- **📈 Performance**: 92.3% disease detection accuracy
- **🌱 Impact**: Early disease detection for farmers

---

## 🛠️ Tech Stack

### 🤖 Deep Learning Frameworks
<table>
<tr>
<td align="center" width="25%">
<img src="https://www.tensorflow.org/images/tf_logo_social.png" width="50"/><br>
<b>TensorFlow</b>
</td>
<td align="center" width="25%">
<img src="https://upload.wikimedia.org/wikipedia/commons/a/ae/Keras_logo.svg" width="50"/><br>
<b>Keras</b>
</td>
<td align="center" width="25%">
<img src="https://opencv.org/wp-content/uploads/2020/07/OpenCV_logo_black-2.png" width="50"/><br>
<b>OpenCV</b>
</td>
<td align="center" width="25%">
<img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/numpy/numpy-original.svg" width="40"/><br>
<b>NumPy</b>
</td>
</tr>
</table>

### 📊 Data Processing & Visualization
<table>
<tr>
<td align="center" width="33%">
<img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/pandas/pandas-original.svg" width="40"/><br>
<b>Pandas</b>
</td>
<td align="center" width="33%">
<img src="https://matplotlib.org/_static/images/logo2.svg" width="50"/><br>
<b>Matplotlib</b>
</td>
<td align="center" width="33%">
<img src="https://seaborn.pydata.org/_static/logo-wide-lightbg.svg" width="60"/><br>
<b>Seaborn</b>
</td>
</tr>
</table>

### 💻 Development Environment
<table>
<tr>
<td align="center" width="33%">
<img src="https://upload.wikimedia.org/wikipedia/commons/3/38/Jupyter_logo.svg" width="40"/><br>
<b>Jupyter Notebook</b>
</td>
<td align="center" width="33%">
<img src="https://colab.research.google.com/img/colab_favicon_256px.png" width="40"/><br>
<b>Google Colab</b>
</td>
<td align="center" width="33%">
<img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/vscode/vscode-original.svg" width="40"/><br>
<b>VS Code</b>
</td>
</tr>
</table>

---

## 🚀 Quick Start

```bash
# Clone repository
git clone https://github.com/Ozan-Mohurcu/Convolutional-Neural-Networks-CNN-Public.git
cd Convolutional-Neural-Networks-CNN-Public

# Install dependencies
pip install tensorflow keras opencv-python matplotlib pandas numpy

# For GPU support (optional)
pip install tensorflow-gpu

# Launch Jupyter Notebook
jupyter notebook
```

### 📦 Requirements
```python
tensorflow>=2.8.0
keras>=2.8.0
opencv-python>=4.5.0
matplotlib>=3.5.0
pandas>=1.3.0
numpy>=1.21.0
pillow>=8.3.0
```

---

## 📈 Model Performance

<div align="center">

### 🏆 Accuracy Metrics

| Model | Training Acc | Validation Acc | Test Acc | Epochs |
|-------|-------------|----------------|----------|--------|
| **Cat & Dog CNN** | 96.2% | 94.5% | 93.8% | 25 |
| **Potato Disease** | 94.1% | 92.3% | 91.7% | 30 |

### 🔍 Model Features
- **Data Augmentation**: Rotation, zoom, flip for robust training
- **Transfer Learning**: Pre-trained models for faster convergence  
- **Custom Architecture**: Optimized CNN layers for each task
- **Real-time Inference**: Fast prediction on new images

</div>

---

## 📁 Project Structure

```
Convolutional-Neural-Networks-CNN-Public/
├── 🐱🐶 Cat&Dog/
│   ├── 📓 cat_dog_classifier.ipynb
│   ├── 📊 data/
│   │   ├── train/
│   │   └── validation/
│   ├── 🤖 models/
│   └── 📈 results/
├── 🥔 Potato Health/
│   ├── 📓 potato_disease_detection.ipynb
│   ├── 📊 plant_data/
│   ├── 🤖 trained_models/
│   └── 📈 visualizations/
└── 📖 README.md
```

---

## 🎯 Key Features

- **🔍 Computer Vision**: State-of-the-art image classification
- **🚀 High Accuracy**: 90%+ performance on both projects
- **⚡ Real-time**: Fast inference for production use
- **🔄 Transfer Learning**: Efficient model training
- **📊 Comprehensive Analysis**: Detailed performance metrics
- **🌱 Real-world Impact**: Agricultural and pet recognition applications

---

## 🤝 Usage Examples

### Cat & Dog Prediction
```python
from tensorflow import keras
import cv2

# Load trained model
model = keras.models.load_model('cat_dog_model.h5')

# Predict new image
img = cv2.imread('pet_image.jpg')
prediction = model.predict(img)
result = "Dog" if prediction > 0.5 else "Cat"
```

### Potato Health Detection
```python
# Load disease detection model
disease_model = keras.models.load_model('potato_health_model.h5')

# Analyze plant health
plant_img = cv2.imread('potato_leaf.jpg')
health_status = disease_model.predict(plant_img)
```

---

## 📞 Connect

<div align="center">

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/ozanmhrc/)
[![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/ozanmhrc)

**⭐ Star this repo if you found it helpful!**

</div>
