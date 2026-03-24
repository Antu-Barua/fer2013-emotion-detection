# 😊 Emotion Detection using Facial Images (FER-2013)

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat-square&logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red?style=flat-square&logo=keras)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-blue?style=flat-square&logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen?style=flat-square)

> **University Research Project** — Port City International University, October 2025

A comprehensive comparison of **deep learning** and **classical ML** models for facial emotion recognition using the FER-2013 dataset. This project evaluates 7+ models across multiple train-test splits and data augmentation strategies to classify 7 human emotions.

---

## 🎯 Project Overview

| Feature | Details |
|---|---|
| 📁 Dataset | FER-2013 (Kaggle) |
| 🖼️ Total Images | 35,887 grayscale images |
| 😀 Emotion Classes | 7 (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral) |
| 📐 Image Size | 48 × 48 pixels |
| 🧠 Models Compared | VGG16, ResNet101, DenseNet121, EfficientNetB3, MobileNetV2/V3, Random Forest, Naive Bayes |
| 🎯 Best Accuracy | ~63% |
| ⚡ Compute Optimization | 20% reduction via efficient batch processing |
| 🏫 University | Port City International University, Oct 2025 |

---

## 🏗️ Project Structure

```
fer2013-emotion-detection/
│
├── 📂 preprocessing/
│   └── emotion preprocessing.ipynb         # Data loading, cleaning & augmentation
│
├── 📂 deep_learning/
│   ├── 📂 vgg16/
│   │   ├── DA Vgg16 split 70-30 with pt.ipynb
│   │   ├── vgg16 split 80-20 without pt.ipynb
│   │   ├── with Da VGG-16 + CNN split 70-30 with pt.ipynb
│   │   ├── with Da VGG-16 + CNN split 80-20 without pt.ipynb
│   │   └── with da vgg16+cnn split80-20 with pt.ipynb
│   │
│   ├── 📂 resnet/
│   │   ├── ResNet101_MobileNetV3_70_30_WDA.ipynb
│   │   └── ResNet101_MobileNetV3_70_30_PT_WDA.ipynb
│   │
│   ├── 📂 densenet/
│   │   ├── DenseNet121 split 70-30 without pt.ipynb
│   │   ├── DenseNet121 split80-20 with pt.ipynb
│   │   └── with da DenseNet121_MobileNetV2 split 70-30 without pt.ipynb
│   │
│   └── 📂 efficientnet/
│       ├── with DA EfficientNetB3 split 70-30 with PT.ipynb
│       ├── without DA EfficientNetB3 split 80-20 with PT.ipynb
│       ├── with Da EfficientNet + Naive Bayes split 70-30 with pt.ipynb
│       └── without Da EfficientNet + Naive Bayes split 80-20 with pt.ipynb
│
├── 📂 classical_ml/
│   ├── 📂 random_forest/
│   │   ├── random forest split 70-30 with pt.ipynb
│   │   ├── random forest split 70-30 without pt.ipynb
│   │   └── Da random forest split70-30 with pt.ipynb
│   │
│   └── 📂 naive_bayes/
│       ├── Naive Bayes split 80-20 with pt.ipynb
│       └── Da Naive Bayes split 70-30 with pt.ipynb
│
└── README.md
```

---

## 🧪 Models & Experiments

### Deep Learning Models

| Model | Split | Data Aug | Notes |
|---|---|---|---|
| VGG16 + CNN | 70-30 | ✅ Yes | Transfer learning |
| VGG16 + CNN | 80-20 | ❌ No | Baseline |
| VGG16 + CNN | 80-20 | ✅ Yes | With pretrained weights |
| ResNet101 + MobileNetV3 | 70-30 | ✅ Yes | Ensemble approach |
| ResNet101 + MobileNetV3 | 70-30 | ❌ No | Without augmentation |
| DenseNet121 | 70-30 | ❌ No | Dense connections |
| DenseNet121 | 80-20 | ✅ Yes | With pretrained weights |
| DenseNet121 + MobileNetV2 | 70-30 | ✅ Yes | Combined model |
| EfficientNetB3 | 70-30 | ✅ Yes | Efficient scaling |
| EfficientNetB3 | 80-20 | ❌ No | Without augmentation |

### Classical ML Models

| Model | Split | Data Aug | Notes |
|---|---|---|---|
| Random Forest | 70-30 | ✅ Yes | Flattened pixel features |
| Random Forest | 70-30 | ❌ No | Baseline |
| Naive Bayes | 70-30 | ✅ Yes | Gaussian NB |
| Naive Bayes | 80-20 | ✅ Yes | With pretrained features |
| EfficientNet + Naive Bayes | 70-30 | ✅ Yes | Hybrid approach |
| EfficientNet + Naive Bayes | 80-20 | ❌ No | Without augmentation |

> 📌 **Note:** FER-2013 is a notoriously challenging dataset. State-of-the-art models typically achieve 65–75%. Our best accuracy (~63%) is consistent with published benchmarks for similar architectures.

---

## ⚙️ Data Preprocessing Pipeline

```python
# Steps in emotion preprocessing.ipynb
1. Load FER-2013 CSV and parse pixel strings
2. Reshape images to (48, 48, 1) grayscale format
3. Normalize pixel values to [0, 1]
4. One-hot encode 7 emotion labels
5. Train/test split: 70-30 and 80-20 experiments

# Data Augmentation (where applied)
- Horizontal flip
- Rotation range: ±10°
- Zoom range: 10%
- Width/height shift: 10%
```

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install tensorflow keras scikit-learn numpy pandas matplotlib seaborn opencv-python
```

### Run
```bash
# Clone the repo
git clone https://github.com/antubarua/fer2013-emotion-detection.git
cd fer2013-emotion-detection

# Start with preprocessing
jupyter notebook preprocessing/emotion\ preprocessing.ipynb

# Then run any model notebook
jupyter notebook deep_learning/vgg16/with\ Da\ VGG-16\ +\ CNN\ split\ 70-30\ with\ pt.ipynb
```

### Google Colab
All notebooks are compatible with Google Colab. Upload the notebook and mount your Drive with the FER-2013 dataset.

---

## 🔑 Key Highlights

- ✅ **7+ models** compared — deep learning vs. classical ML
- ✅ **Multiple experiments** — 70-30 and 80-20 splits tested
- ✅ **Data augmentation impact** — with and without DA compared
- ✅ **Hybrid approaches** — EfficientNet + Naive Bayes, ResNet + MobileNet
- ✅ **20% reduction** in computation time via efficient batch processing
- ✅ **University research project** at Port City International University

---

## 🏫 University Project

> **Emotion Detection Using FER-2013 Dataset**
> Antu Barua — Port City International University, Bangladesh
> Completed: October 2025

---

## 👨‍💻 Author

**Antu Barua**
CSE Graduate | Data Science & ML Engineer

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat-square&logo=linkedin)](https://linkedin.com/in/antubarua)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?style=flat-square&logo=github)](https://github.com/antubarua)
[![Portfolio](https://img.shields.io/badge/Portfolio-Visit-orange?style=flat-square)](https://antubarua.dev)

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center">⭐ If you found this project helpful, please give it a star!</p>
