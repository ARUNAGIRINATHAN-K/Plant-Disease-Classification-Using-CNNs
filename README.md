---
title: Plant Disease Classification
emoji: 🌿
colorFrom: green
colorTo: green
sdk: gradio
sdk_version: "4.50.0"
python_version: "3.10"
app_file: app.py
pinned: false
---

# 🌿 Plant Disease Classification Using CNNs

*Trained a CNN model to automatically identify diseases in plant leaves from images, which helps farmers detect crop issues early and reduce losses.*

---

## Dataset ([view on Kaggle](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset))
```python
import kagglehub
path = kagglehub.dataset_download("abdallahalidev/plantvillage-dataset")
```

- ~54,000 images across **38 classes** (healthy + diseased)
- Covers **14 crop species** (tomato, potato, corn, apple, etc.)
- Available on Kaggle and TensorFlow Datasets

## Model
- **Architecture:** EfficientNetB3 (Transfer Learning + Fine-tuning)
- **Framework:** TensorFlow / Keras
- **Input:** 224 × 224 RGB leaf images
- **Output:** 38-class softmax prediction

## Features
- 🔍 Upload any leaf image for instant disease detection
- 📊 Top-3 predictions with confidence scores
- 💊 Treatment & prevention recommendations
- 🔥 Grad-CAM visualization of model attention