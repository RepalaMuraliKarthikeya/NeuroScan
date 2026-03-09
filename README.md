# AI-Based Pneumonia Detection System

A full-stack, production-level AI web application that detects Pneumonia from Chest X-ray images. The system utilizes Deep Learning (ResNet50 Transfer Learning) and Explainable AI (Grad-CAM) integrated into a Flask backend with a beautiful, responsive dark-themed UI.

## Features
- **User Authentication**: Secure registration and login.
- **Dashboard**: Track past predictions and performance metrics.
- **X-Ray Inference**: Real-time upload and prediction.
- **Explainable AI**: Grad-CAM heatmaps generated to visually explain AI decisions.
- **Model Training Suite**: Scripts included for building custom CNNs and fine-tuning ResNet50.

## Tech Stack
- **Backend & Web**: Python, Flask, Flask-SQLAlchemy, Flask-Login, SQLite.
- **Deep Learning**: TensorFlow, Keras, OpenCV.
- **Frontend**: HTML5, Vanilla Premium Glassmorphism CSS, Bootstrap 5.

## Installation & Deployment Guide

### 1. Setup Virtual Environment & Dependencies
```bash
# Navigate to project directory
cd pneumonia_detection

# Create virtual env
python -m venv venv

# Activate (Windows)
.\venv\Scripts\activate

# Activate (Mac/Linux)
# source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 2. Prepare the AI Model
You have two options to prepare the underlying `.h5` model:

**Option A (Fast - For Web App Testing)**
Generates a dummy untrained architecture in seconds strictly to test the UI flow and Grad-CAM functionality.
```bash
python scripts/generate_dummy_model.py
```

**Option B (Full - Real Training)**
1. Run `python scripts/prepare_data.py` to see dataset download instructions.
2. Place your Kaggle images inside `dataset_split/`.
3. Train the model: `python scripts/train_resnet.py` (May take hours depending on GPU).

### 3. Run the Web Application
Start the Flask server:
```bash
python app.py
```
By default, the application will initialize its SQLite database automatically on the first run.
Open `http://localhost:5000` in your browser.

## Directory Structure
- `app/`: Flask application code (routes, templates, DB models).
- `models/`: Deep learning `.h5` model storage, Inference logic, and Grad-CAM utils.
- `scripts/`: Data fetching and model training scripts.
- `dataset_split/`: Target folder for the raw image data.
