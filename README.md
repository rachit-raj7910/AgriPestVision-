# Agricultural Pest Prediction Using Deep Learning

## Project Overview

This project aims to build a robust deep learning model for accurately predicting pest species in agricultural environments based on image data. By leveraging convolutional neural networks and state-of-the-art techniques like transfer learning and data augmentation, the model helps farmers and agronomists take preventive action against pest infestations, improving crop yield and ensuring food security.

## Key Features

- ✅ Image-based pest classification using EfficientNetB4 (pretrained on ImageNet)  
- ✅ Data preprocessing with augmentation and CutMix strategy  
- ✅ Performance evaluation using accuracy, top-k accuracy, and confusion matrix  
- ✅ Easy-to-use Gradio-based User Interface (UI)  
- ✅ Fully reproducible code using local datasets  

## Libraries and Tools Used

- `NumPy` and `Pandas` – Data manipulation and analysis  
- `TensorFlow` and `Keras` – Deep learning model creation and training  
- `OpenCV (cv2)` – Image preprocessing  
- `Matplotlib` and `Seaborn` – Visualization  
- `Scikit-learn` – Evaluation metrics  
- `Gradio` – Interactive UI for real-time predictions  

## Dataset

The image data is sourced from a Kaggle dataset:  
[🔗 Agricultural Pests Image Dataset](https://www.kaggle.com/datasets/vencerlanz09/agricultural-pests-image-dataset)

After unzipping, the dataset is organized into folders based on pest categories like:
- Aphids
- Army worm
- Cutworm
- Earworm
- Fruit borer
- Green leafhopper
- Hairy caterpillar
- Leaf folder
- Stem borer
- Whitefly

Each folder contains `.jpg` images representing instances of each pest.

## ZIP Extraction Step

Before running the notebook, extract the dataset ZIP file to the following structure:

```
/mnt/data/extracted_data/Dataset/
├── Aphids.csv
├── Army worm.csv
├── ...
├── Whitefly.csv
```

You can extract the ZIP in Python using:

```python
import zipfile

zip_path = "/mnt/data/archive.zip"  # your ZIP file
extract_path = "/mnt/data/extracted_data"

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)
```

Ensure your notebook points to the extracted CSV/image paths.

## Model Architecture

- Base Model: `EfficientNetB4` (with ImageNet weights)  
- Additional Layers:
  - Global Average Pooling
  - Dense layers with Dropout
  - Softmax output for multiclass classification

The model is fine-tuned using:
- **Categorical crossentropy loss**
- **Adam optimizer**
- **Accuracy and Top-K Accuracy as evaluation metrics**

## Training and Evaluation

- ✅ Data augmentation using `ImageDataGenerator`  
- ✅ CutMix augmentation technique to reduce overfitting  
- ✅ Evaluation using confusion matrix and accuracy scores  

Sample performance metrics:
- Training Accuracy: ~99%  
- Validation Accuracy: ~98%  

## User Interface (Gradio)

An interactive Gradio UI allows for real-time predictions:
- Upload a pest image
- View the predicted pest class and confidence score
- Ideal for field-level diagnostic use

## Installation Requirements

Install the required libraries before running the notebook:

```bash
pip install numpy pandas tensorflow matplotlib opencv-python seaborn scikit-learn gradio
```

## Usage Instructions

1. **Unzip the dataset** into `/mnt/data/extracted_data/`.
2. **Open and run** `agriculture-pest-prediction-updated.ipynb` in Jupyter.
3. Follow cells sequentially for preprocessing, model training, evaluation, and UI
4. Use the **Gradio UI** for pest image prediction.
