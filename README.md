# Wine Type Prediction

A binary classification project that predicts whether a wine is red or white based on its physicochemical properties, using a deep neural network built with Keras.

---

## Overview

This project uses the UCI Wine Quality Dataset, combining red and white wine samples into a single binary classification task. The model achieves 99.77% test accuracy and a ROC-AUC score of 0.9997, correctly classifying 1297 out of 1300 test samples.

---

## Dataset

Source: UCI Machine Learning Repository — Wine Quality Dataset

- Red wine samples: 1,599
- White wine samples: 4,898
- Total samples: 6,497
- Features: 12 physicochemical properties

Features used:

- Fixed acidity
- Volatile acidity
- Citric acid
- Residual sugar
- Chlorides
- Free sulfur dioxide
- Total sulfur dioxide
- Density
- pH
- Sulphates
- Alcohol
- Quality

Target: Wine type — Red (1) or White (0)

---

## Model Architecture

A sequential feedforward neural network with the following structure:

| Layer | Units | Activation |
|---|---|---|
| Dense | 64 | ReLU |
| BatchNormalization + Dropout (0.3) | - | - |
| Dense | 32 | ReLU |
| BatchNormalization + Dropout (0.3) | - | - |
| Dense | 16 | ReLU |
| BatchNormalization + Dropout (0.2) | - | - |
| Dense | 8 | ReLU |
| BatchNormalization | - | - |
| Dense (Output) | 1 | Sigmoid |

- Optimizer: Adam
- Loss: Binary crossentropy
- Epochs: Up to 100 with EarlyStopping (patience=5)
- Batch size: 32

---

## Key Techniques

- StandardScaler for feature normalisation — fitted on training set only
- Stratified train/test split to preserve class ratio
- Class weights to handle the 3:1 white-to-red imbalance
- BatchNormalization for training stability
- Dropout for regularisation
- EarlyStopping with restore_best_weights to prevent overfitting

---

## Results

| Metric | Score |
|---|---|
| Test Accuracy | 99.77% |
| Test Loss | 0.0136 |
| ROC-AUC | 0.9997 |
| Misclassifications | 3 out of 1300 |

Classification Report:

| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| White | 1.00 | 1.00 | 1.00 |
| Red | 1.00 | 0.99 | 1.00 |

---

## Project Structure

```
wine-type-prediction/
│
├── Prediction_of_Wine_Type.ipynb   # Main notebook
├── redwinequality.csv              # Red wine dataset
├── whitewinequality.csv            # White wine dataset
└── README.md
```

---

## Requirements

```
numpy
pandas
matplotlib
seaborn
scikit-learn
tensorflow
keras
```

Install all dependencies:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow
```

---

## How to Run

1. Clone the repository:

```bash
git clone https://github.com/your-username/wine-type-prediction.git
cd wine-type-prediction
```

2. Install dependencies:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow
```

3. Place `redwinequality.csv` and `whitewinequality.csv` in the project directory.

4. Open and run the notebook:

```bash
jupyter notebook Prediction_of_Wine_Type.ipynb
```

Or open directly in Google Colab.

---
