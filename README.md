# Water Quality Classification for Safe Drinking Prediction

A complete Machine Learning project that predicts whether water is **safe for drinking (Potable)** or **unsafe (Not Potable)** using multiple classification algorithms.

This project includes:

* Exploratory Data Analysis (EDA)
* Data preprocessing
* Feature engineering
* SMOTE for class balancing
* Model training & evaluation
* Feature importance analysis
* Performance comparison across models

---

# Project Overview

Clean drinking water is essential for health and safety.
This project uses physicochemical properties of water samples to classify water quality using supervised machine learning techniques.

The workflow covers the complete ML pipeline from raw dataset to final model evaluation.

---

# Dataset Information

**Dataset:** `water_potability.csv`
**Source:** [Kaggle Water Potability Dataset](https://www.kaggle.com/datasets/adityakadiwal/water-potability?utm_source=chatgpt.com)

| Attribute       | Details                      |
| --------------- | ---------------------------- |
| Total Samples   | 3,276                        |
| Features        | 9                            |
| Target Variable | `Potability`                 |
| Classes         | 0 = Not Potable, 1 = Potable |

---

# Features Used

| Feature           | Description                  |
| ----------------- | ---------------------------- |
| `ph`              | pH value of water            |
| `Hardness`        | Water hardness               |
| `Solids`          | Total dissolved solids       |
| `Chloramines`     | Chloramine concentration     |
| `Sulfate`         | Sulfate concentration        |
| `Conductivity`    | Electrical conductivity      |
| `Organic_carbon`  | Organic carbon content       |
| `Trihalomethanes` | Trihalomethane concentration |
| `Turbidity`       | Water turbidity              |

---

# Technologies & Libraries

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* XGBoost
* Imbalanced-learn (SMOTE)
* Plotly *(optional)*

---

# Project Structure

```bash
.
├── water_quality_classification.py
├── water_potability.csv
├── README.md
└── outputs/
    ├── 01_missing_values.png
    ├── 02_class_distribution.png
    ├── 03_correlation_heatmap.png
    ├── 04_boxplots.png
    ├── 05_feature_distributions.png
    ├── 06_violin_ph.png
    ├── 07_model_comparison.png
    ├── 08_roc_curves.png
    ├── cm_*.png
    └── fi_*.png
```

---

# Installation

Clone the repository and install dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn
```

Optional:

```bash
pip install plotly
```

---

# How to Run

### Step 1 — Download Dataset

Download `water_potability.csv` from Kaggle and place it in the project folder.

### Step 2 — Run the Script

```bash
python water_quality_classification.py
```

---

# Machine Learning Pipeline

## 1. Exploratory Data Analysis (EDA)

The project performs:

* Missing value analysis
* Class distribution analysis
* Correlation heatmaps
* Boxplots for outlier detection
* Feature distribution visualization
* Violin plots for pH analysis

---

## 2. Data Preprocessing

### Missing Value Handling

Mean imputation is applied to:

* `ph`
* `Sulfate`
* `Trihalomethanes`

### Feature Engineering

New engineered features include:

* `pH_Hardness_ratio`
* `TDS_Conductivity_ratio`
* `Chloramines_THM_ratio`
* `ph_sq`
* `Turbidity_sq`
* `Organic_carbon_sq`

### Class Imbalance Handling

SMOTE is used to balance the minority class.

### Feature Scaling

StandardScaler normalization is applied before model training.

---

# Models Used

The following machine learning models are trained and evaluated:

| Model                        | Description                  |
| ---------------------------- | ---------------------------- |
| Logistic Regression          | Linear baseline classifier   |
| Decision Tree                | Tree-based classifier        |
| Random Forest                | Ensemble learning method     |
| XGBoost                      | Gradient boosting classifier |
| K-Nearest Neighbors          | Distance-based classifier    |
| Support Vector Machine (SVM) | RBF kernel classifier        |
| AdaBoost                     | Adaptive boosting classifier |

---

# Evaluation Metrics

Each model is evaluated using:

* Accuracy Score
* ROC-AUC Score
* Classification Report
* Confusion Matrix
* ROC Curves

---

# Visual Outputs

The script automatically generates visualizations including:

* Missing value charts
* Correlation heatmaps
* Boxplots
* KDE plots
* ROC curves
* Model comparison graphs
* Feature importance charts

All output images are saved in the `outputs/` folder.

---

# Results & Insights

Key findings from the project:

* Random Forest and XGBoost achieved the best overall performance.
* SMOTE improved minority class prediction significantly.
* Important predictors include:

  * pH
  * Sulfate
  * Solids
* SVM provided strong baseline performance among non-ensemble methods.

---

# Future Improvements

Possible future enhancements:

* Hyperparameter tuning with GridSearchCV
* Deep learning implementation
* Streamlit web application deployment
* Real-time water quality prediction API
* Model explainability using SHAP

# DEPLOYMENT

# 💧 AquaGuard — Water Quality Classification using Machine Learning

AquaGuard is an AI-powered Streamlit web application that predicts whether water is safe for drinking using multiple Machine Learning classification models.

The application performs:
- Water potability prediction
- Exploratory Data Analysis (EDA)
- ROC curve analysis
- Feature importance visualization
- Model comparison dashboard

Built using:
- Streamlit
- Scikit-learn
- XGBoost
- Imbalanced-learn (SMOTE)
- Pandas
- NumPy
- Matplotlib
- Seaborn
#  Features

 Real-time water quality prediction  
 7 Machine Learning models  
 Beautiful dark-themed dashboard  
 ROC curve comparison  
 Feature importance analysis  
 Dataset exploration  
 Interactive sliders for water sample testing  
SMOTE balancing for imbalanced dataset handling  

---

#  Machine Learning Models Used

- Random Forest
- XGBoost
- Support Vector Machine (SVM)
- AdaBoost
- K-Nearest Neighbors (KNN)
- Decision Tree
- Logistic Regression


# Author

Developed as a Machine Learning classification project for water quality prediction.
