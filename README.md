# 💳 Financial Fraud Detection Project (DataThon)

## 📌 Overview
This project aims to build a robust machine learning model to detect financial fraud. By analyzing the **Financial Transactions Dataset**, we identify patterns and risk factors associated with fraudulent activities. This repository documents the complete data science workflow, from data cleaning and feature engineering to model training, hyperparameter tuning, and evaluation.

## 📊 Dataset
- **Source**: [Transactions Fraud Datasets (Kaggle)](https://www.kaggle.com/datasets/computingvictor/transactions-fraud-datasets/data)
- **Description**: The dataset contains demographic and financial information about users, including:
    - **Demographics**: `Current Age`, `Gender`, `Address` (Latitude/Longitude)
    - **Financial Status**: `Credit Score`, `Yearly Income`, `Total Debt`, `Per Capita Income`
    - **Target Variable**: `fraud_risk` (1: High Risk/Fraud, 0: Normal)

## 🛠 Technology Stack
- **Language**: Python
- **Libraries**:
    - **Data Analysis**: `Pandas`, `NumPy`
    - **Machine Learning**: `Scikit-learn`, `XGBoost`, `LightGBM`
    - **Visualization**: `Matplotlib`, `Seaborn` (implied)

## 🚀 Key Features

### 1. Data Preprocessing
- **Cleaning**: Handling missing values and outliers.
- **Feature Engineering**:
    - Scaling numerical features (`StandardScaler`).
    - Encoding categorical variables (`OneHotEncoder`).
    - Geographic data processing (Latitude/Longitude).
- **Imbalance Handling**: Utilization of `class_weight='balanced'` and stratified sampling to handle class imbalance in fraud data.

### 2. Model Development
We implemented and compared a variety of classification algorithms:
- **Linear Models**: Logistic Regression
- **Ensemble Methods**: Random Forest, Gradient Boosting
- **Advanced Boosting**: XGBoost, LightGBM
- **Others**: SVM (RBF Kernel), K-Nearest Neighbors (KNN)

### 3. Hyperparameter Tuning
- Utilized `RandomizedSearchCV` for efficient parameter optimization.
- Applied `StratifiedKFold` cross-validation to ensure reliable performance estimates.

### 4. Evaluation & Interpretability
- **Metrics**: Precision, Recall, F1-Score, ROC-AUC, and **PR-AUC** (Average Precision) focusing on the minority fraud class.
- **Feature Importance**: Analyzed top contributing features using permutation importance and model-specific importance scores.

## 📂 Repository Structure

| File/Directory | Description |
|:---|:---|
| `dataThon_ml_base_sc.ipynb` | **Main Notebook**: Contains the complete ML pipeline, model training, tuning, and evaluation. |
| `dataThon_dataPrep_sc.ipynb` | **Data Prep**: Notebook for initial data cleaning, exploratory analysis, and feature formatting. |
| `users_prep.csv` | Processed dataset ready for analysis. |
| `models_eval/` | Directory containing evaluation reports and artifacts. |
| `figures/` | Directory for saved plots and visualizations. |

## 📈 Performance Summary
The project emphasizes maximizing **Recall** and **PR-AUC** to effectively catch fraudulent cases while managing false positives. 
*(Refer to the notebooks for detailed performance tables and confusion matrices.)*

## 🔧 How to Run
1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   ```
2. **Install dependencies**:
   ```bash
   pip install pandas numpy scikit-learn xgboost lightgbm matplotlib scipy
   ```
3. **Run the Notebooks**:
   Open `dataThon_ml_base_sc.ipynb` in Jupyter Notebook or JupyterLab to execute the pipeline.

---
*Created for the DataThon Project.*