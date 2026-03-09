# Insurance Fraud Detection

Supervised classification project detecting fraudulent insurance claims using Logistic Regression, Decision Tree, Random Forest, and SVM, with hyperparameter tuning and model comparison.

## Overview

Insurance fraud costs the industry billions annually and is notoriously difficult to detect due to class imbalance — fraudulent claims are a small minority of all claims. This project builds and compares multiple classification models to identify fraudulent claims, handling missing data, class imbalance through undersampling, and feature engineering before model training.

## Dataset

- **Source:** Kaggle — Insurance Fraud Detection Dataset
- **Target:** `fraud_reported` — binary, Y = fraud, N = no fraud
- **Features:** Policy details, incident information, customer demographics, claim amounts

## Project Structure

├── Insurance_Fraud_Project.ipynb    
├── README.md                        


## Methodology

### 1. Exploratory Data Analysis
- Replaced `?` placeholder values with NaN for proper missing value handling
- Analyzed numerical and categorical feature distributions
- Checked skewness across all numeric columns
- Examined class balance of the fraud target variable

### 2. Preprocessing
- Dropped high-cardinality and low-signal columns (policy ID, incident city, hobbies, etc.)
- Median imputation for numeric missing values, mode imputation for categorical
- Feature engineering:
  - Grouped education levels into Basic / College / Advanced
  - Binned customer age into Young / Mid / Senior groups
- Applied dummy coding (one-hot encoding) for all categorical variables

### 3. Class Imbalance Handling
- Used **random undersampling** of the majority class to balance training data
- Ensures the model is not biased toward predicting non-fraud for every case

### 4. Models & Results

Five models were trained and compared:

| Model | Accuracy | Notes |
|---|---|---|
| Logistic Regression | Baseline | Good recall, lower precision |
| Decision Tree | Comparable | Interpretable but prone to overfitting |
| Random Forest (Run 1) | Improved | Better generalization |
| Random Forest (Tuned) | Best overall | GridSearchCV optimization |
| SVM (RBF kernel) | Competitive | Tuned with GridSearchCV |

### 5. Hyperparameter Tuning
Both Random Forest and SVM were tuned using **GridSearchCV**:

**Random Forest parameters tuned:**
- `n_estimators`: number of trees
- `max_depth`: tree depth limit
- `min_samples_leaf`: minimum samples per leaf

**SVM parameters tuned:**
- `C`: regularization strength
- `gamma`: kernel coefficient
- `kernel`: RBF vs linear

### 6. Model Evaluation
All models evaluated using:
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix
- Final comparison table across all models

## Key Findings

- Random Forest with tuning produced the best overall performance
- Undersampling was effective at improving recall for the minority fraud class
- SVM with RBF kernel was competitive but slower to train
- Decision Trees were interpretable but overfit without proper depth constraints
- Feature engineering (age groups, education grouping) helped simplify the feature space

## Tools & Libraries

- Python, Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn (LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, SVC, GridSearchCV, StandardScaler, train_test_split)

## How to Run

1. Clone the repository
2. Upload the insurance fraud dataset to your Google Drive
3. Open the notebook in Google Colab
4. Update the file path in the first cell
5. Run all cells in order
