"""
Census Income Prediction - Baseline Approach (Gradient Boosting)
----------------------------------------------------------------
This script implements a production-grade XGBoost classifier to predict
income brackets (>50K vs <=50K).

Key Features:
1. Robust Preprocessing: Handling missing values and categorical encoding.
2. Hyperparameter Tuning: RandomizedSearchCV to optimize tree depth, learning rate, and regularization.
3. Imbalance Handling: Calculation of 'scale_pos_weight' vs standard training.
4. Threshold Tuning: Analysis of Precision-Recall trade-offs for risk-sensitive deployment.

Performance Benchmark: (Use case need precision)
- Accuracy: ~87.5%
- Precision (>50k): ~80%
- Recall (>50k): ~64%

Author: Devansh
Date: 2025
"""

import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_curve
from collections import Counter
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
CONFIG = {
    'seed': 42,
    'test_size': 0.3,
    'cv_folds': 3,
    'n_iter_search': 50,
    'target_precision': 0.85, # For safety threshold analysis
    'model_path': '../backend/model_tuned.pkl' # Where to save the final model
}

def load_and_preprocess_data(filepath='adult.csv'):
    """
    Loads data, handles missing values, discretizes features, and encodes categories.
    Returns X_train, X_test, y_train, y_test, and feature metadata.
    """
    print(f"Loading data from {filepath}...")
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        # Fallback for folder structure
        df = pd.read_csv('../data/adult.csv')

    # 1. Handle Missing Values
    df = df.replace("?", np.nan)
    for col in df.columns:
        # Fill with mode (most frequent value)
        df[col] = df[col].fillna(df[col].mode()[0])

    # 2. Discretization (Marital Status)
    # Normalize column names (dot vs hyphen)
    ms_col = 'marital.status' if 'marital.status' in df.columns else 'marital-status'
    
    df[ms_col].replace(
        ['Divorced', 'Married-AF-spouse', 'Married-civ-spouse', 'Married-spouse-absent',
         'Never-married', 'Separated', 'Widowed'],
        ['divorced', 'married', 'married', 'married',
         'not married', 'not married', 'not married'], inplace=True
    )

    # 3. Label Encoding
    labelEncoder = preprocessing.LabelEncoder()
    cols_to_encode = ['workclass', 'race', 'education', ms_col, 'occupation', 
                      'relationship', 'sex', 'native.country', 'income']

    actual_cols = df.columns.tolist()
    for col in cols_to_encode:
        # Handle variations in column names
        if col in actual_cols:
            df[col] = labelEncoder.fit_transform(df[col])
        elif col == 'native.country' and 'country' in actual_cols:
            df['country'] = labelEncoder.fit_transform(df['country'])
        elif col == 'income' and 'salary' in actual_cols:
            df['salary'] = labelEncoder.fit_transform(df['salary'])

    # 4. Feature Selection
    # Drop redundant columns (fnlwgt is noise, education.num is redundant with education)
    drop_cols = [c for c in ['fnlwgt', 'education.num', 'education-num'] if c in df.columns]
    df = df.drop(drop_cols, axis=1)

    # 5. Split
    X = df.iloc[:, :-1].values
    Y = df.iloc[:, -1].values
    
    # Stratify ensuring class distribution remains consistent
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=CONFIG['test_size'], random_state=CONFIG['seed'], stratify=Y
    )

    print(f"Data Split -> Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test


def optimize_xgboost(X_train, y_train):
    """
    Runs RandomizedSearchCV to find the best hyperparameters for XGBoost.
    Includes optimization for class imbalance.
    """
    # Calculate scale_pos_weight for imbalance
    counter = Counter(y_train)
    estimate = counter[0] / counter[1]
    print(f"Class Imbalance Ratio (Neg/Pos): {estimate:.2f}")

    # Hyperparameter Grid
    param_dist = {
        'n_estimators': [100, 200, 300, 500],        # Number of trees
        'learning_rate': [0.01, 0.05, 0.1, 0.2],     # Step size
        'max_depth': [3, 4, 5, 6, 8, 10],            # Depth of trees
        'colsample_bytree': [0.6, 0.7, 0.8, 1.0],    # Feature fraction
        'subsample': [0.6, 0.7, 0.8, 1.0],           # Row fraction
        'scale_pos_weight': [1, estimate],           # Test standard vs weighted
        'gamma': [0, 0.1, 0.2]                       # Regularization
    }

    xgb_model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=CONFIG['seed']
    )

    print("\nStarting Hyperparameter Tuning...")
    random_search = RandomizedSearchCV(
        estimator=xgb_model,
        param_distributions=param_dist,
        n_iter=CONFIG['n_iter_search'],
        scoring='accuracy', # Prioritizing overall correctness
        cv=CONFIG['cv_folds'],
        verbose=1,
        n_jobs=-1,
        random_state=CONFIG['seed']
    )

    random_search.fit(X_train, y_train)
    print(f"Best Parameters: {random_search.best_params_}")
    
    return random_search.best_estimator_


def evaluate_model(model, X_test, y_test):
    """
    Generates comprehensive evaluation metrics: Accuracy, Classification Report,
    Confusion Matrix, and Precision-Recall Threshold Analysis.
    """
    print("\n--- Evaluation Report ---")
    
    # Predictions
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    # Standard Metrics
    acc = accuracy_score(y_test, preds)
    print(f"Accuracy: {acc*100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, preds, target_names=['<=50K', '>50K']))

    # Confusion Matrix Plot
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
                xticklabels=['<=50K', '>50K'], yticklabels=['<=50K', '>50K'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix (XGBoost)')
    plt.show()

    # Precision-Recall Safety Analysis
    print(f"\n--- Safety Analysis (Target Precision: {CONFIG['target_precision']*100}%) ---")
    precisions, recalls, thresholds = precision_recall_curve(y_test, probs)
    
    # Find index where precision meets target
    # Note: precisions array is length thresholds + 1
    idx = next((i for i, p in enumerate(precisions) if p >= CONFIG['target_precision']), -1)
    
    if idx != -1 and idx < len(thresholds):
        safe_recall = recalls[idx]
        safe_threshold = thresholds[idx]
        print(f"To achieve {CONFIG['target_precision']*100}% Precision:")
        print(f" - Set Probability Threshold to: {safe_threshold:.4f}")
        print(f" - Expected Recall (Loans Issued): {safe_recall*100:.2f}%")
    else:
        print(f"Could not achieve {CONFIG['target_precision']*100}% Precision with this model.")


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # 1. Load Data
    X_train, X_test, y_train, y_test = load_and_preprocess_data()

    # 2. Train & Tune
    best_model = optimize_xgboost(X_train, y_train)

    # 3. Evaluate
    evaluate_model(best_model, X_test, y_test)

    # 4. Save Model (Optional, uncomment if you want to overwrite)
    # pickle.dump(best_model, open(CONFIG['model_path'], 'wb'))
    # print(f"\nModel saved to {CONFIG['model_path']}")