# 💳 Credit Card Fraud Detection using Machine Learning

This project detects fraudulent credit card transactions using machine learning models like Random Forest and Logistic Regression. The dataset used is highly imbalanced, making fraud detection a challenging yet valuable problem.

---

## 🚀 Project Overview

- **Goal:** Classify transactions as fraudulent or not.
- **Tech Stack:** Python, Scikit-learn, Pandas, Joblib
- **Algorithms Used:** Random Forest, Logistic Regression
- **Model Evaluation Metrics:** Accuracy, Confusion Matrix
- **Final Output:** Trained model saved for future prediction use.

---

## 📂 Dataset

- Source: [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Features:** V1 to V28 (PCA-transformed), Time, Amount
- **Target:** `Class` (1 = Fraud, 0 = Normal)

---

## 🧠 Workflow

1. **Data Loading & Cleaning**
   - Handled missing values
   - Balanced class inspection
2. **Exploratory Data Analysis**
   - Reviewed feature distributions and class imbalance
3. **Model Building**
   - Trained using `RandomForestClassifier` and `LogisticRegression`
4. **Evaluation**
   - Measured using accuracy and confusion matrix
5. **Model Saving**
   - Used `joblib` to serialize the final model

---

## ✅ Results

- **Random Forest**
  - Accuracy: 100%
  - Caught 2/2 frauds in test set

- **Logistic Regression**
  - Accuracy: ~99.94%
  - Confusion Matrix:
    ```
    [[1592    1]
     [   0    2]]
    ```

---
## Project Structure
credit-card-fraud-detection/
├── Credit Card Fraud detection.ipynb   # Main notebook for fraud detection
├── fraud_detection_model.joblib        # Trained machine learning model
├── requirements.txt                    # List of dependencies
└── README.md                           # Project documentation


## 🛠️ Installation
pip install pandas scikit-learn joblib
jupyter notebook "Credit Card Fraud detection.ipynb"

Clone the repository:
   ```bash
   git clone https://github.com/yourusername/credit-card-fraud-detection.git
   cd credit-card-fraud-detection

