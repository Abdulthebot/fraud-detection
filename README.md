# AI-Powered Fraud Detection System

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white) ![Scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white) ![XGBoost](https://img.shields.io/badge/XGBoost-006600?style=for-the-badge&logo=xgboost&logoColor=white) ![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)

[cite_start]A real-time, high-recall machine learning pipeline to detect fraudulent financial transactions, built with Python and deployed via a Flask API[cite: 24, 25].

## Core Achievement

* [cite_start]**98% Recall Rate:** The final ensemble model successfully identifies 98% of all fraudulent transactions, minimizing financial risk.

## 1. Problem Statement

Financial institutions require robust, automated systems to identify fraud in real-time. This project's goal was to build a machine learning pipeline that prioritizes minimizing false negatives (i.e., maximizing recall) to prevent fraudulent transactions from going unnoticed.

## 2. Solution Architecture

The solution is a two-part system:
1.  [cite_start]**A Machine Learning Model:** An XGBoost ensemble learning model was trained on a transaction dataset to classify transactions as fraudulent or legitimate[cite: 24, 25].
2.  [cite_start]**A Real-Time API:** A lightweight Flask API provides a real-time monitoring endpoint for banking applications to check transactions as they occur.

## 3. Getting Started

### Prerequisites
- Python 3.8+
- Pip

### Installation & Execution
```bash
# Clone the repository
git clone [https://github.com/Abdulthebot/fraud-detection-pipeline.git](https://github.com/Abdulthebot/fraud-detection-pipeline.git)
cd fraud-detection-pipeline

# Install dependencies
pip install -r requirements.txt

# Run the Flask API server
python app.py
```

## 4. Architect's Notes

* **Why XGBoost?** XGBoost was selected over other models due to its superior performance on tabular data and its inherent ability to handle class imbalance effectively, which was critical for achieving the high-recall target.
* **Focus on Recall:** In fraud detection, the cost of a false negative (missing a fraudulent transaction) is far higher than a false positive (flagging a legitimate transaction for review). Therefore, the entire modeling process was optimized for recall, resulting in the 98% score.

---
