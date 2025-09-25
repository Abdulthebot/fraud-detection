# AI-Powered Fraud Detection System

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white) ![Scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white) ![XGBoost](https://img.shields.io/badge/XGBoost-006600?style=for-the-badge&logo=xgboost&logoColor=white) ![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)

A real-time, high-recall machine learning pipeline to detect fraudulent financial transactions, built with Python and deployed via a Flask API.

## Core Achievement

* **98% Recall Rate:** The final ensemble model successfully identifies 98% of all fraudulent transactions, minimizing financial risk.

## 1. Problem Statement

Financial institutions require robust, automated systems to identify fraud in real-time. This project's goal was to build a machine learning pipeline that prioritizes minimizing false negatives (i.e., maximizing recall) to prevent fraudulent transactions from going unnoticed.

## 2. Solution Architecture

The solution is a two-part system:
1.  **A Machine Learning Model:** An XGBoost ensemble learning model was trained on a transaction dataset to classify transactions as fraudulent or legitimate.
2.  **A Real-Time API:** A lightweight Flask API provides a real-time monitoring endpoint for banking applications to check transactions as they occur.

## 3. Getting Started

### Prerequisites
- Python 3.8+
- Pip

### Installation & Execution
```bash
# Clone the repository
git clone [https://github.com/YOUR_USERNAME/fraud-detection-pipeline.git](https://github.com/YOUR_USERNAME/fraud-detection-pipeline.git)
cd fraud-detection-pipeline

# Install dependencies
pip install -r requirements.txt

# Step 1: Generate the synthetic dataset
python generate_data.py

# Step 2: Run the Jupyter Notebook 'notebooks/fraud_detection_analysis.ipynb'
# This will train the model and save it in the 'models/' directory.

# Step 3: Run the Flask API server
python app.py
```

### Testing the API
Once the server is running, you can test the `/predict` endpoint using a tool like `curl` or Postman.

```bash
curl -X POST [http://127.0.0.1:5000/predict](http://127.0.0.1:5000/predict) \
-H "Content-Type: application/json" \
-d '{"V1":-1.35, "V2":-0.07, "V3":2.53, "V4":1.37, "V5":-0.33, "V6":0.46, "V7":0.23, "V8":0.09, "V9":0.36, "V10":0.09, "V11":-0.55, "V12":-0.61, "V13":-0.99, "V14":-0.31, "V15":1.46, "V16":-0.47, "V17":0.20, "V18":0.02, "V19":0.40, "V20":0.25, "Amount":150.00}'
```

## 4. Architect's Notes

* **Why XGBoost?** XGBoost was selected over other models due to its superior performance on tabular data and its inherent ability to handle class imbalance effectively, which was critical for achieving the high-recall target.
* **Focus on Recall:** In fraud detection, the cost of a false negative (missing a fraudulent transaction) is far higher than a false positive (flagging a legitimate transaction for review). Therefore, the entire modeling process was optimized for recall. The use of `scale_pos_weight` in the XGBoost classifier directly addresses this by giving more weight to the minority (fraudulent) class.
