from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Initialize Flask application
app = Flask(__name__)

# Load the trained model
# Make sure the path is correct based on your folder structure
try:
    model = joblib.load('models/xgboost_model.pkl')
    print("Model loaded successfully.")
except FileNotFoundError:
    print("Model file not found. Make sure you've run the notebook to train and save the model.")
    model = None

@app.route('/')
def home():
    return "Fraud Detection API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model is not loaded'}), 500

    try:
        # Get data from the POST request
        data = request.get_json(force=True)
        
        # Convert the JSON data into a pandas DataFrame
        # The model expects a DataFrame with the same columns as the training data
        df_to_predict = pd.DataFrame(data, index=[0])
        
        # Make prediction
        prediction = model.predict(df_to_predict)
        
        # Get prediction probability
        probability = model.predict_proba(df_to_predict)
        
        # Prepare the response
        is_fraud = int(prediction[0])
        fraud_probability = float(probability[0][1])

        response = {
            'is_fraud': is_fraud,
            'fraud_probability': fraud_probability
        }
        
        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # Use port 5000 by default, useful for local testing
    app.run(debug=True, port=5000)
