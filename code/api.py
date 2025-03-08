from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import pickle

app = Flask(__name__)

# Load or train model (for simplicity, train here; in production, load from file)
def load_model():
    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        print("Model loaded from model.pkl")
        return model
    except FileNotFoundError:
        print("Error: model.pkl not found. Run ccpp_model.py first to train and save the model.")
        exit(1)

model = load_model()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Expect JSON like {"T": 25.0, "AP": 1013.0, "RH": 60.0, "V": 40.0}
        data = request.get_json()
        input_data = pd.DataFrame([[data['T'], data['AP'], data['RH'], data['V']]], 
                                columns=['T', 'AP', 'RH', 'V'])
        prediction = model.predict(input_data)[0]
        return jsonify({'prediction': float(prediction), 'unit': 'MW'})
    except KeyError as e:
        return jsonify({'error': f'Missing field: {e}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)