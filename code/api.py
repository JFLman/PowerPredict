from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import pickle

app = Flask(__name__)

# Load or train model (for simplicity, train here; in production, load from file)
def load_or_train_model():
    data = pd.read_csv('data/CCPP_data.csv')
    X = data[['T', 'AP', 'RH', 'V']]
    y = data['PE']
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X, y)
    # Optionally save for reuse: pickle.dump(model, open('model.pkl', 'wb'))
    return model

model = load_or_train_model()

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