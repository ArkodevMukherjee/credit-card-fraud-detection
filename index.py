from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('random_forest_model.pkl')

@app.route('/')
def index():
    return "Random Forest Fraud Detection API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)

        # Extract features from JSON
        features = data.get('features')
        if not features or len(features) != 30:
            return jsonify({'error': 'Invalid input. Expected 30 numerical features.'}), 400

        # Convert input to numpy array
        input_array = np.array(features).reshape(1, -1)

        # Make prediction
        prediction = model.predict(input_array)[0]
        probability = model.predict_proba(input_array)[0][1]

        return jsonify({
            'prediction': int(prediction),
            'fraud_probability': round(probability, 4)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)