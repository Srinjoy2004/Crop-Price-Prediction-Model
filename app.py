from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib  # modern way to load sklearn models
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

app = Flask(__name__)

# Load the list of columns used during model training
training_columns = list(pd.read_csv("encoded_feature_columns.csv").columns)

# Lazy-load model to avoid high memory usage at startup
def get_model():
    if not hasattr(get_model, "model"):
        get_model.model = joblib.load("prediction_model.pkl")
        if not isinstance(get_model.model, (RandomForestRegressor, DecisionTreeRegressor)):
            raise TypeError("Loaded model is not a valid RandomForestRegressor or DecisionTreeRegressor.")
    return get_model.model

# Load dataset for fetching reference options
apy_df = pd.read_csv("APY.csv", encoding="utf-8")
apy_df.rename(columns=lambda x: x.strip(), inplace=True)

# Extract dropdown options
states = sorted(apy_df['State'].dropna().unique().tolist())
districts = sorted(apy_df['District'].dropna().unique().tolist())
crops = sorted(apy_df['Crop'].dropna().unique().tolist())
seasons = sorted(apy_df['Season'].dropna().unique().tolist())

@app.route('/')
def home():
    return "Crop Production Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    try:
        # Extract and validate input
        state = data['state']
        district = data['district']
        crop = data['crop']
        season = data['season']
        area = float(data['area'])
        year = int(data['year'])

        if area <= 0:
            return jsonify({'error': "Area must be greater than zero."}), 400
        if year < 2000 or year > 2050:
            return jsonify({'error': "Year should be between 2000 and 2050."}), 400

        # Create DataFrame for prediction
        row = pd.DataFrame({
            'State': [state],
            'District': [district],
            'Crop_Year': [year],
            'Season': [season],
            'Crop': [crop],
            'Area': [area]
        })

        # One-hot encode and align with training columns
        row_encoded = pd.get_dummies(row)
        missing_cols = [col for col in training_columns if col not in row_encoded.columns]
        for col in missing_cols:
            row_encoded[col] = 0
        row_encoded = row_encoded[training_columns]

        # Predict using model
        model = get_model()
        prediction = model.predict(row_encoded)[0]
        return jsonify({'predicted_production': round(prediction, 2)})

    except KeyError as e:
        return jsonify({'error': f'Missing input field: {e}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
