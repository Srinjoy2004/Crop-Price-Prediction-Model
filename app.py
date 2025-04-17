from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

app = Flask(__name__)

# Get base directory for safe file paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load training columns safely
training_columns_path = os.path.join(BASE_DIR, "encoded_feature_columns.csv")
training_columns = list(pd.read_csv(training_columns_path).columns)

# Lazy-load model
model = None

def get_model():
    global model
    if model is None:
        model_path = os.path.join(BASE_DIR, "prediction_model.pkl")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file missing: {model_path}")
        model = joblib.load(model_path)
        if not isinstance(model, (RandomForestRegressor, DecisionTreeRegressor)):
            raise TypeError("Loaded model is not a valid RandomForestRegressor or DecisionTreeRegressor.")
    return model

# Load dataset for reference options
apy_path = os.path.join(BASE_DIR, "APY.csv")
apy_df = pd.read_csv(apy_path, encoding="utf-8")
apy_df.rename(columns=lambda x: x.strip(), inplace=True)

# Extract dropdown options (optional if you plan to expose them via route)
states = sorted(apy_df['State'].dropna().unique().tolist())
districts = sorted(apy_df['District'].dropna().unique().tolist())
crops = sorted(apy_df['Crop'].dropna().unique().tolist())
seasons = sorted(apy_df['Season'].dropna().unique().tolist())

@app.route('/')
def home():
    return "✅ Crop Production Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        print("🚀 Predict endpoint hit")
        print("Data received:", data)

        # Input parsing
        state = data['state']
        district = data['district']
        crop = data['crop']
        season = data['season']
        area = float(data['area'])
        year = int(data['year'])

        # Input validation
        if area <= 0:
            return jsonify({'error': "Area must be greater than zero."}), 400
        if year < 2000 or year > 2050:
            return jsonify({'error': "Year should be between 2000 and 2050."}), 400

        # Prepare input DataFrame
        row = pd.DataFrame({
            'State': [state],
            'District': [district],
            'Crop_Year': [year],
            'Season': [season],
            'Crop': [crop],
            'Area': [area]
        })

        # One-hot encoding + column alignment
        row_encoded = pd.get_dummies(row)
        missing_cols = [col for col in training_columns if col not in row_encoded.columns]
        for col in missing_cols:
            row_encoded[col] = 0
        row_encoded = row_encoded[training_columns]

        # Make prediction
        model = get_model()
        prediction = model.predict(row_encoded)[0]
        return jsonify({'predicted_production': round(prediction, 2)})

    except KeyError as e:
        return jsonify({'error': f'Missing input field: {e}'}), 400
    except Exception as e:
        print("❌ Error in /predict:", str(e))
        return jsonify({'error': str(e)}), 500

@app.route('/testmodel')
def test_model():
    try:
        model_path = os.path.join(BASE_DIR, "prediction_model.pkl")
        if not os.path.exists(model_path):
            return f"❌ Model file missing: {model_path}"
        
        # Ensure model is loaded properly
        model = get_model()
        return f"✅ Model loaded successfully: {type(model)}"
    except FileNotFoundError as e:
        return f"❌ {str(e)}"
    except Exception as e:
        return f"❌ Model loading failed: {str(e)}"

@app.route('/testfiles')
def test_files():
    try:
        # Check if files exist
        assert os.path.exists("prediction_model.pkl"), "❌ prediction_model.pkl is missing"
        assert os.path.exists("encoded_feature_columns.csv"), "❌ encoded_feature_columns.csv is missing"
        return "✅ Both files exist."
    except AssertionError as e:
        return f"{str(e)}"

# Load model at app startup
@app.before_first_request
def load_model():
    global model
    try:
        model = get_model()
    except Exception as e:
        print(f"❌ Error during model loading at startup: {str(e)}")

if __name__ == '__main__':
    app.run(debug=False)
