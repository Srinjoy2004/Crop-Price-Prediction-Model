from flask import Flask, request, jsonify
import pandas as pd
import pickle
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

app = Flask(__name__)

# Load trained model and column names
model = pickle.load(open("prediction_model.pkl", "rb"))
training_columns = list(pd.read_csv("encoded_feature_columns.csv").columns)

# Validate model
if not isinstance(model, (RandomForestRegressor, DecisionTreeRegressor)):
    raise TypeError("Loaded model is not a valid RandomForestRegressor or DecisionTreeRegressor.")

# Load dataset to fetch options (can be used to validate inputs if needed)
apy_df = pd.read_csv("APY.csv", encoding="utf-8")
apy_df.rename(columns=lambda x: x.strip(), inplace=True)

# Sample dropdown options (optional, for reference or future GET endpoints)
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

        # Create input DataFrame
        row = pd.DataFrame({
            'State': [state],
            'District': [district],
            'Crop_Year': [year],
            'Season': [season],
            'Crop': [crop],
            'Area': [area]
        })

        # One-hot encoding
        row_encoded = pd.get_dummies(row)

        # Add missing columns
        # Identify missing columns
        missing_cols = [col for col in training_columns if col not in row_encoded.columns]

# Create a DataFrame of zeros for those missing columns
        missing_df = pd.DataFrame(0, index=row_encoded.index, columns=missing_cols)

# Concatenate them all at once
        row_encoded = pd.concat([row_encoded, missing_df], axis=1)


        # Reorder columns
        row_encoded = row_encoded[training_columns]

        # Predict
        pred = model.predict(row_encoded)[0]
        return jsonify({'predicted_production': round(pred, 2)})

    except KeyError as e:
        return jsonify({'error': f'Missing input field: {e}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
