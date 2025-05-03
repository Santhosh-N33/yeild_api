import os
import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load trained model and pre-saved encoders and scaler
MODEL = load_model("D:/yield_api/lstm_yield_model.h5",compile=False)
ENCODERS = joblib.load("D:/yield_api/label_encoders.pkl")
SCALER = joblib.load("D:/yield_api/scaler.pkl")

# Define feature order
FEATURES = [
    "State", "District", "Crop", "Season", "SowingMonth",
    "Area(ha)", "N(kg/ha)", "P(kg/ha)", "K(kg/ha)",
    "SoilType", "AnnualRainfall(mm)", "AvgTemperature(C)"
]

@app.route("/predict", methods=["POST"])
def predict_yield():
    try:
        payload = request.get_json()
        if not payload:
            return jsonify({"error": "Empty request"}), 400

        # Convert input to dataframe
        row = pd.DataFrame([payload], columns=FEATURES)

        # Label encode categorical fields
        for col in ENCODERS:
            if col in row:
                row[col] = ENCODERS[col].transform(row[col])

        # Scale the data
        X = SCALER.transform(row.values)

        # Reshape for LSTM
        X = X.reshape((1, 1, len(FEATURES)))

        # Make prediction
        pred = MODEL.predict(X)[0][0]

        return jsonify({"predicted_yield(t/ha)": round(float(pred), 2)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5006))
    app.run(host="0.0.0.0", port=port)
