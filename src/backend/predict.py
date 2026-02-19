import os
import joblib
import pandas as pd

# Base directory of the project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(BASE_DIR, "models")

rf_model = joblib.load(os.path.join(MODEL_DIR, "random_forest_v1.pkl"))
xgb_model = joblib.load(os.path.join(MODEL_DIR, "xgboost_v1.pkl"))


def rule_override(soil_moisture):
    """
    Safety rule override based on soil moisture.
    """
    if soil_moisture > 90:
        return 2  # Overwatered
    if soil_moisture < 20:
        return 1  # Needs Water
    return None


def ensemble_prediction(input_data):
    """
    Combines Random Forest and XGBoost predictions
    with rule-based override.

    input_data: dictionary containing ALL 18 required features
    """

    # 1️⃣ Rule-based override first
    soil_moisture = input_data["Soil_Moisture"]
    rule_decision = rule_override(soil_moisture)
    if rule_decision is not None:
        return rule_decision

    # 2️⃣ Convert dictionary to DataFrame with EXACT feature order
    features = pd.DataFrame([{
        "Plant_ID": input_data["Plant_ID"],
        "Soil_Moisture": input_data["Soil_Moisture"],
        "Ambient_Temperature": input_data["Ambient_Temperature"],
        "Soil_Temperature": input_data["Soil_Temperature"],
        "Humidity": input_data["Humidity"],
        "Light_Intensity": input_data["Light_Intensity"],
        "Soil_pH": input_data["Soil_pH"],
        "Nitrogen_Level": input_data["Nitrogen_Level"],
        "Phosphorus_Level": input_data["Phosphorus_Level"],
        "Potassium_Level": input_data["Potassium_Level"],
        "Chlorophyll_Content": input_data["Chlorophyll_Content"],
        "Electrochemical_Signal": input_data["Electrochemical_Signal"],
        "days_since_last_watering": input_data["days_since_last_watering"],
        "watering_sma_3": input_data["watering_sma_3"],
        "Year": input_data["Year"],
        "Month": input_data["Month"],
        "Day": input_data["Day"],
        "Hour": input_data["Hour"],
    }])

    # 3️⃣ Model predictions
    rf_pred = rf_model.predict(features)[0]
    xgb_pred = xgb_model.predict(features)[0]

    # 4️⃣ Majority voting
    predictions = [rf_pred, xgb_pred]
    final_prediction = max(set(predictions), key=predictions.count)

    return int(final_prediction)
