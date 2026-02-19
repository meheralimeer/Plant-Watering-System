from predict import ensemble_prediction

sample_input = {
    "Plant_ID": 1,
    "Soil_Moisture": 35,
    "Ambient_Temperature": 28,
    "Soil_Temperature": 26,
    "Humidity": 65,
    "Light_Intensity": 400,
    "Soil_pH": 6.5,
    "Nitrogen_Level": 40,
    "Phosphorus_Level": 30,
    "Potassium_Level": 25,
    "Chlorophyll_Content": 45,
    "Electrochemical_Signal": 0.8,
    "days_since_last_watering": 2,
    "watering_sma_3": 1,
    "Year": 2026,
    "Month": 2,
    "Day": 19,
    "Hour": 14
}

result = ensemble_prediction(sample_input)

print("Final Prediction:", result)
