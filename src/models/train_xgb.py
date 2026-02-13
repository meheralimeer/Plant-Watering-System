import os
import joblib
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ---------------------------
# 1️⃣ Load Preprocessed Data
# ---------------------------
X_train = pd.read_csv("data/processed/X_train.csv")
X_test = pd.read_csv("data/processed/X_test.csv")
y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()
y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()

# ---------------------------
# 2️⃣ Handle Timestamp Column
# ---------------------------
if "Timestamp" in X_train.columns:
    for df in [X_train, X_test]:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        df["hour"]        = df["Timestamp"].dt.hour
        df["day"]         = df["Timestamp"].dt.day
        df["month"]       = df["Timestamp"].dt.month
        df["day_of_week"] = df["Timestamp"].dt.dayofweek
        df.drop(columns=["Timestamp"], inplace=True)

# ---------------------------
# 3️⃣ Create Folders if Not Exist
# ---------------------------
os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)
os.makedirs("plots", exist_ok=True)

# ---------------------------
# 4️⃣ Train XGBoost Model
# ---------------------------
model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    random_state=42,
    eval_metric='logloss'
)

model.fit(X_train, y_train)

# ---------------------------
# 5️⃣ Predictions
# ---------------------------
y_pred = model.predict(X_test)

# ---------------------------
# 6️⃣ Metrics
# ---------------------------
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("Accuracy:", accuracy)
print("\nClassification Report:\n", report)

# Save metrics to file
with open("reports/metrics.txt", "w") as f:
    f.write(f"Accuracy: {accuracy}\n\n")
    f.write("Classification Report:\n")
    f.write(report)

# ---------------------------
# 7️⃣ Confusion Matrix Plot
# ---------------------------
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("plots/confusion_matrix.png")
plt.close()

# ---------------------------
# 8️⃣ Feature Importance Plot
# ---------------------------
plt.figure(figsize=(8,6))
xgb.plot_importance(model, importance_type='weight')
plt.title("Feature Importance")
plt.savefig("plots/feature_importance.png")
plt.close()

# ---------------------------
# 9️⃣ Save Trained Model
# ---------------------------
joblib.dump(model, "models/xgboost_v1.pkl")

print("✅ Model, metrics, and plots saved successfully.")
