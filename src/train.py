import os
import pickle
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from preprocessing import load_data, preprocess

# =========================
# 🔥 PATH SETUP
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

data_path = os.path.join(BASE_DIR, "data", "loan_prediction_dataset.csv")
model_path = os.path.join(BASE_DIR, "models", "model.pkl")

# =========================
# 📊 LOAD DATA
# =========================
df = load_data(data_path)

print("\n🔹 Target Distribution:")
print(df['Loan_Approved'].value_counts())

# =========================
# 🔧 PREPROCESS
# =========================
df = preprocess(df)

# =========================
# 📊 CORRELATION
# =========================
print("\n🔹 Feature Correlation with Target:")
print(df.corr(numeric_only=True)['Loan_Approved'].sort_values(ascending=False))

# =========================
# 🎯 SPLIT
# =========================
X = df.drop("Loan_Approved", axis=1)
y = df["Loan_Approved"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# 🔥 MODEL 1: LOGISTIC REGRESSION (WITH SCALING)
# =========================
print("\n===== Logistic Regression =====")

lr_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=2000))
])

lr_model.fit(X_train, y_train)

y_pred_lr = lr_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print("\nClassification Report:\n", classification_report(y_test, y_pred_lr))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lr))

# =========================
# 🔥 MODEL 2: RANDOM FOREST (BALANCED)
# =========================
print("\n===== Random Forest =====")

rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    class_weight='balanced',   # 🔥 IMPORTANT
    random_state=42
)

rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("\nClassification Report:\n", classification_report(y_test, y_pred_rf))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))

# =========================
# 🔁 CROSS VALIDATION
# =========================
cv_scores = cross_val_score(rf_model, X, y, cv=5)

print("\n🔹 Cross Validation Scores:", cv_scores)
print("🔹 Mean CV Score:", cv_scores.mean())

# =========================
# 💾 SAVE MODEL
# =========================
pickle.dump(rf_model, open(model_path, "wb"))

print("\n✅ Model saved successfully at:", model_path)