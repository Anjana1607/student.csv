import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

print("1. Starting program...", flush=True)

# File paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "Dataset", "student.csv")
CLEAN_PATH = os.path.join(BASE_DIR, "Dataset", "students_clean.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model", "student_models.pkl")

# Read dataset
print("2. Reading dataset...", flush=True)
data = pd.read_csv(DATA_PATH)

print(data.head(), flush=True)

# Check missing values
print("3. Checking missing values...", flush=True)
print(data.isnull().sum(), flush=True)

data = data.dropna(subset=["result"])

numerical_columns = ["study_hours", "attendance", "previous_marks"]

for column in numerical_columns:
    data[column] = data[column].fillna(data[column].median())

# Remove duplicates
print("4. Removing duplicates...", flush=True)
data = data.drop_duplicates()

# Save cleaned dataset
data.to_csv(CLEAN_PATH, index=False)
print("5. Cleaned dataset saved.", flush=True)

# Convert result into numerical values
data["result"] = data["result"].astype(str).str.strip().str.title()
data = data[data["result"].isin(["Pass", "Fail"])].copy()
data["result"] = data["result"].map({"Pass": 1, "Fail": 0})

x = data[["study_hours", "attendance", "previous_marks"]]
y = data["result"]

if y.nunique() < 2 or y.value_counts().min() < 2:
    raise ValueError("Need at least two records for each result class.")

# Split dataset
print("6. Splitting dataset...", flush=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# Train model
print("7. Training Random Forest...", flush=True)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=1
)

model.fit(x_train, y_train)

print("8. Model training completed!", flush=True)

# Evaluate model
y_pred = model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy * 100, "%", flush=True)
print(classification_report(y_test, y_pred, zero_division=0))

# Save model
print("9. Saving model...", flush=True)

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

joblib.dump(model, MODEL_PATH)

print("10. SUCCESS! Model saved.", flush=True)
print("Model location:", MODEL_PATH, flush=True)
print("Model size:", os.path.getsize(MODEL_PATH), "bytes", flush=True)