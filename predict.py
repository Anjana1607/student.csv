import os
import sys
import joblib
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "student_models.pkl")

# Load trained model
if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) == 0:
    print("Model file missing or empty. Run check.py first.")
    sys.exit(1)

model = joblib.load(MODEL_PATH)
print("Model loaded successfully!")

# Get student details
study_hours = float(input("Enter study hours: "))
attendance = float(input("Enter attendance percentage: "))
previous_marks = float(input("Enter previous marks: "))

# Prepare input
input_data = pd.DataFrame([{
    "study_hours": study_hours,
    "attendance": attendance,
    "previous_marks": previous_marks
}])

# Predict
prediction = model.predict(input_data)

result = "PASS" if prediction[0] == 1 else "FAIL"

print("\n------------------------------")
print("STUDENT PERFORMANCE RESULT")
print("------------------------------")
print("Study Hours:", study_hours)
print("Attendance:", attendance, "%")
print("Previous Marks:", previous_marks)
print("Predicted Result:", result)
print("------------------------------")