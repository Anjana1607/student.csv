import os
import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Project paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "student_models.pkl")

# Create FastAPI application
app = FastAPI(
    title="Student Performance Prediction System",
    description="Machine Learning Based Student Result Prediction",
    version="1.0.0"
)

# Load trained ML model
try:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model file not found. Run check.py first.")

    if os.path.getsize(MODEL_PATH) == 0:
        raise ValueError("Model file is empty. Run check.py again.")

    model = joblib.load(MODEL_PATH)
    print("ML Model loaded successfully!")

except Exception as error:
    model = None
    print("Model loading failed:", error)


# Input validation
class StudentData(BaseModel):
    study_hours: float = Field(ge=0)
    attendance: float = Field(ge=0, le=100)
    previous_marks: float = Field(ge=0, le=100)


# Home route
@app.get("/")
def home():
    return {
        "message": "Student Performance Prediction API is running!",
        "status": "Success"
    }


# Model status route
@app.get("/model-status")
def model_status():
    if model is None:
        return {
            "model_loaded": False,
            "message": "Model is unavailable. Run check.py first."
        }

    return {
        "model_loaded": True,
        "message": "ML Model is ready!"
    }


# Prediction route
@app.post("/predict")
def predict(student: StudentData):

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="ML Model is not available."
        )

    try:
        input_data = pd.DataFrame([{
            "study_hours": student.study_hours,
            "attendance": student.attendance,
            "previous_marks": student.previous_marks
        }])

        prediction = model.predict(input_data)

        result = "PASS" if prediction[0] == 1 else "FAIL"

        return {
            "student_result": result,
            "input_data": {
                "study_hours": student.study_hours,
                "attendance": student.attendance,
                "previous_marks": student.previous_marks
            },
            "message": "Prediction completed successfully!"
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error)
        )