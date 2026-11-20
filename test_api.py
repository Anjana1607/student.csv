import requests

BASE_URL = "http://127.0.0.1:8000"

print("STARTING API TESTS...\n")

# Test 1: Home endpoint
try:
    response = requests.get(BASE_URL + "/")

    print("TEST 1 - HOME API")
    print("Status Code:", response.status_code)
    print("Response:", response.json())

except requests.exceptions.ConnectionError:
    print("ERROR: API server is not running!")
    exit()


# Test 2: Model status
print("\nTEST 2 - MODEL STATUS")

response = requests.get(BASE_URL + "/model-status")

print("Status Code:", response.status_code)
print("Response:", response.json())


# Test 3: Prediction endpoint
print("\nTEST 3 - STUDENT PREDICTION")

student_data = {
    "study_hours": 7,
    "attendance": 85,
    "previous_marks": 70
}

response = requests.post(
    BASE_URL + "/predict",
    json=student_data
)

print("Status Code:", response.status_code)
print("Response:", response.json())

if response.status_code == 200:
    print("\nAPI TEST COMPLETED SUCCESSFULLY!")
else:
    print("\nAPI TEST FAILED!")