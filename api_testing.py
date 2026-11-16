"""Send a sample census record to the deployed prediction API."""

import requests

API_URL = "https://ignore-96g8.onrender.com/predict"
SAMPLE_RECORD = {
    "age": 34,
    "workclass": "Private",
    "fnlgt": 198872,
    "education": "Bachelors",
    "education-num": 13,
    "marital-status": "Never-married",
    "occupation": "Tech-support",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Female",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States",
}


def main() -> None:
    """Request a prediction and print the HTTP status and API response."""
    response = requests.post(API_URL, json=SAMPLE_RECORD, timeout=30)
    print(f"HTTP status: {response.status_code}")
    response.raise_for_status()
    print(f"Prediction response: {response.json()}")


if __name__ == "__main__":
    main()
