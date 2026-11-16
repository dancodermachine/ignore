"""Tests for the census income API in main.py."""

from fastapi.testclient import TestClient

from main import app

client = TestClient(app)

# Married executive with a doctorate and large capital gains.
HIGH_EARNER = {
    "age": 52,
    "workclass": "Self-emp-inc",
    "fnlgt": 287927,
    "education": "Doctorate",
    "education-num": 16,
    "marital-status": "Married-civ-spouse",
    "occupation": "Exec-managerial",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 15024,
    "capital-loss": 0,
    "hours-per-week": 55,
    "native-country": "United-States",
}

# Teenager working part time in a service job.
LOW_EARNER = {
    "age": 19,
    "workclass": "Private",
    "fnlgt": 168294,
    "education": "11th",
    "education-num": 7,
    "marital-status": "Never-married",
    "occupation": "Other-service",
    "relationship": "Own-child",
    "race": "White",
    "sex": "Female",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 20,
    "native-country": "United-States",
}


def test_get_root_returns_welcome_message():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "message": "Welcome to the Census Income Classifier API!"
    }


def test_post_predict_above_50k():
    response = client.post("/predict", json=HIGH_EARNER)

    assert response.status_code == 200
    assert response.json() == {"salary": ">50K"}


def test_post_predict_at_or_below_50k():
    response = client.post("/predict", json=LOW_EARNER)

    assert response.status_code == 200
    assert response.json() == {"salary": "<=50K"}


def test_post_predict_rejects_incomplete_record():
    incomplete = {"age": 40, "workclass": "Private"}

    response = client.post("/predict", json=incomplete)

    assert response.status_code == 422
    assert "detail" in response.json()
