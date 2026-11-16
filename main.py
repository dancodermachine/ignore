"""FastAPI app that serves salary predictions from the trained pipeline."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from sklearn.pipeline import Pipeline

from starter.ml.model import load_model

MODEL_PATH = Path(__file__).resolve().parent / "model" / "model.pkl"

app = FastAPI(
    title="Census Income Classifier",
    description="Predicts whether a person earns more than $50K a year.",
    version="1.0.0",
)


class CensusRecord(BaseModel):
    """One person's census attributes.

    The JSON keys use the same hyphenated names as the CSV columns
    (e.g. ``marital-status``); the aliases map them onto valid Python
    attribute names.
    """

    age: int = Field(ge=0, examples=[37])
    workclass: str = Field(examples=["Private"])
    fnlgt: int = Field(examples=[284582])
    education: str = Field(examples=["Masters"])
    education_num: int = Field(alias="education-num", examples=[14])
    marital_status: str = Field(
        alias="marital-status", examples=["Married-civ-spouse"]
    )
    occupation: str = Field(examples=["Prof-specialty"])
    relationship: str = Field(examples=["Wife"])
    race: str = Field(examples=["White"])
    sex: str = Field(examples=["Female"])
    capital_gain: int = Field(alias="capital-gain", examples=[0])
    capital_loss: int = Field(alias="capital-loss", examples=[0])
    hours_per_week: int = Field(alias="hours-per-week", examples=[45])
    native_country: str = Field(
        alias="native-country", examples=["United-States"]
    )


class Prediction(BaseModel):
    salary: Literal["<=50K", ">50K"]


@lru_cache(maxsize=1)
def get_model() -> Pipeline:
    """Load the pipeline once and reuse it for every request."""
    return load_model(MODEL_PATH)


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "Welcome to the Census Income Classifier API!"}


@app.post("/predict")
def predict(record: CensusRecord) -> Prediction:
    # by_alias=True restores the original CSV column names the pipeline
    # was trained on.
    features = pd.DataFrame([record.model_dump(by_alias=True)])
    label = get_model().predict(features)[0]
    return Prediction(salary=">50K" if label == 1 else "<=50K")
