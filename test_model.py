"""Unit tests for the functions in starter/ml/model.py."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from starter.ml.model import (
    CAT_FEATURES,
    LABEL,
    compute_model_metrics,
    compute_slice_metrics,
    inference,
    load_model,
    save_model,
    split_features_label,
    train_model,
)

DATA_PATH = Path(__file__).resolve().parent / "data" / "census.csv"
SAMPLE_SIZE = 2000


@pytest.fixture(scope="module")
def census_sample() -> pd.DataFrame:
    """A small, fixed slice of the census data so tests stay fast."""
    data = pd.read_csv(DATA_PATH, skipinitialspace=True)
    return data.sample(n=SAMPLE_SIZE, random_state=0)


@pytest.fixture(scope="module")
def fitted_model(census_sample: pd.DataFrame) -> Pipeline:
    X, y = split_features_label(census_sample)
    return train_model(X, y)


def test_split_features_label_separates_binary_target(census_sample):
    X, y = split_features_label(census_sample)

    assert LABEL not in X.columns
    assert set(CAT_FEATURES).issubset(X.columns)
    assert len(X) == len(y) == SAMPLE_SIZE
    assert set(np.unique(y)) == {0, 1}
    assert y.sum() == (census_sample[LABEL] == ">50K").sum()


def test_train_model_returns_fitted_random_forest_pipeline(fitted_model):
    assert isinstance(fitted_model, Pipeline)
    classifier = fitted_model.named_steps["classifier"]
    assert isinstance(classifier, RandomForestClassifier)
    # Only a fitted estimator exposes the classes it learned.
    assert list(classifier.classes_) == [0, 1]


def test_inference_returns_one_binary_prediction_per_row(
    fitted_model, census_sample
):
    X, _ = split_features_label(census_sample.head(50))

    preds = inference(fitted_model, X)

    assert preds.shape == (50,)
    assert set(np.unique(preds)).issubset({0, 1})


def test_inference_handles_unseen_category(fitted_model, census_sample):
    X, _ = split_features_label(census_sample.head(1))
    X = X.assign(workclass="Never-seen-before")

    preds = inference(fitted_model, X)

    assert preds.shape == (1,)


@pytest.mark.parametrize(
    "y, preds, expected",
    [
        ([1, 0, 1, 0], [1, 0, 1, 0], (1.0, 1.0, 1.0)),
        ([1, 1, 0, 0], [1, 0, 1, 0], (0.5, 0.5, 0.5)),
        ([1, 1, 1, 0], [1, 0, 0, 0], (1.0, 1 / 3, 0.5)),
    ],
)
def test_compute_model_metrics_known_values(y, preds, expected):
    precision, recall, fbeta = compute_model_metrics(
        np.array(y), np.array(preds)
    )

    assert (precision, recall, fbeta) == pytest.approx(expected)


def test_save_and_load_model_round_trip(fitted_model, census_sample, tmp_path):
    X, _ = split_features_label(census_sample.head(20))
    path = tmp_path / "nested" / "model.pkl"

    save_model(fitted_model, path)
    reloaded = load_model(path)

    assert isinstance(reloaded, Pipeline)
    np.testing.assert_array_equal(
        inference(reloaded, X), inference(fitted_model, X)
    )


def test_compute_slice_metrics_one_row_per_value(fitted_model, census_sample):
    slices = compute_slice_metrics(fitted_model, census_sample, "sex")

    assert isinstance(slices, pd.DataFrame)
    assert set(slices["value"]) == set(census_sample["sex"].unique())
    assert slices["n_samples"].sum() == len(census_sample)
    for column in ["precision", "recall", "fbeta"]:
        assert slices[column].between(0, 1).all()
