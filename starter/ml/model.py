from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

CAT_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
LABEL = "salary"
POSITIVE_LABEL = ">50K"


def split_features_label(data, label=LABEL):
    """
    Splits a raw dataframe into features and a binary label.

    Inputs
    ------
    data : pd.DataFrame
        Raw data including the label column.
    label : str
        Name of the label column.
    Returns
    -------
    X : pd.DataFrame
        Features, with the label column dropped.
    y : np.ndarray
        1 where the label is ">50K", 0 otherwise.
    """
    X = data.drop(columns=[label])
    y = (data[label] == POSITIVE_LABEL).astype(int).to_numpy()
    return X, y


def train_model(X_train, y_train, categorical_features=CAT_FEATURES):
    """
    Trains a machine learning pipeline and returns it.

    The pipeline one-hot encodes the categorical features and passes the
    continuous ones through unchanged, then fits a random forest. Because
    the encoder lives inside the pipeline, the single returned object is
    all that needs to be saved for inference.

    Inputs
    ------
    X_train : pd.DataFrame
        Raw training features (no label column).
    y_train : np.ndarray
        Binary labels.
    categorical_features : list[str]
        Columns of `X_train` to one-hot encode.
    Returns
    -------
    model : sklearn.pipeline.Pipeline
        Trained preprocessing + RandomForestClassifier pipeline.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                list(categorical_features),
            )
        ],
        remainder="passthrough",
    )
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=100, random_state=42, n_jobs=-1
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and
    F1.

    Inputs
    ------
    y : np.ndarray
        Known labels, binarized.
    preds : np.ndarray
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : sklearn.pipeline.Pipeline
        Trained pipeline from `train_model`.
    X : pd.DataFrame
        Raw features used for prediction (same columns as training).
    Returns
    -------
    preds : np.ndarray
        Predictions from the model (1 means ">50K").
    """
    return model.predict(X)


def save_model(model, path):
    """ Save a trained pipeline (encoder + classifier) to disk.

    Inputs
    ------
    model : sklearn.pipeline.Pipeline
        Trained pipeline from `train_model`.
    path : str or pathlib.Path
        Destination file. Parent folders are created if needed.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path):
    """ Load a pipeline saved with `save_model`.

    Inputs
    ------
    path : str or pathlib.Path
        File written by `save_model`.
    Returns
    -------
    model : sklearn.pipeline.Pipeline
        The trained pipeline, ready for `inference`.
    """
    return joblib.load(path)


def compute_slice_metrics(model, data, feature):
    """ Compute metrics for each value of one feature, holding it fixed.

    For example, with feature="education" there is one row of metrics for
    "Bachelors", one for "HS-grad", and so on.

    Inputs
    ------
    model : sklearn.pipeline.Pipeline
        Trained pipeline from `train_model`.
    data : pd.DataFrame
        Raw data including the label column.
    feature : str
        Name of the column to slice on.
    Returns
    -------
    slices : pd.DataFrame
        One row per value of `feature`, with columns feature, value,
        n_samples, precision, recall and fbeta.
    """
    rows = []
    for value, group in data.groupby(feature):
        X, y = split_features_label(group)
        precision, recall, fbeta = compute_model_metrics(
            y, inference(model, X)
        )
        rows.append(
            {
                "feature": feature,
                "value": value,
                "n_samples": len(group),
                "precision": precision,
                "recall": recall,
                "fbeta": fbeta,
            }
        )
    return pd.DataFrame(rows)
