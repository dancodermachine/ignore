# Script to train machine learning model.

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from starter.ml.model import (
    CAT_FEATURES,
    LABEL,
    compute_model_metrics,
    compute_slice_metrics,
    inference,
    save_model,
    split_features_label,
    train_model,
)

# Load the data from the project root.
project_root = Path(__file__).resolve().parents[1]
data_path = project_root / "data" / "census.csv"
model_path = project_root / "model" / "model.pkl"
slice_output_path = project_root / "slice_output.txt"
data = pd.read_csv(data_path, skipinitialspace=True)

# Optional: use K-fold cross validation instead of a train-test split.
train, test = train_test_split(
    data, test_size=0.20, random_state=42, stratify=data[LABEL]
)

X_train, y_train = split_features_label(train)
X_test, y_test = split_features_label(test)

# Train the pipeline (encoding + model) and evaluate it.
model = train_model(X_train, y_train)

preds = inference(model, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, preds)
print(
    f"Test metrics -> precision: {precision:.4f}, "
    f"recall: {recall:.4f}, F1: {fbeta:.4f}"
)

# Save the single pipeline artifact (encoder + classifier).
save_model(model, model_path)
print(f"Saved pipeline to {model_path}")

# Metrics on the test set for every value of every categorical feature.
with open(slice_output_path, "w") as f:
    for feature in CAT_FEATURES:
        slices = compute_slice_metrics(model, test, feature)
        f.write(f"=== {feature} ===\n")
        f.write(slices.drop(columns="feature").to_string(
            index=False, float_format="{:.4f}".format
        ))
        f.write("\n\n")
print(f"Wrote slice metrics to {slice_output_path}")
