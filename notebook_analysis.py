#BASIC GOALS:
#Use a new target variable for binary classification since injury_occured three level classification had indistinguishable differences between classes 0 and 1

#LOAD DATA
df = pd.read_csv('/kaggle/input/datasets/anjalibhegam/multimodal-sports-injury-dataset/multimodal_sports_injury_dataset.csv')
print(df.head())
print(df["playing_surface"].head(20))
# print(df.isna().sum())
# print(df.isna().any(axis=1).sum())
# print(df.info)
# print(df.describe())

#heart_rate, hydration_level, sleep_quality, muscle_activity, gait_speed and training_intensity have like 300-600 missing values
missing = df.isnull().sum()

missing = pd.DataFrame({
    "Missing Values": missing,
    "Percent": missing / len(df) * 100
})

missing.sort_values("Percent", ascending=False)

#training load has a big outlier



#Preprocessing
from sklearn.model_selection import train_test_split

#START
# Create binary target:
# 0 = Not Injured (original classes 0 and 1)
# 1 = Injured (original class 2)

df["injury_binary"] = (df["injury_occurred"] == 2).astype(int)

print(df["injury_binary"].value_counts())
print(df["injury_binary"].value_counts(normalize=True))


# Separate features and target
X = df.drop([
    "injury_occurred",
    "injury_binary",
    "athlete_id",
    "session_id",
    "gender"
], axis=1)

y = df["injury_binary"]

#straify the test data (y) for more spread out, matching representation of dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

#get some info on features
numeric_features = X_train.select_dtypes(include=["number"]).columns.tolist()
categorical_features = X_train.select_dtypes(include=["object"]).columns.tolist()

# playing_surface is numerically encoded, but semantically categorical
numeric_features.remove("playing_surface")
categorical_features.append("playing_surface")

print("Numeric:", numeric_features)
print("Categorical:", categorical_features)

# Get names of columns with missing values
cols_with_missing = [col for col in X_train.columns
                     if X_train[col].isnull().any()]
print("COLS WITH MISSING: ", cols_with_missing)


# ==========================================
# DATA SANITY CHECKS
# ==========================================

# Check train/test dimensions
print("Training shape:", X_train.shape)
print("Test shape:", X_test.shape)

# Make sure stratification maintained approximately the same
# class distribution in the training and test sets
print("\nTraining target distribution:")
print(y_train.value_counts(normalize=True))

print("\nTest target distribution:")
print(y_test.value_counts(normalize=True))

# Check which training features contain missing values
# These will be handled by the final model's preprocessing pipeline
print("\nMissing values in training data:")
missing_train = X_train.isnull().sum()
print(missing_train[missing_train > 0])

# Verify our feature groups
print("\nNumeric features:")
print(numeric_features)

print("\nCategorical features:")
print(categorical_features)


# ==========================================
# FINAL MODEL SETUP
# ==========================================

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# HGB handles numeric NaNs natively, so numerical columns can pass through
numeric_pipeline_hgb = "passthrough"

# Handle categorical features separately
categorical_pipeline_hgb = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    ))
])

# Preprocessing specifically for the final HGB model
hgb_preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline_hgb, numeric_features),
        ("cat", categorical_pipeline_hgb, categorical_features)
    ]
)

# Tuned parameters found using RandomizedSearchCV
best_hgb = Pipeline(steps=[
    ("preprocessor", hgb_preprocessor),
    ("classifier", HistGradientBoostingClassifier(
        class_weight="balanced",
        min_samples_leaf=20,
        max_leaf_nodes=63,
        max_iter=200,
        learning_rate=0.03,
        l2_regularization=0.1,
        random_state=42
    ))
])

# Decision threshold via threshold tuning selected by maximizing injured-class F1
# using out-of-fold predictions on the training set
BEST_THRESHOLD = 0.51

# Train final model using all training data
best_hgb.fit(X_train, y_train)

# Predict injury probabilities
y_test_prob = best_hgb.predict_proba(X_test)[:, 1]

# Apply threshold selected using out-of-fold training predictions
y_test_pred = (y_test_prob >= BEST_THRESHOLD).astype(int)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_test_pred))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_test_pred,
    target_names=["Not Injured", "Injured"]
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_test_pred))


# ============================================================
# SHAP MODEL INTERPRETABILITY
# ============================================================

import shap
import pandas as pd
import matplotlib.pyplot as plt

# Extract fitted preprocessing/model components
preprocessor = best_hgb.named_steps["preprocessor"]
hgb_model = best_hgb.named_steps["classifier"]

# Transform data exactly as the classifier sees it
X_train_transformed = preprocessor.transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

# Get transformed feature names
feature_names = preprocessor.get_feature_names_out()

# Verify alignment
assert X_train_transformed.shape[1] == len(feature_names)
assert X_test_transformed.shape[1] == len(feature_names)

# Create labeled transformed DataFrames
X_train_shap = pd.DataFrame(
    X_train_transformed,
    columns=feature_names,
    index=X_train.index
)

X_test_shap = pd.DataFrame(
    X_test_transformed,
    columns=feature_names,
    index=X_test.index
)

# Create tree-based SHAP explainer
explainer = shap.TreeExplainer(hgb_model)

# Calculate SHAP values for held-out test data
shap_values = explainer(X_test_shap)

print("SHAP values shape:", shap_values.shape)

# Global feature importance
shap.plots.bar(
    shap_values,
    max_display=15
)

# Feature value + direction of effect
shap.plots.beeswarm(
    shap_values,
    max_display=15
)

#some stats
print(df.groupby("injury_binary")["recovery_score"].describe())
print(df.groupby("injury_binary")["recovery_score"].mean())
print(df[["recovery_score", "injury_binary"]].corr())


# ============================================================
# SHAP LOCAL INTERPRETABILITY — TP / FP / FN
# ============================================================

import numpy as np
import shap
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# 1. Get final model predictions using our tuned threshold
# ------------------------------------------------------------

# Probability of class 1 = Injured
y_proba = best_hgb.predict_proba(X_test)[:, 1]

# Apply our selected threshold
y_pred = (y_proba >= BEST_THRESHOLD).astype(int)

# Convert y_test to numpy so positional indexing is easy
y_true = y_test.to_numpy()


# ------------------------------------------------------------
# 2. Find TP, FP, and FN examples
# ------------------------------------------------------------

# True Positive:
# Actually injured AND predicted injured
tp_indices = np.where(
    (y_true == 1) & (y_pred == 1)
)[0]

# False Positive:
# Actually not injured BUT predicted injured
fp_indices = np.where(
    (y_true == 0) & (y_pred == 1)
)[0]

# False Negative:
# Actually injured BUT predicted not injured
fn_indices = np.where(
    (y_true == 1) & (y_pred == 0)
)[0]


print("Prediction counts")
print("-----------------")
print("True Positives:", len(tp_indices))
print("False Positives:", len(fp_indices))
print("False Negatives:", len(fn_indices))


# ------------------------------------------------------------
# 3. Select one example from each category
# ------------------------------------------------------------

tp_idx = tp_indices[0]
fp_idx = fp_indices[0]
fn_idx = fn_indices[0]


# ------------------------------------------------------------
# 4. Display prediction information
# ------------------------------------------------------------

print("\nSelected examples")
print("-----------------")

print(
    f"True Positive  | "
    f"Actual: {y_true[tp_idx]} | "
    f"Predicted: {y_pred[tp_idx]} | "
    f"Injury probability: {y_proba[tp_idx]:.3f}"
)

print(
    f"False Positive | "
    f"Actual: {y_true[fp_idx]} | "
    f"Predicted: {y_pred[fp_idx]} | "
    f"Injury probability: {y_proba[fp_idx]:.3f}"
)

print(
    f"False Negative | "
    f"Actual: {y_true[fn_idx]} | "
    f"Predicted: {y_pred[fn_idx]} | "
    f"Injury probability: {y_proba[fn_idx]:.3f}"
)


# ------------------------------------------------------------
# 5. SHAP WATERFALL — TRUE POSITIVE
# ------------------------------------------------------------

print("\nTRUE POSITIVE")
print(
    f"Actual = Injured | "
    f"Predicted = Injured | "
    f"Probability = {y_proba[tp_idx]:.3f}"
)

shap.plots.waterfall(
    shap_values[tp_idx],
    max_display=15
)

plt.show()


# ------------------------------------------------------------
# 6. SHAP WATERFALL — FALSE POSITIVE
# ------------------------------------------------------------

print("\nFALSE POSITIVE")
print(
    f"Actual = Not Injured | "
    f"Predicted = Injured | "
    f"Probability = {y_proba[fp_idx]:.3f}"
)

shap.plots.waterfall(
    shap_values[fp_idx],
    max_display=15
)

plt.show()


# ------------------------------------------------------------
# 7. SHAP WATERFALL — FALSE NEGATIVE
# ------------------------------------------------------------

print("\nFALSE NEGATIVE")
print(
    f"Actual = Injured | "
    f"Predicted = Not Injured | "
    f"Probability = {y_proba[fn_idx]:.3f}"
)

shap.plots.waterfall(
    shap_values[fn_idx],
    max_display=15
)

plt.show()

# TRUE POSITIVE
print("\nTRUE POSITIVE")
print("----------------")

for feature, value, shap_val in zip(
    X_test_shap.columns,
    X_test_shap.iloc[tp_idx],
    shap_values[tp_idx].values
):
    print(f"{feature}: value={value:.3f}, SHAP={shap_val:+.3f}")


# FALSE POSITIVE
print("\nFALSE POSITIVE")
print("----------------")

for feature, value, shap_val in zip(
    X_test_shap.columns,
    X_test_shap.iloc[fp_idx],
    shap_values[fp_idx].values
):
    print(f"{feature}: value={value:.3f}, SHAP={shap_val:+.3f}")


# FALSE NEGATIVE
print("\nFALSE NEGATIVE")
print("----------------")

for feature, value, shap_val in zip(
    X_test_shap.columns,
    X_test_shap.iloc[fn_idx],
    shap_values[fn_idx].values
):
    print(f"{feature}: value={value:.3f}, SHAP={shap_val:+.3f}")




# ============================================================
# PROBABILITY CALIBRATION — BASELINE EVALUATION
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss


# ------------------------------------------------------------
# 1. Get current UNCALIBRATED probabilities
# ------------------------------------------------------------

# Probability that each test example belongs to:
# class 1 = Injured
y_proba_uncalibrated = best_hgb.predict_proba(X_test)[:, 1]


# ------------------------------------------------------------
# 2. Calculate Brier Score
# ------------------------------------------------------------
# Brier score measures how close predicted probabilities
# are to the actual binary outcomes.
#
# Perfect = 0
# Lower = better
#
# Example:
# prediction = 0.90, actual = 1 -> small error
# prediction = 0.90, actual = 0 -> large error

brier_uncalibrated = brier_score_loss(
    y_test,
    y_proba_uncalibrated
)

print("Uncalibrated Brier Score:", brier_uncalibrated)
# Uncalibrated HGB Brier: 0.11194

# ------------------------------------------------------------
# 3. Create calibration curve
# ------------------------------------------------------------
# We divide predictions into probability bins.
#
# For each bin:
#
# mean_predicted_value =
#     average probability predicted by the model
#
# fraction_of_positives =
#     actual fraction of injured examples in that bin
#
# Example:
#
# Average prediction = 0.70
# Actual injury rate = 0.50
#
# -> model is overconfident

fraction_of_positives, mean_predicted_value = calibration_curve(
    y_test,
    y_proba_uncalibrated,
    n_bins=10,
    strategy="quantile"
)


# ------------------------------------------------------------
# 4. Print calibration values
# ------------------------------------------------------------

print("\nCalibration Results")
print("-------------------")

for predicted, actual in zip(
    mean_predicted_value,
    fraction_of_positives
):
    print(
        f"Predicted probability: {predicted:.3f} | "
        f"Actual injury rate: {actual:.3f}"
    )


# ------------------------------------------------------------
# 5. Plot calibration curve
# ------------------------------------------------------------

plt.figure(figsize=(8, 6))

# Perfect calibration line
plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Perfect calibration"
)

# Our HGB model
plt.plot(
    mean_predicted_value,
    fraction_of_positives,
    marker="o",
    label="HGB (uncalibrated)"
)

plt.xlabel("Mean Predicted Injury Probability")
plt.ylabel("Actual Fraction Injured")
plt.title("Calibration Curve — HistGradientBoosting")

plt.legend()
plt.grid(alpha=0.3)

plt.show()


# ------------------------------------------------------------
# 6. Inspect probability distribution
# ------------------------------------------------------------

plt.figure(figsize=(8, 6))

plt.hist(
    y_proba_uncalibrated,
    bins=20,
    edgecolor="black"
)

plt.xlabel("Predicted Injury Probability")
plt.ylabel("Number of Test Examples")
plt.title("Distribution of Predicted Injury Probabilities")

plt.show()



# ============================================================
# PROBABILITY CALIBRATION — SIGMOID VS ISOTONIC
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score


# ------------------------------------------------------------
# 1. Current uncalibrated model probabilities
# ------------------------------------------------------------

proba_uncalibrated = best_hgb.predict_proba(X_test)[:, 1]


# ------------------------------------------------------------
# 2. Build calibrated versions
# ------------------------------------------------------------
# We use cv=5 so calibration is learned using cross-validation
# on the TRAINING SET only.
#
# This avoids fitting the calibrator directly on X_test/y_test.

sigmoid_model = CalibratedClassifierCV(
    estimator=best_hgb,
    method="sigmoid",
    cv=5
)

isotonic_model = CalibratedClassifierCV(
    estimator=best_hgb,
    method="isotonic",
    cv=5
)


# ------------------------------------------------------------
# 3. Fit calibrated models on TRAINING DATA ONLY
# ------------------------------------------------------------

sigmoid_model.fit(X_train, y_train)

isotonic_model.fit(X_train, y_train)


# ------------------------------------------------------------
# 4. Get calibrated probabilities on HELD-OUT TEST SET
# ------------------------------------------------------------

proba_sigmoid = sigmoid_model.predict_proba(X_test)[:, 1]

proba_isotonic = isotonic_model.predict_proba(X_test)[:, 1]


# ------------------------------------------------------------
# 5. Compare Brier scores
# ------------------------------------------------------------
# Lower Brier score = better probability calibration

brier_uncalibrated = brier_score_loss(
    y_test,
    proba_uncalibrated
)

brier_sigmoid = brier_score_loss(
    y_test,
    proba_sigmoid
)

brier_isotonic = brier_score_loss(
    y_test,
    proba_isotonic
)


# ------------------------------------------------------------
# 6. Compare ROC-AUC
# ------------------------------------------------------------
# ROC-AUC should usually remain fairly similar because
# calibration mainly changes probability scaling rather than
# the overall ranking of examples.

auc_uncalibrated = roc_auc_score(
    y_test,
    proba_uncalibrated
)

auc_sigmoid = roc_auc_score(
    y_test,
    proba_sigmoid
)

auc_isotonic = roc_auc_score(
    y_test,
    proba_isotonic
)


# ------------------------------------------------------------
# 7. Print comparison table
# ------------------------------------------------------------

results = pd.DataFrame({
    "Model": [
        "Uncalibrated HGB",
        "Sigmoid calibrated HGB",
        "Isotonic calibrated HGB"
    ],
    "Brier Score": [
        brier_uncalibrated,
        brier_sigmoid,
        brier_isotonic
    ],
    "ROC-AUC": [
        auc_uncalibrated,
        auc_sigmoid,
        auc_isotonic
    ]
})

print(results)


# ------------------------------------------------------------
# 8. Build calibration curves
# ------------------------------------------------------------

frac_uncal, mean_uncal = calibration_curve(
    y_test,
    proba_uncalibrated,
    n_bins=10,
    strategy="quantile"
)

frac_sigmoid, mean_sigmoid = calibration_curve(
    y_test,
    proba_sigmoid,
    n_bins=10,
    strategy="quantile"
)

frac_isotonic, mean_isotonic = calibration_curve(
    y_test,
    proba_isotonic,
    n_bins=10,
    strategy="quantile"
)


# ------------------------------------------------------------
# 9. Plot all calibration curves together
# ------------------------------------------------------------

plt.figure(figsize=(9, 7))

# Perfect calibration
plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Perfect calibration"
)

# Uncalibrated HGB
plt.plot(
    mean_uncal,
    frac_uncal,
    marker="o",
    label="Uncalibrated HGB"
)

# Sigmoid calibrated
plt.plot(
    mean_sigmoid,
    frac_sigmoid,
    marker="o",
    label="Sigmoid"
)

# Isotonic calibrated
plt.plot(
    mean_isotonic,
    frac_isotonic,
    marker="o",
    label="Isotonic"
)

plt.xlabel("Mean Predicted Injury Probability")
plt.ylabel("Actual Fraction Injured")
plt.title("Calibration Comparison")

plt.legend()
plt.grid(alpha=0.3)

plt.show()


# ------------------------------------------------------------
# 10. Print calibration-bin values
# ------------------------------------------------------------

print("\nSIGMOID CALIBRATION")
print("-------------------")

for predicted, actual in zip(
    mean_sigmoid,
    frac_sigmoid
):
    print(
        f"Predicted: {predicted:.3f} | "
        f"Actual injury rate: {actual:.3f}"
    )


print("\nISOTONIC CALIBRATION")
print("--------------------")

for predicted, actual in zip(
    mean_isotonic,
    frac_isotonic
):
    print(
        f"Predicted: {predicted:.3f} | "
        f"Actual injury rate: {actual:.3f}"
    )




# ============================================================
# CALIBRATED MODEL — THRESHOLD TUNING
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ------------------------------------------------------------
# 1. Cross-validation setup
# ------------------------------------------------------------
# Same CV strategy we've been using.
#
# IMPORTANT:
# Threshold selection happens using X_train/y_train ONLY.
# The held-out test set remains untouched.

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# ------------------------------------------------------------
# 2. Generate out-of-fold CALIBRATED probabilities
# ------------------------------------------------------------
# sigmoid_model is our:
#
# CalibratedClassifierCV(
#     estimator=best_hgb,
#     method="sigmoid",
#     cv=5
# )
#
# cross_val_predict creates predictions for each training
# observation from a model that did NOT train on that
# observation.
#
# This gives us probabilities suitable for threshold selection.

oof_calibrated_proba = cross_val_predict(
    sigmoid_model,
    X_train,
    y_train,
    cv=cv,
    method="predict_proba",
    n_jobs=-1
)[:, 1]


print("Number of OOF predictions:", len(oof_calibrated_proba))
print("Training examples:", len(y_train))


# ------------------------------------------------------------
# 3. Search candidate thresholds
# ------------------------------------------------------------
# Because calibration moved probabilities downward,
# the optimal threshold may now be substantially below 0.50.
#
# We'll search a broad range.

thresholds = np.arange(
    0.05,
    0.61,
    0.01
)

results = []

for threshold in thresholds:

    y_pred_threshold = (
        oof_calibrated_proba >= threshold
    ).astype(int)

    precision = precision_score(
        y_train,
        y_pred_threshold,
        zero_division=0
    )

    recall = recall_score(
        y_train,
        y_pred_threshold,
        zero_division=0
    )

    f1 = f1_score(
        y_train,
        y_pred_threshold,
        zero_division=0
    )

    results.append({
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1
    })


threshold_results = pd.DataFrame(results)


# ------------------------------------------------------------
# 4. Find threshold with best OOF F1
# ------------------------------------------------------------

best_row = threshold_results.loc[
    threshold_results["f1"].idxmax()
]

CALIBRATED_THRESHOLD = best_row["threshold"]


print("\nBEST CALIBRATED THRESHOLD")
print("-------------------------")

print(
    f"Threshold: {CALIBRATED_THRESHOLD:.2f}"
)

print(
    f"OOF Precision: {best_row['precision']:.4f}"
)

print(
    f"OOF Recall: {best_row['recall']:.4f}"
)

print(
    f"OOF F1: {best_row['f1']:.4f}"
)


# ------------------------------------------------------------
# 5. Show thresholds around the optimum
# ------------------------------------------------------------

best_index = threshold_results["f1"].idxmax()

start = max(0, best_index - 5)
end = min(
    len(threshold_results),
    best_index + 6
)

print("\nThresholds around optimum:")
print(
    threshold_results.iloc[start:end].to_string(
        index=False
    )
)


# ------------------------------------------------------------
# 6. Plot Precision / Recall / F1 vs threshold
# ------------------------------------------------------------

plt.figure(figsize=(9, 6))

plt.plot(
    threshold_results["threshold"],
    threshold_results["precision"],
    label="Precision"
)

plt.plot(
    threshold_results["threshold"],
    threshold_results["recall"],
    label="Recall"
)

plt.plot(
    threshold_results["threshold"],
    threshold_results["f1"],
    label="F1"
)

plt.axvline(
    CALIBRATED_THRESHOLD,
    linestyle="--",
    label=f"Best threshold = {CALIBRATED_THRESHOLD:.2f}"
)

plt.xlabel("Decision Threshold")
plt.ylabel("Score")
plt.title(
    "Threshold Tuning — Sigmoid Calibrated HGB"
)

plt.legend()
plt.grid(alpha=0.3)

plt.show()


# ============================================================
# FINAL HELD-OUT TEST EVALUATION
# ============================================================


# ------------------------------------------------------------
# 7. Fit final calibrated model
# ------------------------------------------------------------
# We already fitted sigmoid_model earlier, but fitting here
# makes this section self-contained and ensures it is trained
# on all of X_train before final evaluation.

sigmoid_model.fit(
    X_train,
    y_train
)


# ------------------------------------------------------------
# 8. Generate calibrated TEST probabilities
# ------------------------------------------------------------

test_calibrated_proba = (
    sigmoid_model.predict_proba(X_test)[:, 1]
)


# ------------------------------------------------------------
# 9. Apply LOCKED threshold
# ------------------------------------------------------------

y_pred_calibrated = (
    test_calibrated_proba >= CALIBRATED_THRESHOLD
).astype(int)


# ------------------------------------------------------------
# 10. Final test metrics
# ------------------------------------------------------------

test_accuracy = accuracy_score(
    y_test,
    y_pred_calibrated
)

test_precision = precision_score(
    y_test,
    y_pred_calibrated
)

test_recall = recall_score(
    y_test,
    y_pred_calibrated
)

test_f1 = f1_score(
    y_test,
    y_pred_calibrated
)


print("\nFINAL CALIBRATED TEST RESULTS")
print("-----------------------------")

print(
    f"Threshold: {CALIBRATED_THRESHOLD:.2f}"
)

print(
    f"Accuracy:  {test_accuracy:.4f}"
)

print(
    f"Precision: {test_precision:.4f}"
)

print(
    f"Recall:    {test_recall:.4f}"
)

print(
    f"F1:        {test_f1:.4f}"
)


# ------------------------------------------------------------
# 11. Classification report
# ------------------------------------------------------------

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred_calibrated,
        target_names=[
            "Not Injured",
            "Injured"
        ]
    )
)


# ------------------------------------------------------------
# 12. Confusion matrix
# ------------------------------------------------------------

cm = confusion_matrix(
    y_test,
    y_pred_calibrated
)

print("Confusion Matrix:")
print(cm)




import joblib

# ============================================================
# SAVE FINAL DEPLOYMENT ARTIFACT
# ============================================================

deployment_artifact = {
    "model": sigmoid_model,
    "threshold": float(CALIBRATED_THRESHOLD)
}

joblib.dump(
    deployment_artifact,
    "injury_risk_model.joblib"
)

print("Model saved!")


# ============================================================
# RELOAD MODEL
# ============================================================

loaded_artifact = joblib.load(
    "injury_risk_model.joblib"
)

loaded_model = loaded_artifact["model"]
loaded_threshold = loaded_artifact["threshold"]

print("Loaded threshold:", loaded_threshold)


# ============================================================
# TEST RAW INFERENCE
# ============================================================

# Grab one RAW row from X_test.
# Do NOT manually preprocess it.
example = X_test.iloc[[0]]

probability = loaded_model.predict_proba(example)[0, 1]

prediction = int(
    probability >= loaded_threshold
)

print(f"Injury probability: {probability:.4f}")
print(f"Threshold: {loaded_threshold:.2f}")
print(f"Elevated injury risk: {bool(prediction)}")
print(f"Actual class: {y_test.iloc[0]}")


original_probability = sigmoid_model.predict_proba(
    X_test.iloc[[0]]
)[0, 1]

loaded_probability = loaded_model.predict_proba(
    X_test.iloc[[0]]
)[0, 1]

print("Original:", original_probability)
print("Reloaded:", loaded_probability)

assert np.isclose(
    original_probability,
    loaded_probability
)

print("Serialization test passed!")
