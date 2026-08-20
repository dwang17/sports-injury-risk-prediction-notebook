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
