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
from sklearn.preprocessing import MinMaxScaler
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


#Preprocessing PIPELINE for automating of the above steps...
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", MinMaxScaler())
])

categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    ))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features)
    ]
)


#printing/validating processing
print("TRAIN BELOW")
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)


X_train_processed = pd.DataFrame(
    X_train_processed,
    columns=preprocessor.get_feature_names_out()
)

print(X_train_processed.describe())
# print(X_train_processed)
print(X_train_processed.shape)
print("TEST BELOW")

X_test_processed = pd.DataFrame(
    X_test_processed,
    columns=preprocessor.get_feature_names_out()
)

print(X_test_processed.describe())
# print(X_test_processed)
print(X_test_processed.shape)


#New Baseline Logistic Regression for Binary Classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

binary_logistic = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        max_iter=1000
    ))
])

binary_logistic.fit(X_train, y_train)

y_pred = binary_logistic.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Not Injured", "Injured"]
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# Binary Logistic Regression Baseline
# Model: Logistic Regression (No Class Weighting)
# Accuracy: 0.87
# Note: majority-class baseline is ~0.85, so accuracy alone is misleading
# Macro F1: 0.69
# Injured Precision: 0.65
# -> When the model predicts Injured, it is correct 65% of the time
# Injured Recall: 0.35
# -> Model only catches 35% of actual injuries
# Injured F1: 0.46
# Not Injured Recall: 0.97
# -> Model still strongly favors the majority Not Injured class



#Now run with class weight balanced
#New Baseline Logistic Regression for Binary Classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

binary_logistic = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        max_iter=1000, class_weight="balanced"
    ))
])

binary_logistic.fit(X_train, y_train)

y_pred = binary_logistic.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Not Injured", "Injured"]
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# Binary Logistic Regression with Balanced Class Weights

# Accuracy: 0.80
# Accuracy decreased from 0.87, but minority-class performance improved
# Macro F1: 0.72 -> improved from 0.69
# Injured Precision: 0.42
# -> 42% of athletes predicted as Injured were actually Injured
# Injured Recall: 0.87
# -> catches 87% of actual Injured athletes, up significantly from 35%
# Injured F1: 0.57 -> improved from 0.46
# Not Injured Recall: 0.79
# -> decreased from 0.97 due to more false-positive injury predictions
# Conclusion:
# Class weighting greatly improves injury detection at the cost of
# increased false positives and lower overall accuracy.


#Try Random Forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Numeric preprocessing - no scaling needed for Random Forest
numeric_pipeline_rf = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

# Categorical preprocessing
categorical_pipeline_rf = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

rf_preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline_rf, numeric_features),
        ("cat", categorical_pipeline_rf, categorical_features)
    ]
)

# Binary Random Forest
binary_rf = Pipeline(steps=[
    ("preprocessor", rf_preprocessor),
    ("classifier", RandomForestClassifier(
        class_weight="balanced",
        random_state=42
    ))
])

binary_rf.fit(X_train, y_train)

y_pred = binary_rf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Not Injured", "Injured"]
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Binary Random Forest with Balanced Class Weights

# Accuracy: 0.86 -> higher than balanced Logistic Regression (0.80),
# but accuracy is misleading because ~85% of the dataset is Not Injured

# Macro F1: 0.58 -> significantly worse than balanced Logistic Regression (0.72)

# Injured Precision: 0.66
# -> when the model predicts Injured, it is correct 66% of the time

# Injured Recall: 0.14
# -> only catches 14% of actual Injured athletes, which is very poor

# Injured F1: 0.24 -> poor balance between precision and recall

# Not Injured Recall: 0.99
# -> model overwhelmingly favors the Not Injured class

# Conclusion:
# Random Forest achieves higher overall accuracy and injury precision,
# but misses most actual injuries. Balanced Logistic Regression is
# currently much better for detecting the minority Injured class.

# balanced Logistic Regression currently beats a more complex ensemble on minority-class detection, despite RF having higher headline accuracy.


#Try HGB
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# HGB can handle numeric NaNs directly
hgb_preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numeric_features),
        ("cat", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        ), categorical_features)
    ]
)

binary_hgb = Pipeline(steps=[
    ("preprocessor", hgb_preprocessor),
    ("classifier", HistGradientBoostingClassifier(
        random_state=42,
        class_weight="balanced"
    ))
])

# Train
binary_hgb.fit(X_train, y_train)

# Predict
y_pred = binary_hgb.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Not Injured", "Injured"]
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# Binary HistGradientBoosting Baseline (No Class Weighting)

# Accuracy: 0.88 -> highest so far, but remember majority baseline is ~0.85

# Macro F1: 0.72

# Injured Precision: 0.64
# -> 64% of athletes predicted as Injured were actually Injured

# Injured Recall: 0.42
# -> catches 42% of actual Injured athletes

# Injured F1: 0.51
# -> better than Random Forest (0.24), but below balanced Logistic (0.57)

# Not Injured Recall: 0.96
# -> still favors the majority Not Injured class

# Conclusion:
# HGB provides a better precision/recall balance than Random Forest and
# achieves strong overall accuracy, but still misses 58% of actual injuries.
# Balanced Logistic Regression remains better for maximizing injury detection.



# Binary HistGradientBoosting with Balanced Class Weights

# Accuracy: 0.81 -> decreased from 0.88 after weighting

# Macro F1: 0.72 -> remained approximately the same

# Injured Precision: 0.43
# -> decreased from 0.64 due to increased false-positive injury predictions

# Injured Recall: 0.83
# -> major improvement from 0.42; catches 83% of actual injuries

# Injured F1: 0.57
# -> improved from 0.51 due to much stronger injury recall

# Not Injured Recall: 0.81
# -> decreased from 0.96 as the model predicts Injured more aggressively

# Conclusion:
# Class weighting significantly improved injury detection while increasing
# false positives. Balanced HGB provides a much better minority-class
# precision/recall tradeoff than the unweighted model.


#Lastly trying XGBoost
from xgboost import XGBClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# XGBoost doesn't need feature scaling
numeric_pipeline_xgb = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline_xgb = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(
        handle_unknown="ignore"
    ))
])

xgb_preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline_xgb, numeric_features),
        ("cat", categorical_pipeline_xgb, categorical_features)
    ]
)

# Calculate imbalance ratio using TRAINING data only
negative_count = (y_train == 0).sum()
positive_count = (y_train == 1).sum()

scale_pos_weight = negative_count / positive_count

print("scale_pos_weight:", scale_pos_weight)

# Binary XGBoost
binary_xgb = Pipeline(steps=[
    ("preprocessor", xgb_preprocessor),
    ("classifier", XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42
    ))
])

# Train
binary_xgb.fit(X_train, y_train)

# Predict
y_pred = binary_xgb.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Not Injured", "Injured"]
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Binary XGBoost with Class Imbalance Weighting

# scale_pos_weight: 5.66
# -> Injured samples are weighted ~5.66x more heavily because the
#    training data contains ~5.66 Not Injured samples per Injured sample

# Accuracy: 0.79
# -> lower overall accuracy, largely due to more false-positive injury predictions

# Macro F1: 0.71

# Injured Precision: 0.41
# -> 41% of athletes predicted as Injured were actually Injured

# Injured Recall: 0.89
# -> catches 89% of actual injuries, the highest recall of our models so far

# Injured F1: 0.56
# -> strong recall but lower precision keeps F1 around 0.56

# Not Injured Recall: 0.77
# -> decreased because the model is more aggressive about predicting Injured

# Confusion Matrix:
# 414 / 463 actual injuries correctly detected
# 49 / 463 actual injuries missed
# 590 false-positive injury predictions

# Conclusion:
# XGBoost achieves excellent injury recall but generates many false positives.
# Its Injured F1 is similar to Balanced Logistic/HGB, so it does not clearly
# outperform them yet.
