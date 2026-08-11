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
