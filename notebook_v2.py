#look at class details
print(df["injury_occurred"].value_counts())
print(df["injury_occurred"].value_counts(normalize=True))

df.groupby("injury_occurred")[numeric_features].mean().T

#investigate the data some more
import matplotlib.pyplot as plt

features = [
    "sleep_quality",
    "recovery_score",
    "training_load",
    "fatigue_index"
]

for feature in features:
    df.boxplot(column=feature, by="injury_occurred")
    plt.title(feature)
    plt.suptitle("")
    plt.xlabel("Injury Class")
    plt.ylabel(feature)
    plt.show()

#Results show that for sleep_quality, fatigue_index, recovery_score, and training_load, we have similar baseline values for class 0 and 1, which causes model issues

#To alleviate we will attempt custom weighting, then slight feature engineering, then maybe changing the problem to binary classification

#Custom Weighting Logistic Regression
from sklearn.linear_model import LogisticRegression

logistic_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight={
            0: 1.0,   # Healthy
            1: 1.5,   # Low Risk
            2: 2.0    # Injured
        }
    ))
])

logistic_model.fit(X_train, y_train)
y_pred = logistic_model.predict(X_test)

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Custom-weight Logistic Regression improved the balance between Healthy and Injured,
# but still completely fails to identify Low Risk
# Model: Logistic Regression (Custom Class Weights)
# Accuracy: 0.66
# Macro F1: 0.45
# Injured Recall: 0.63 -> improved compared to default Logistic Regression
# Low Risk Recall: 0.00 -> still completely fails to identify Low Risk
# Healthy Recall: 0.89 -> remains strong while allowing more Injured predictions


#Trying some feature engineering....
df_fe = df.copy()

# Feature engineering (+1 to avoid dividing by 0...)

df_fe["load_recovery_ratio"] = (
    df_fe["training_load"] / (df_fe["recovery_score"] + 1)
)

df_fe["fatigue_recovery_ratio"] = (
    df_fe["fatigue_index"] / (df_fe["recovery_score"] + 1)
)

df_fe["intensity_duration"] = (
    df_fe["training_intensity"] * df_fe["training_duration"]
)

df_fe["sleep_recovery_score"] = (
    df_fe["sleep_quality"] * df_fe["recovery_score"]
)

engineered_features = [
    "load_recovery_ratio",
    "fatigue_recovery_ratio",
    "intensity_duration",
    "sleep_recovery_score"
]

print(
    df_fe.groupby("injury_occurred")[engineered_features]
    .mean()
    .T
)


import matplotlib.pyplot as plt

for feature in engineered_features:
    df_fe.boxplot(
        column=feature,
        by="injury_occurred"
    )

    plt.title(feature)
    plt.suptitle("")
    plt.xlabel("Injury Class")
    plt.ylabel(feature)
    plt.show()


X_fe = df_fe.drop(
    ["injury_occurred", "athlete_id", "session_id", "gender"],
    axis=1
)

y_fe = df_fe["injury_occurred"]


X_train_fe, X_test_fe, y_train_fe, y_test_fe = train_test_split(
    X_fe,
    y_fe,
    test_size=0.2,
    stratify=y_fe,
    random_state=42
)

numeric_features_fe = X_train_fe.select_dtypes(
    include=["number"]
).columns

categorical_features_fe = X_train_fe.select_dtypes(
    include=["object"]
).columns


numeric_pipeline_fe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", MinMaxScaler())
])

categorical_pipeline_fe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(
        handle_unknown="ignore"
    ))
])

preprocessor_fe = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline_fe, numeric_features_fe),
        ("cat", categorical_pipeline_fe, categorical_features_fe)
    ]
)


logistic_fe_model = Pipeline(steps=[
    ("preprocessor", preprocessor_fe),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ))
])

logistic_fe_model.fit(X_train_fe, y_train_fe)

y_pred_fe = logistic_fe_model.predict(X_test_fe)

print("Accuracy:", accuracy_score(y_test_fe, y_pred_fe))

print("\nClassification Report:")
print(classification_report(y_test_fe, y_pred_fe))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test_fe, y_pred_fe))


# Feature engineering did not meaningfully improve class separation.
# Engineered features still show significant overlap between Healthy and Low Risk.

# Model: Balanced Logistic Regression + Engineered Features

# Accuracy: 0.47

# Macro F1: 0.46

# Injured Recall: 0.86 -> very strong, slightly improved from 0.84

# Low Risk Recall: 0.40 -> slight improvement from 0.38

# Healthy Recall: 0.40 -> unchanged

# Conclusion: engineered interaction features did not meaningfully improve
# the model's ability to distinguish Healthy vs Low Risk.
