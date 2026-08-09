import numpy as np
import pandas as pd

#BASIC GOALS:
#Make model to predict target risk variable of injury_occured (Categorical/Percentage)

#LOAD DATA
df = pd.read_csv('/kaggle/input/datasets/anjalibhegam/multimodal-sports-injury-dataset/multimodal_sports_injury_dataset.csv')
df.head()
# print(df.isna().sum())
# print(df.isna().any(axis=1).sum())
# print(df.info)
print(df.describe())

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

#Seperate features and target
#drop injury occured since its target, ids are identifiers so drop, drop gender for potential bias
X = df.drop(['injury_occurred', 'athlete_id', 'session_id', 'gender'], axis=1) #for now exclude gender, we will train another model WITH gender later (can maybe do same w sport type)
y = df["injury_occurred"]

#straify the test data (y) for more spread out, matching representation of dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

#get some info on features
numeric_features = X_train.select_dtypes(include=["number"]).columns
categorical_features = X_train.select_dtypes(include=["object"]).columns
# print(numeric_features)
# print(categorical_features)

# Get names of columns with missing values
cols_with_missing = [col for col in X_train.columns
                     if X_train[col].isnull().any()]


#for learning purposes, prob wont use for model deployment
# from sklearn.impute import SimpleImputer
# #imputation (test mae of options 2 and 3 for imputtion in missing values tutorial, use 2 for now)
# #use median imputation for now to handle outliers
# my_imputer = SimpleImputer(strategy="median")
# imputed_X_train = X_train.copy()
# imputed_X_test = X_test.copy()

# imputed_X_train[numeric_features] = my_imputer.fit_transform(
#     X_train[numeric_features]
# )

# imputed_X_test[numeric_features] = my_imputer.transform(
#     X_test[numeric_features]
# )

# #handle sport type seperately with one-hot-encoding
# from sklearn.preprocessing import OneHotEncoder

# OH_encoder = OneHotEncoder(
#     handle_unknown="ignore",
#     sparse_output=False
# )

# # encode categorical columns
# OH_cols_train = pd.DataFrame(
#     OH_encoder.fit_transform(imputed_X_train[categorical_features]),
#     index=imputed_X_train.index,
#     columns=OH_encoder.get_feature_names_out(categorical_features)
# )

# OH_cols_test = pd.DataFrame(
#     OH_encoder.transform(imputed_X_test[categorical_features]),
#     index=imputed_X_test.index,
#     columns=OH_encoder.get_feature_names_out(categorical_features)
# )

# # drop original categorical columns from the IMPUTED data
# num_X_train = imputed_X_train.drop(categorical_features, axis=1)
# num_X_test = imputed_X_test.drop(categorical_features, axis=1)

# # combine numeric and encoded categorical features
# OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
# OH_X_test = pd.concat([num_X_test, OH_cols_test], axis=1)

# print(OH_X_train.isnull().sum().sum())  # should print 0
# print(OH_X_train.shape)
# print(OH_X_test.shape)

# #last thing, scale to given range on training set with Min Max Scaler
# scaler = MinMaxScaler()

# OH_X_train = scaler.fit_transform(OH_X_train)
# OH_X_test = scaler.transform(OH_X_test)


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
print("TRAIN")
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)


X_train_processed = pd.DataFrame(
    X_train_processed,
    columns=preprocessor.get_feature_names_out()
)

print(X_train_processed.describe())
# print(X_train_processed)
# print(X_train_processed.shape)
print("TEST")

X_test_processed = pd.DataFrame(
    X_test_processed,
    columns=preprocessor.get_feature_names_out()
)

print(X_test_processed.describe())
# print(X_test_processed)
# print(X_test_processed.shape)


#Define the logistic regression model to start
from sklearn.linear_model import LogisticRegression

logistic_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

logistic_model.fit(X_train, y_train)
y_pred = logistic_model.predict(X_test)

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

#Results show that we aren't correctly predicting low risk right now. fix for next time
# Model: Logistic Regression
# Accuracy: 0.678
# Macro F1: 0.44
# Injured Recall: 0.44
# Low Risk Recall: 0.00

#Now we want to improve on the old one, so we will add some tweaks to the model parameters
from sklearn.linear_model import LogisticRegression

logistic_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")) #since we have an imbalance in healthy, low risk, and injured, we balance it with class_weight param
])

logistic_model.fit(X_train, y_train)
y_pred = logistic_model.predict(X_test)

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

#New results look like we are overtuning towards injury risk classes more now...
# Model: Logistic Regression
# Accuracy: 0.46
# Macro F1: 0.46
# Injured Recall: 0.84
# Low Risk Recall: 0.38

#Healthy Recall: 0.40 -> noticably lower now



#We will now try a bagging classifier with Random Forest which averages out performance off multiple trees
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Numeric preprocessing
# Random Forest doesn't need MinMaxScaler
numeric_pipeline_rf = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

# Categorical preprocessing (just in case sport_type has missing values, kinda not needed since it does'nt have any missing but nice to have)
categorical_pipeline_rf = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

# Apply appropriate preprocessing to each column type
rf_preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline_rf, numeric_features),
        ("cat", categorical_pipeline_rf, categorical_features)
    ]
)

# Full preprocessing + Random Forest pipeline
rf_model = Pipeline(steps=[
    ("preprocessor", rf_preprocessor),
    ("classifier", RandomForestClassifier(
        class_weight="balanced",
        random_state=42
    ))
])

# Train
rf_model.fit(X_train, y_train)

# Predict
y_pred = rf_model.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Baseline Random Forest doesn't look great...
# Model: Random Forest (Balanced)
# Accuracy: 0.66
# Macro F1: 0.40
# Injured Recall: 0.31
# Low Risk Recall: 0.00 -> model completely fails to identify Low Risk
# Healthy Recall: 0.97 -> heavily favors predicting Healthy
