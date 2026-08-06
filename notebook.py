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

#imputation (test mae of options 2 and 3 for imputtion in missing values tutorial)
X_train_plus = X_train.copy()
y_train_plus = y_train.copy()
