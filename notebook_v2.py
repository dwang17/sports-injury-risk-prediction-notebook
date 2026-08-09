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
