#Testing default model
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df_baseline = df.copy()

X_baseline = df_baseline.drop(
    ["injury_occurred", "athlete_id", "session_id",
     "sport_type", "gender"],
    axis=1
)

y_baseline = df_baseline["injury_occurred"]

X_baseline = X_baseline.fillna(X_baseline.median())

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_baseline)

X_train_base, X_test_base, y_train_base, y_test_base = train_test_split(
    X_scaled,
    y_baseline,
    test_size=0.3,
    stratify=y_baseline,
    random_state=42
)

baseline_rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

baseline_rf.fit(X_train_base, y_train_base)

y_pred_base = baseline_rf.predict(X_test_base)

print("Accuracy:", accuracy_score(y_test_base, y_pred_base))

print(classification_report(
    y_test_base,
    y_pred_base,
    target_names=["Healthy", "Low Risk", "Injured"]
))

print(confusion_matrix(y_test_base, y_pred_base))
