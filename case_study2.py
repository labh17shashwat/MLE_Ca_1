import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve

from imblearn.over_sampling import SMOTE

from xgboost import XGBClassifier


data = pd.read_csv("credit_card_fraud_dataset.csv")


print("First 5 rows:")
print(data.head())


print("\nShape of dataset:")
print(data.shape)


print("\nMissing values:")
print(data.isnull().sum())


print("\nFraud distribution:")
print(data["IsFraud"].value_counts())


data["TransactionDate"] = pd.to_datetime(
    data["TransactionDate"]
)


data["Year"] = data[
    "TransactionDate"
].dt.year

data["Month"] = data[
    "TransactionDate"
].dt.month

data["Day"] = data[
    "TransactionDate"
].dt.day

data["Hour"] = data[
    "TransactionDate"
].dt.hour


data = data.drop(
    ["TransactionID", "TransactionDate"],
    axis=1
)


data = pd.get_dummies(
    data,
    columns=[
        "TransactionType",
        "Location"
    ],
    drop_first=True
)


X = data.drop(
    "IsFraud",
    axis=1
)

y = data["IsFraud"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTraining data before SMOTE:")
print(y_train.value_counts())


smote = SMOTE(
    random_state=42
)


X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)


print("\nTraining data after SMOTE:")
print(y_train_smote.value_counts())


xgb_model = XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42,
    eval_metric="logloss"
)


xgb_model.fit(
    X_train_smote,
    y_train_smote
)


xgb_probability = xgb_model.predict_proba(
    X_test
)[:, 1]


xgb_auc = roc_auc_score(
    y_test,
    xgb_probability
)


print("\nXGBoost ROC-AUC:")
print(xgb_auc)


thresholds = [0.20, 0.30, 0.40, 0.50]


for threshold in thresholds:

    xgb_prediction = (
        xgb_probability >= threshold
    ).astype(int)

    cm = confusion_matrix(
        y_test,
        xgb_prediction
    )

    print(
        "\nThreshold:",
        threshold
    )

    print("Confusion Matrix:")

    print(cm)


threshold = 0.30


xgb_prediction = (
    xgb_probability >= threshold
).astype(int)


print("\nFinal XGBoost Confusion Matrix:")

print(
    confusion_matrix(
        y_test,
        xgb_prediction
    )
)


print("\nFinal XGBoost Classification Report:")

print(
    classification_report(
        y_test,
        xgb_prediction
    )
)


fpr_xgb, tpr_xgb, thresholds_xgb = roc_curve(
    y_test,
    xgb_probability
)


plt.figure(figsize=(6, 4))

plt.plot(
    fpr_xgb,
    tpr_xgb,
    label="XGBoost"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Guess"
)

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("XGBoost ROC Curve")

plt.legend()

plt.show()


importance = pd.Series(
    xgb_model.feature_importances_,
    index=X.columns
)


importance = importance.sort_values(
    ascending=False
)


print("\nFeature Importance:")

print(importance)


plt.figure(figsize=(8, 5))

importance.head(10).sort_values().plot(
    kind="barh"
)

plt.xlabel("Importance Score")

plt.ylabel("Features")

plt.title("Top 10 Important Features")

plt.show()


scaler = StandardScaler()


X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)


svm_model = SVC(
    kernel="rbf",
    probability=True,
    random_state=42
)


svm_model.fit(
    X_train_scaled,
    y_train
)


svm_probability = svm_model.predict_proba(
    X_test_scaled
)[:, 1]


svm_auc = roc_auc_score(
    y_test,
    svm_probability
)


print("\nSVM ROC-AUC:")
print(svm_auc)


svm_prediction = (
    svm_probability >= 0.50
).astype(int)


print("\nSVM Confusion Matrix:")

print(
    confusion_matrix(
        y_test,
        svm_prediction
    )
)


print("\nSVM Classification Report:")

print(
    classification_report(
        y_test,
        svm_prediction
    )
)


fpr_svm, tpr_svm, thresholds_svm = roc_curve(
    y_test,
    svm_probability
)


plt.figure(figsize=(6, 4))

plt.plot(
    fpr_xgb,
    tpr_xgb,
    label="XGBoost"
)

plt.plot(
    fpr_svm,
    tpr_svm,
    label="SVM"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Guess"
)

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("XGBoost vs SVM")

plt.legend()

plt.show()


print("\nModel Comparison:")

print(
    "XGBoost ROC-AUC:",
    xgb_auc
)

print(
    "SVM ROC-AUC:",
    svm_auc
)