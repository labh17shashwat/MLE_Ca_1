import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve


data = pd.read_csv("hospital_readmissions_30k.csv")


print("First 5 rows:")
print(data.head())


print("\nShape of dataset:")
print(data.shape) 


print("\nMissing values:")
print(data.isnull().sum())


data[["systolic_bp", "diastolic_bp"]] = data[
    "blood_pressure"
].str.split("/", expand=True)

data["systolic_bp"] = data["systolic_bp"].astype(float)

data["diastolic_bp"] = data["diastolic_bp"].astype(float)

data = data.drop("blood_pressure", axis=1)




data["readmitted_30_days"] = data[
    "readmitted_30_days"
].map({
    "Yes": 1,
    "No": 0
})





plt.figure(figsize=(6, 4))

data["readmitted_30_days"].value_counts().plot(
    kind="pie"


)

plt.xlabel("Readmitted within 30 Days")

plt.ylabel("Number of Patients")

plt.title("30-Day Readmission")

plt.xticks(

    [0, 1],
    ["No", "Yes"],
    rotation=0
    
)

plt.show()


data = data.drop("patient_id", axis=1)


X = data.drop("readmitted_30_days", axis=1)

y = data["readmitted_30_days"]


X = pd.get_dummies(
    X,
    columns=[
        "gender",
        "diabetes",
        "hypertension",
        "discharge_destination"
    ],
    drop_first=True
)


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)


model = LogisticRegression(
    penalty="l2",
    C=1.0,
    max_iter=1000
)


model.fit(
    X_train,
    y_train
)


probability = model.predict_proba(
    X_test
)[:, 1]


auc = roc_auc_score(
    y_test,
    probability
)

print("\nROC-AUC:", auc)


fpr, tpr, thresholds = roc_curve(
    y_test,
    probability
)


plt.figure(figsize=(6, 4))

plt.plot(
    fpr,
    tpr,
    label="Logistic Regression"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Guess"
)

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.show()


prediction = (
    probability >= 0.5
).astype(int)


cm = confusion_matrix(
    y_test,
    prediction
)

print("\nConfusion Matrix:")

print(cm)


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        prediction
    )
)