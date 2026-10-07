import pandas as pd
import numpy as np

from scipy.sparse import load_npz
from sklearn.svm import LinearSVC
from sklearn.metrics import precision_score, recall_score, f1_score


# Load processed data
X_train = load_npz("processed_data/X_train.npz")
X_val = load_npz("processed_data/X_val.npz")
X_test = load_npz("processed_data/X_test.npz")

y_train = pd.read_csv("processed_data/y_train.csv").squeeze()
y_val = pd.read_csv("processed_data/y_val.csv").squeeze()
y_test = pd.read_csv("processed_data/y_test.csv").squeeze()

# Train SVM
svm_model = LinearSVC()
svm_model.fit(X_train, y_train)


# Get validation scores
val_scores = svm_model.decision_function(X_val)


# Test different thresholds
thresholds = np.arange(-1.0, 0.51, 0.05)

precision_values = []
recall_values = []
f1_values = []

for threshold in thresholds:

    val_pred = (val_scores >= threshold).astype(int)

    precision_values.append(
        precision_score(y_val, val_pred, zero_division=0)
    )

    recall_values.append(
        recall_score(y_val, val_pred, zero_division=0)
    )

    f1_values.append(
        f1_score(y_val, val_pred, zero_division=0)
    )


# Find best threshold
best_index = np.argmax(f1_values)

best_threshold = thresholds[best_index]
best_f1 = f1_values[best_index]

print("Best Threshold:", best_threshold)
print("Best Validation F1:", best_f1)
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))

plt.plot(thresholds, precision_values, marker="o", label="Precision")
plt.plot(thresholds, recall_values, marker="o", label="Recall")
plt.plot(thresholds, f1_values, marker="o", label="F1 Score")

plt.axvline(
    best_threshold,
    linestyle="--",
    label=f"Best Threshold = {best_threshold:.2f}"
)

plt.xlabel("SVM Decision Threshold")
plt.ylabel("Score")
plt.title("Adaptive Threshold Analysis")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Get test scores
test_scores = svm_model.decision_function(X_test)

# Default SVM threshold = 0
default_pred = (test_scores >= 0).astype(int)

# Adaptive threshold
adaptive_pred = (test_scores >= best_threshold).astype(int)


# Confusion matrices
default_cm = confusion_matrix(y_test, default_pred)
adaptive_cm = confusion_matrix(y_test, adaptive_pred)


# Display default SVM confusion matrix
disp = ConfusionMatrixDisplay(
    confusion_matrix=default_cm,
    display_labels=["Ham", "Spam"]
)

disp.plot()
plt.title("Default SVM Confusion Matrix")
plt.show()


# Display adaptive SVM confusion matrix
disp = ConfusionMatrixDisplay(
    confusion_matrix=adaptive_cm,
    display_labels=["Ham", "Spam"]
)

disp.plot()
plt.title("Adaptive SVM Confusion Matrix")
plt.show()
# Model Comparison

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# Train Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
nb_pred = nb_model.predict(X_test)


# Train Logistic Regression
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)


# Store predictions
models = {
    "Naive Bayes": nb_pred,
    "Logistic Regression": lr_pred,
    "Linear SVM": default_pred,
    "Adaptive SVM": adaptive_pred
}


# Calculate metrics
comparison = []

for name, predictions in models.items():

    comparison.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions, zero_division=0),
        "Recall": recall_score(y_test, predictions, zero_division=0),
        "F1 Score": f1_score(y_test, predictions, zero_division=0)
    })
comparison_df = pd.DataFrame(comparison)

print("\n--- Model Comparison ---")
print(comparison_df.round(4).to_string(index=False))
# Model Comparison Chart

import matplotlib.pyplot as plt

metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]

x = np.arange(len(metrics))
width = 0.2
plt.figure(figsize=(10, 6))
for i, row in comparison_df.iterrows():
    values = [
        row["Accuracy"],
        row["Precision"],
        row["Recall"],
        row["F1 Score"]
    ]
    plt.bar(
        x + (i - 1.5) * width,
        values,
        width,
        label=row["Model"]
    )
plt.xticks(x, metrics)
plt.ylabel("Score")
plt.ylim(0, 1.05)
plt.title("Model Performance Comparison")
plt.legend()
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()