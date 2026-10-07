import pandas as pd
from scipy.sparse import load_npz
# Load processed data
X_train = load_npz("processed_data/X_train.npz")
X_val = load_npz("processed_data/X_val.npz")
X_test = load_npz("processed_data/X_test.npz")
y_train = pd.read_csv("processed_data/y_train.csv").squeeze()
y_val = pd.read_csv("processed_data/y_val.csv").squeeze()
y_test = pd.read_csv("processed_data/y_test.csv").squeeze()
print("Training data:", X_train.shape)
print("Validation data:", X_val.shape)
print("Testing data:", X_test.shape)
# Multinomial Naive Bayes
from sklearn.naive_bayes import MultinomialNB
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
nb_pred = nb_model.predict(X_test)
#print("\n--- Multinomial Naive Bayes ---")
#print("Predictions:", nb_pred[:10])
from sklearn.metrics import accuracy_score, classification_report

#print("\n--- Naive Bayes Evaluation ---")

#print("Accuracy:", accuracy_score(y_test, nb_pred))

#print("\nClassification Report:")
"""
print(classification_report(
    y_test,
    nb_pred,
    target_names=["Ham", "Spam"]
))
"""
# Logistic Regression
from sklearn.linear_model import LogisticRegression
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
#print("\n--- Logistic Regression ---")
#print("Predictions:", lr_pred[:10])
#print("\n--- Logistic Regression Evaluation ---")

#print("Accuracy:", accuracy_score(y_test, lr_pred))

#print("\nClassification Report:")
"""
print(classification_report(
    y_test,
    lr_pred,
    target_names=["Ham", "Spam"]
))
"""
from sklearn.svm import LinearSVC
# Linear SVM

from sklearn.svm import LinearSVC

svm_model = LinearSVC()

svm_model.fit(X_train, y_train)
import joblib

joblib.dump(svm_model, "processed_data/adaptive_svm.pkl")

# Get decision scores for validation data
val_scores = svm_model.decision_function(X_val)

print("\n--- SVM Validation ---")
print("Validation scores:", val_scores[:10])
#print("\n--- Linear SVM ---")
#print("Predictions:", svm_pred[:10])
#print("\n--- Linear SVM Evaluation ---")

#print("Accuracy:", accuracy_score(y_test, svm_pred))

#print("\nClassification Report:")
# Find Best Threshold

from sklearn.metrics import f1_score
import numpy as np

thresholds = np.arange(-1.0, 0.51, 0.05)

best_threshold = 0
best_f1 = 0

for threshold in thresholds:

    val_pred = (val_scores >= threshold).astype(int)

    f1 = f1_score(y_val, val_pred)

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print("\n--- Best Threshold ---")
print("Threshold:", best_threshold)
print("Validation F1:", best_f1)
# Final Evaluation on Test Data

test_scores = svm_model.decision_function(X_test)

final_pred = (test_scores >= best_threshold).astype(int)

print("\n--- Final Adaptive SVM ---")
print("Threshold:", best_threshold)
print("Accuracy:", accuracy_score(y_test, final_pred))

print("\nClassification Report:")
print(classification_report(
    y_test,
    final_pred,
    target_names=["Ham", "Spam"]
))
"""
print(classification_report(
    y_test,
    svm_pred,
    target_names=["Ham", "Spam"]
))
"""
# Model Comparison
"""
results = pd.DataFrame({
    "Model": [
        "Multinomial Naive Bayes",
        "Logistic Regression",
        "Linear SVM"
    ],
    "Accuracy": [
        accuracy_score(y_test, nb_pred),
        accuracy_score(y_test, lr_pred),
        accuracy_score(y_test, svm_pred)
    ]
})
"""
#print("\n--- Model Comparison ---")
#print(results)
# SVM Decision Scores
svm_scores = svm_model.decision_function(X_test)
#print("\n--- SVM Decision Scores ---")
#print("First 10 scores:", svm_scores[:10])
# Adaptive Threshold Analysis

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

thresholds = [-0.5, -0.25, 0, 0.25, 0.5]

#print("\n--- Threshold Analysis ---")

for threshold in thresholds:

    threshold_pred = (svm_scores >= threshold).astype(int)

    precision = precision_score(y_test, threshold_pred)
    recall = recall_score(y_test, threshold_pred)
    f1 = f1_score(y_test, threshold_pred)
"""
    print(
        f"Threshold: {threshold:>5} | "
        f"Precision: {precision:.3f} | "
        f"Recall: {recall:.3f} | "
        f"F1: {f1:.3f}"
    )
"""
# Final Model Comparison

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
import pandas as pd


# Default Linear SVM prediction
# Default SVM threshold = 0
svm_pred = (test_scores >= 0).astype(int)


models = {
    "Naive Bayes": nb_pred,
    "Logistic Regression": lr_pred,
    "Linear SVM": svm_pred,
    "Adaptive SVM": final_pred
}


comparison = []

for name, predictions in models.items():

    comparison.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test,
            predictions,
            zero_division=0
        ),
        "Recall": recall_score(
            y_test,
            predictions,
            zero_division=0
        ),
        "F1 Score": f1_score(
            y_test,
            predictions,
            zero_division=0
        )
    })


comparison_df = pd.DataFrame(comparison)

print("\n--- Final Model Comparison ---")
print(comparison_df.round(4).to_string(index=False))
joblib.dump(best_threshold, "processed_data/best_threshold.pkl")
