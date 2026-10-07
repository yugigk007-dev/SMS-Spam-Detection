import pandas as pd
from sklearn.preprocessing import LabelEncoder
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
df = pd.read_csv("spam.csv", encoding="latin1")
#print(df.head())
#print("\nShape:", df.shape)
#print(df.columns)
df = df.drop(columns=["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"])
#print(df.head())
#print("\nShape:", df.shape)
#print(df.columns)
df.columns = ["label", "message"]
#print(df.head())
#print(df.columns)
#print(df.isnull().sum())
#print("Duplicate rows:", df.duplicated().sum())
df = df.drop_duplicates()
#print("Shape after removing duplicates:", df.shape)
encoder=LabelEncoder()
df["label"]=encoder.fit_transform(df["label"])
#print(df.head())
#print(df["label"].value_counts())
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
def preprocess(text):
    # lowercase
    text = text.lower()
    # remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)
    # remove numbers
    text = re.sub(r"\d+", "", text)
    # remove punctuation
    text = re.sub(r"[^\w\s]", "", text)
    # tokenize
    words = nltk.word_tokenize(text)
    # remove stopwords and lemmatize
    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]
    return " ".join(words)
df["clean_message"] = df["message"].apply(preprocess)
#print(df[["message","clean_message"]].head(10))
from sklearn.model_selection import train_test_split

X = df["clean_message"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.2,
    random_state=42,
    stratify=y_train
)
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print("\nTraining class distribution:")
print(y_train.value_counts())
print("\nTesting class distribution:")
print(y_test.value_counts())
# TF-IDF Feature Extraction
tfidf = TfidfVectorizer(
    max_features=3000,
    ngram_range=(1, 2)
)
X_train_tfidf = tfidf.fit_transform(X_train)
X_val_tfidf = tfidf.transform(X_val)
X_test_tfidf = tfidf.transform(X_test)
print("\n--- TF-IDF ---")
print("Training feature matrix:", X_train_tfidf.shape)
print("Testing feature matrix :", X_test_tfidf.shape)
# Chi-Square Feature Selection

selector = SelectKBest(
    score_func=chi2,
    k=1000
)
X_train_selected = selector.fit_transform(
    X_train_tfidf,
    y_train
)
X_val_selected = selector.transform(X_val_tfidf)
X_test_selected = selector.transform(X_test_tfidf)
print("\n--- Feature Selection ---")
print("Before selection:", X_train_tfidf.shape)
print("After selection :", X_train_selected.shape)
from scipy.sparse import save_npz

import os
from scipy.sparse import save_npz

os.makedirs("processed_data", exist_ok=True)

save_npz("processed_data/X_train.npz", X_train_selected)
save_npz("processed_data/X_val.npz", X_val_selected)
save_npz("processed_data/X_test.npz", X_test_selected)

y_train.to_csv("processed_data/y_train.csv", index=False)
y_val.to_csv("processed_data/y_val.csv", index=False)
y_test.to_csv("processed_data/y_test.csv", index=False)

print("\nProcessed data saved successfully.")
import joblib

joblib.dump(tfidf, "processed_data/tfidf.pkl")
joblib.dump(selector, "processed_data/selector.pkl")

print("Preprocessing objects saved successfully.")

