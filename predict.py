import joblib
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# Load saved objects
tfidf = joblib.load("processed_data/tfidf.pkl")
selector = joblib.load("processed_data/selector.pkl")
svm_model = joblib.load("processed_data/adaptive_svm.pkl")
best_threshold = joblib.load("processed_data/best_threshold.pkl")


# Text preprocessing
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def preprocess(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^\w\s]", "", text)

    words = nltk.word_tokenize(text)

    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)


# Get SMS from user
message = input("Enter SMS: ")


# Preprocess
clean_message = preprocess(message)


# TF-IDF transformation
message_tfidf = tfidf.transform([clean_message])


# Feature selection
message_selected = selector.transform(message_tfidf)


# SVM decision score
score = svm_model.decision_function(message_selected)[0]


# Adaptive threshold classification
if score >= best_threshold:
    prediction = "SPAM"
else:
    prediction = "HAM"


print("\nPrediction:", prediction)
print("Decision Score:", score)
print("Threshold:", best_threshold)