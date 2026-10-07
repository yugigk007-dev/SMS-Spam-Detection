import streamlit as st
import joblib
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# Page settings
st.set_page_config(
    page_title="SMS Spam Detector",
    page_icon="📱",
    layout="centered"
)


# Load saved ML objects
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


# UI
st.title("📱 SMS Spam Detector")
st.write("Enter an SMS message to classify it as Ham or Spam.")

message = st.text_area(
    "Enter your SMS:",
    height=150,
    placeholder="Example: Congratulations! You have won a free prize..."
)


if st.button("Check Message"):

    if not message.strip():
        st.warning("Please enter an SMS message.")

    else:
        # Preprocess message
        clean_message = preprocess(message)

        # TF-IDF
        message_tfidf = tfidf.transform([clean_message])

        # Feature selection
        message_selected = selector.transform(message_tfidf)

        # SVM decision score
        score = svm_model.decision_function(message_selected)[0]

        # Adaptive threshold
        if score >= best_threshold:
            prediction = "SPAM"
        else:
            prediction = "HAM"


        # Display result
        st.subheader("Result")

        if prediction == "SPAM":
            st.error("🚨 SPAM MESSAGE")
        else:
            st.success("✅ HAM MESSAGE")

        st.write(f"**Decision Score:** {score:.4f}")
        st.write(f"**Adaptive Threshold:** {best_threshold:.2f}")