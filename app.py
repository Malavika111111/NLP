import streamlit as st
import joblib
import nltk
import re
import string
import os
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# NLTK Downloads
nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")

# Label mapping
label_to_condition = {
    0: "Depression",
    1: "High Blood Pressure",
    2: "Diabetes"
}

# Medicine suggestions
condition_to_meds = {
    "Depression": ["Zoloft", "Prozac", "Lexapro"],
    "High Blood Pressure": ["Lisinopril", "Amlodipine", "Losartan"],
    "Diabetes": ["Metformin", "Insulin", "Glipizide"]
}

# Text cleaning
def clean_text(text):
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = nltk.word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return ' '.join(tokens)

# File paths
st.set_page_config(page_title="Drug Review Analyzer", layout="centered")
BASE_PATH = os.path.dirname(__file__)
model_path = os.path.join(BASE_PATH, "svm_sentiment_model (1).pkl")
vectorizer_path = os.path.join(BASE_PATH, "tfidf_vectorizer.pkl")

# Show file debug info
st.write("🗂 Current working directory:", os.getcwd())
st.write("📁 Files in current directory:", os.listdir())

# Load model
model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

# UI
st.title("💊 Drug Review Condition Classifier")
st.markdown("Predict **medical condition** from a review and get suggested medicines.")

review_text = st.text_area("📝 Enter your drug review:", height=200)

if st.button("Analyze"):
    if not review_text.strip():
        st.warning(" Please enter a valid review.")
    else:
        cleaned = clean_text(review_text)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        predicted_condition = label_to_condition.get(prediction, "Unknown")

        st.subheader(f"Predicted Condition: **{predicted_condition}**")
        st.markdown("### 💊 Suggested Medicines:")
        st.success(", ".join(condition_to_meds.get(predicted_condition, ["No suggestions available."])))
