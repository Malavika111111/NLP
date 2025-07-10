import streamlit as st
import joblib
import json
import numpy as np

from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import string

# Load model and mappings
model = joblib.load("svm_sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
with open("drug_mapping.json") as f:
    drug_dict = json.load(f)

# Preprocessing function
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

# Streamlit UI
st.set_page_config(page_title="Drug Review Sentiment Analyzer", layout="centered")
st.title("💊 Drug Review Sentiment Prediction")
st.markdown("Enter a **drug review**, select a condition, and get the predicted sentiment along with suggested medicines.")

condition = st.selectbox("Select Condition", ["Depression", "High Blood Pressure", "Diabetes"])
review_text = st.text_area("Enter your drug review here:", height=200)

if st.button("Analyze"):
    if review_text.strip() == "":
        st.warning("Please enter a review.")
    else:
        cleaned = clean_text(review_text)
        transformed = vectorizer.transform([cleaned])
        pred = model.predict(transformed)[0]
        label = "Positive 😊" if pred == 1 else "Negative 😞"
        st.subheader(f"Sentiment: {label}")

        st.markdown("### 💊 Recommended Medicines:")
        st.success(", ".join(drug_dict.get(condition.lower(), ["No data available"])))
