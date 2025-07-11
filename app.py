import streamlit as st
import joblib
import nltk
import os
import re
import string
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ✅ Download required NLTK data at runtime
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# ✅ Load model and vectorizer
model = joblib.load("svm_sentiment_model (1).pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# ✅ Suggested medicines based on condition
condition_to_meds = {
    "Depression": ["Zoloft", "Prozac", "Lexapro"],
    "High Blood Pressure": ["Lisinopril", "Amlodipine", "Losartan"],
    "Diabetes": ["Metformin", "Insulin", "Glipizide"]
}

def clean_text(text):
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = re.findall(r'\b\w+\b', text)  # ✅ Regex tokenizer (no punkt needed)
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return ' '.join(tokens)

# ✅ Streamlit App UI
st.set_page_config(page_title="💊 Drug Review Classifier")
st.title("💊 Drug Review Condition Classifier")
st.markdown("Predict medical condition from a review and get suggested medicines.")

review_text = st.text_area("📝 Enter your drug review:")

if st.button("🔍 Analyze Review"):
    if not review_text.strip():
        st.warning("Please enter a review.")
    else:
        cleaned = clean_text(review_text)
        vec = vectorizer.transform([cleaned])
        label = model.predict(vec)[0]

        # Map label to condition
        label_map = {0: "Depression", 1: "High Blood Pressure", 2: "Diabetes"}
        condition = label_map.get(label, "Unknown")

        st.subheader(f"🧠 Predicted Condition: {condition}")
        st.markdown("### 💊 Suggested Medicines:")
        st.success(", ".join(condition_to_meds.get(condition, ["No suggestions available."])))
