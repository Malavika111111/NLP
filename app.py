import streamlit as st
import joblib
import nltk
import os
import re
import string
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ✅ Download only essential NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

# ✅ Load model and vectorizer
model = joblib.load("svm_sentiment_model (1).pkl")  # Rename if needed
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# ✅ Suggested medicines by condition
condition_to_meds = {
    "Depression": ["Zoloft", "Prozac", "Lexapro"],
    "High Blood Pressure": ["Lisinopril", "Amlodipine", "Losartan"],
    "Diabetes": ["Metformin", "Insulin", "Glipizide"]
}

# ✅ Preprocessing function
def clean_text(text):
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = re.findall(r'\b\w+\b', text)
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return ' '.join(tokens)

# ✅ Streamlit UI
st.set_page_config(page_title="💊 Drug Review Analyzer")
st.title("💊 Drug Review Condition & Sentiment Classifier")
st.markdown("Enter a drug review to predict the **medical condition** and whether the **sentiment is Positive or Negative**.")

review_text = st.text_area("📝 Enter your drug review:")

if st.button("🔍 Analyze Review"):
    if not review_text.strip():
        st.warning("Please enter a review.")
    else:
        cleaned = clean_text(review_text)
        vec = vectorizer.transform([cleaned])
        prediction = model.predict(vec)[0]

        # Mapping labels
        condition_map = {0: "Depression", 1: "High Blood Pressure", 2: "Diabetes"}
        sentiment_map = {0: "Negative 😞", 1: "Neutral 🙂", 2: "Positive 😊"}

        # Check validity
        if prediction in condition_map:
            predicted_condition = condition_map[prediction]
            predicted_sentiment = sentiment_map.get(prediction, "Neutral ❓")

            st.subheader(f"🧠 Predicted Condition: {predicted_condition}")
            st.subheader(f"📊 Sentiment: {predicted_sentiment}")

            st.markdown("### 💊 Suggested Medicines:")
            st.success(", ".join(condition_to_meds.get(predicted_condition, ["No suggestions available."])))
        else:
            st.error("⚠️ Invalid prediction. Condition could not be determined.")
