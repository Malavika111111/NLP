# app.py
import streamlit as st
import joblib
import re
import string
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

# Load saved model
model = joblib.load("sentiment_model_pipeline.pkl")

# Text preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text)
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)

# Streamlit app UI
st.set_page_config(page_title="Drug Review Sentiment Classifier")
st.title("💊 Drug Review Sentiment Classifier")
st.write("Predicts sentiment from a patient's drug review for:")
st.markdown("- Depression\n- High Blood Pressure\n- Diabetes, Type 2")

review = st.text_area("📝 Enter the patient review:")

if st.button("Predict Sentiment"):
    if review.strip() == "":
        st.warning("Please enter a valid review.")
    else:
        clean_review = preprocess_text(review)
        sentiment = model.predict([clean_review])[0]
        st.success(f"Predicted Sentiment: **{sentiment.upper()}**")
