import streamlit as st
import joblib
import nltk
import re
import string
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load model and vectorizer
model = joblib.load("main/svm_sentiment_model.pkl")
vectorizer = joblib.load("main/tfidf_vectorizer.pkl")

# Download NLTK data (only needed once)
nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")

# In-code mapping of condition to suggested medicines
condition_to_meds = {
    "Depression": ["Zoloft", "Prozac", "Lexapro"],
    "High Blood Pressure": ["Lisinopril", "Amlodipine", "Losartan"],
    "Diabetes": ["Metformin", "Insulin", "Glipizide"]
}

# Text cleaning function
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
st.set_page_config(page_title="Drug Review Analyzer", layout="centered")
st.title("💊 Drug Review Sentiment Analyzer")
st.markdown("Analyze user reviews for **Depression**, **High Blood Pressure**, or **Diabetes** drugs.")

condition = st.selectbox("Select Condition", list(condition_to_meds.keys()))
review_text = st.text_area("Enter your review:", height=200)

if st.button("Analyze"):
    if not review_text.strip():
        st.warning("Please enter a review.")
    else:
        cleaned = clean_text(review_text)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]

        sentiment = "Positive 😊" if prediction == 1 else "Negative 😞"
        st.subheader(f"Sentiment: {sentiment}")

        st.markdown("### 💊 Suggested Medicines:")
        st.success(", ".join(condition_to_meds.get(condition, ["No medicine suggestions available."])))
