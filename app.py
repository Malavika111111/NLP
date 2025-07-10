import streamlit as st
import joblib

model = joblib.load("svm_sentimental_model.pkl")

st.title("💊 Drug Review Sentiment Predictor")

user_input = st.text_area("Enter your drug review")

if st.button("Predict"):
    prediction = model.predict([user_input])[0]
    st.success(f"Predicted Sentiment: {prediction}")
