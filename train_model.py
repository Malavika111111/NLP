# Install Required Libraries
!pip install joblib nltk beautifulsoup4 scikit-learn openpyxl

# Import Libraries
import pandas as pd
import joblib
import nltk
import re
import string
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Download NLTK Resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load Excel Dataset
data = pd.read_excel("/content/sample_data/drugsCom_raw.xlsx")

# Drop unwanted column if exists
if 'Unnamed: 0' in data.columns:
    data.drop(columns='Unnamed: 0', inplace=True)

# Filter only specific conditions
target_conditions = ["Depression", "High Blood Pressure", "Diabetes, Type 2"]
data = data[data["condition"].isin(target_conditions)].dropna(subset=["review"]).copy()
data.reset_index(drop=True, inplace=True)

# Text Preprocessing
stop = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess_text(raw_review):
    review_text = BeautifulSoup(raw_review, 'html.parser').get_text()
    letters_only = re.sub('[^a-zA-Z]', ' ', review_text)
    words = letters_only.lower().split()
    meaningful_words = [w for w in words if w not in stop]
    lemmatized_words = [lemmatizer.lemmatize(w) for w in meaningful_words]
    return ' '.join(lemmatized_words)

data["clean_review"] = data["review"].apply(preprocess_text)

# Encode Labels
label_map = {
    "Depression": 0,
    "High Blood Pressure": 1,
    "Diabetes, Type 2": 2
}
data["label"] = data["condition"].map(label_map)

# Split Dataset
X = data["clean_review"]
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=5000)
X_train_vec = tfidf.fit_transform(X_train)
X_test_vec = tfidf.transform(X_test)

# Train SVM
svm_model = SVC(kernel='linear', C=1, probability=True, random_state=42)
svm_model.fit(X_train_vec, y_train)

# Evaluate Model
y_pred = svm_model.predict(X_test_vec)

print("Accuracy   :", accuracy_score(y_test, y_pred))
print("Precision  :", precision_score(y_test, y_pred, average='weighted'))
print("Recall     :", recall_score(y_test, y_pred, average='weighted'))
print("F1 Score   :", f1_score(y_test, y_pred, average='weighted'))

print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save Model and Vectorizer
joblib.dump(svm_model, "svm_sentiment_model.pkl")
joblib.dump(tfidf, "tfidf_vectorizer.pkl")
print("✅ Files saved")

# Download Files from Colab
from google.colab import files
files.download("svm_sentiment_model.pkl")
files.download("tfidf_vectorizer.pkl")
