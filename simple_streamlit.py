# streamlit_app.py
# Simple UI for SMS Spam Classifier using Streamlit

import re
import pandas as pd
import main_streamlit as st

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# -----------------------------
# TEXT CLEANING FUNCTION
# -----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if w not in stop_words]

    stemmer = PorterStemmer()
    words = [stemmer.stem(w) for w in words]

    return " ".join(words)

# -----------------------------
# LOAD & TRAIN MODEL (ONCE)
# -----------------------------
@st.cache_resource
def train_model():
    df = pd.read_csv("spam.csv")
    df['clean_text'] = df['sms'].apply(clean_text)

    X_train, X_test, y_train, y_test = train_test_split(
        df['clean_text'], df['label'], test_size=0.2, random_state=42
    )

    tfidf = TfidfVectorizer()
    X_train_tfidf = tfidf.fit_transform(X_train)

    oversampler = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = oversampler.fit_resample(X_train_tfidf, y_train)

    model = MultinomialNB()
    model.fit(X_resampled, y_resampled)

    return model, tfidf

model, tfidf_vectorizer = train_model()

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.set_page_config(page_title="SMS Spam Detector", page_icon="📩")

st.title("📩 SMS Spam Detection App")
st.write("Enter an SMS below to check whether it is **SPAM** or **HAM**.")

sms_input = st.text_area("✉️ Enter SMS message")

if st.button("Predict"):
    if sms_input.strip() == "":
        st.warning("Please enter an SMS")
    else:
        cleaned = clean_text(sms_input)
        vector = tfidf_vectorizer.transform([cleaned])
        prediction = model.predict(vector)[0]

        if prediction == 1:
            st.error("🚨 SPAM MESSAGE")
        else:
            st.success("✅ HAM (Normal Message)")

st.markdown("---")
st.caption("Built using NLP + Naive Bayes + Streamlit")
