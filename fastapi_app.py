import re
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import nltk

from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

nltk.download("punkt")
nltk.download("stopwords")

app = FastAPI(title="SMS Spam Classifier API")

# -----------------------------
# TEXT CLEANING
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
# TRAIN MODEL (ONCE)
# -----------------------------
df = pd.read_csv("spam.csv")
df["clean_text"] = df["sms"].apply(clean_text)

X_train, _, y_train, _ = train_test_split(
    df["clean_text"], df["label"], test_size=0.2, random_state=42
)

tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

oversampler = RandomOverSampler(random_state=42)
X_resampled, y_resampled = oversampler.fit_resample(X_train_tfidf, y_train)

model = MultinomialNB()
model.fit(X_resampled, y_resampled)

# -----------------------------
# REQUEST SCHEMA
# -----------------------------
class SMSRequest(BaseModel):
    sms: str

# -----------------------------
# PREDICTION ENDPOINT
# -----------------------------
@app.post("/predict")
def predict_sms(data: SMSRequest):
    cleaned = clean_text(data.sms)
    vector = tfidf_vectorizer.transform([cleaned])
    prediction = model.predict(vector)[0]

    return {
        "prediction": "SPAM" if prediction == 1 else "HAM"
    }

