#Text → Vectorize → Resample → Train


import re
import pandas as pd

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

from imblearn.over_sampling import RandomOverSampler

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


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
# LOAD DATA
# -----------------------------
df = pd.read_csv("spam.csv")

df['clean_text'] = df['sms'].apply(clean_text) #ek nya column clean_text bna rha hai

print("\nOriginal Label Distribution:")
print(df['label'].value_counts())   #0-1000, 1-400 


# -----------------------------
# TRAIN TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split( df['clean_text'], df['label'], test_size=0.2, random_state=42 )

print("\nTraining Label Distribution (Before Oversampling):")
print(y_train.value_counts())   #0-805 , 1-315 that is 80% 


# -----------------------------
# TF-IDF VECTORIZATION - to convert inputs X into vectors/numbers [ Vectorisation sirf INPUT (X) 
# ke liye hoti hai, kyunki model FEATURES se seekhta hai, LABEL se nahi.]
# -----------------------------
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)


# -----------------------------
# OVERSAMPLING (TRAINING WALA PART ONLY) - ["Fit" sirf - training data pe hota hai, "Transform" - training+test dono pe hota hai]
# -----------------------------
oversampler = RandomOverSampler(random_state=42)
X_resampled, y_resampled = oversampler.fit_resample( X_train_tfidf, y_train )

print("\nTraining Label Distribution (After Oversampling):")
print(pd.Series(y_resampled).value_counts())    #0-805 , 1-805 that is y oversample ho gya, x jitna same.


# -----------------------------
# MODEL TRAINING
# -----------------------------
model = MultinomialNB()
model.fit(X_resampled, y_resampled)

print("\nModel training completed!")


# -----------------------------
# PREDICTION ON TEST DATA
# -----------------------------
y_pred = model.predict(X_test_tfidf)


# -----------------------------
# EVALUATION
# -----------------------------
print("\nAccuracy:")
print(accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# -----------------------------
# REAL SMS TESTING
# -----------------------------
def predict_sms(sms):
    cleaned = clean_text(sms)
    vector = tfidf_vectorizer.transform([cleaned])
    prediction = model.predict(vector)[0]
    return "SPAM 🚨" if prediction == 1 else "HAM ✅"


print("\n--- Real SMS Testing ---")
print(predict_sms("Congratulations! You have won a free iPhone. Call now"))
print(predict_sms("Hi bro, we will meet at 6 pm today"))
