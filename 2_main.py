"""Raw SMS → Text Cleaning → Train - Test Split → TF-IDF Vectorization → Oversampling (Train only)
 → Model Training → Evaluation"""

import re
import pandas as pd

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


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

df = pd.read_csv("spam.csv") #csv has sms and label
print(df)

df['clean_text'] = df['sms'].apply(lambda x: clean_text(x)) #making 3rd column clean_text
print(df)

print(df["label"].value_counts())

X_train, X_test, y_train, y_test = train_test_split( df['clean_text'], df['label'], test_size=0.2, random_state=42)  #Dataset Count X-Train (80%)=1120, X-Test(20%)=280

tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

#Oversampling (SIRF TRAIN DATA)
oversampler = RandomOverSampler()
X_resampled, y_resampled = oversampler.fit_resample(X_train_tfidf, y_train)

print("Before Oversampling:")
print(y_train.value_counts())

print("After Oversampling:")
print(pd.Series(y_resampled).value_counts())
