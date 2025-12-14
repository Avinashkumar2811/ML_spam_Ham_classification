import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

def clean_text(text):
    # Convert text to lowercase
    text = text.lower()

    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Remove links
    text = re.sub(r'http\S+', '', text)

    # Tokenize the text
    words = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word not in stop_words]

    # Initialize Porter Stemmer
    stemmer = PorterStemmer()

    # Perform stemming
    stemmed_words = [stemmer.stem(word) for word in filtered_words]

    # Join the stemmed words back into a single string
    cleaned_text = " ".join(stemmed_words)

    return cleaned_text


text = "I am LOVING Machine Learning!!! Visit https://ml.com"
print(clean_text(text))

