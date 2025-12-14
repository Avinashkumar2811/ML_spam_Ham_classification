# 📩 SMS Spam Classifier (End-to-End ML Project)

This project builds an **end-to-end SMS Spam Detection system** using **Natural Language Processing (NLP)** and **Machine Learning**.
It classifies SMS messages into:

* **HAM (0)** – Normal message
* **SPAM (1)** – Promotional/Fraud message

---

## 🚀 Project Flow (Pipeline)

```
Raw SMS
 → Text Cleaning
 → Train-Test Split
 → TF-IDF Vectorization
 → Oversampling (Train data only)
 → Model Training (Naive Bayes)
 → Evaluation
 → Real SMS Prediction
```

---

## 🧠 Key Concepts Used

* Text Preprocessing (NLTK)
* TF-IDF Vectorization
* Handling Imbalanced Data (RandomOverSampler)
* Naive Bayes Classification
* Model Evaluation Metrics

---

## 📂 Project Structure

```
ml_spam_ham/
│
├── main.py          # Complete ML pipeline code
├── spam.csv         # Dataset (sms, label)
├── README.md        # Project documentation
└── .virt/           # Virtual environment (optional)
```

---

## 📊 Dataset Details

* **File**: `spam.csv`
* **Columns**:

  * `sms`   → SMS text
  * `label` → 0 (HAM), 1 (SPAM)

### Label Distribution (Before Oversampling)

* HAM: 1000
* SPAM: 400

---

## 🧹 Text Cleaning Steps

Each SMS goes through the following preprocessing:

1. Convert text to lowercase
2. Remove URLs
3. Remove punctuation & special characters
4. Tokenization
5. Stopword removal (NLTK)
6. Stemming (PorterStemmer)

**Example:**

```
"Congratulations! You have won a free iPhone"
→
```
