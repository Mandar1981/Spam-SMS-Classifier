"""
app.py
------
Flask web app for the Spam Email/SMS Classifier.

Loads the pre-trained TF-IDF vectorizer + Naive Bayes model
(created by train_model.py) and serves a simple web page where
a user can paste a message and instantly see if it's Spam or Not Spam.

Run with:  python app.py
Then open: http://127.0.0.1:5000
"""

import re
import string
import pickle

from flask import Flask, render_template, request
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

app = Flask(__name__)

STOPWORDS = set(stopwords.words("english"))
STEMMER = PorterStemmer()

# Load the trained model + vectorizer once, at startup
with open("model/model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)


def clean_text(text: str) -> str:
    """Same preprocessing used during training — must match exactly."""
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    text = re.sub(r"\d+", " ", text)
    tokens = text.split()
    tokens = [STEMMER.stem(word) for word in tokens if word not in STOPWORDS and len(word) > 1]
    return " ".join(tokens)


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    confidence = None
    message = ""

    if request.method == "POST":
        message = request.form.get("message", "")
        if message.strip():
            cleaned = clean_text(message)
            vec = vectorizer.transform([cleaned])
            prediction = model.predict(vec)[0]
            proba = model.predict_proba(vec)[0]

            result = "Spam" if prediction == 1 else "Not Spam"
            confidence = round(max(proba) * 100, 2)

    return render_template("index.html", result=result, confidence=confidence, message=message)


if __name__ == "__main__":
    app.run(debug=True)
