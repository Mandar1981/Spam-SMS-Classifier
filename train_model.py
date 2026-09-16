"""
train_model.py
---------------
Trains an SMS/Email Spam Classifier using Naive Bayes.

Pipeline:
1. Load dataset (spam.csv - SMS Spam Collection dataset)
2. Clean + preprocess text (lowercase, remove punctuation/numbers,
   tokenize, remove stopwords, stem)
3. Convert text to numeric features using TF-IDF
4. Train a Multinomial Naive Bayes classifier
5. Evaluate accuracy on a held-out test set
6. Save the trained model + vectorizer to disk (model/ folder)
   so the Flask app can load them instantly without retraining.

Run with:  python train_model.py
"""

import re
import string
import pickle

import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, classification_report

STOPWORDS = set(stopwords.words("english"))
STEMMER = PorterStemmer()


def clean_text(text: str) -> str:
    """
    Basic text preprocessing:
    - lowercase
    - remove punctuation & digits
    - tokenize on whitespace
    - remove stopwords
    - stem each word (running -> run)
    """
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    text = re.sub(r"\d+", " ", text)
    tokens = text.split()
    tokens = [STEMMER.stem(word) for word in tokens if word not in STOPWORDS and len(word) > 1]
    return " ".join(tokens)


def load_data(path: str = "spam.csv") -> pd.DataFrame:
    # The classic Kaggle SMS Spam Collection csv has some extra empty columns
    df = pd.read_csv(path, encoding="latin-1")
    df = df.iloc[:, :2]
    df.columns = ["label", "message"]
    df = df.drop_duplicates()
    df["label_num"] = df["label"].map({"ham": 0, "spam": 1})
    return df


def main():
    print("Loading data...")
    df = load_data()
    print(f"Total messages: {len(df)}  |  Spam: {df['label_num'].sum()}  |  Ham: {(df['label_num']==0).sum()}")

    print("Cleaning text (tokenization + stopword removal + stemming)...")
    df["clean_message"] = df["message"].apply(clean_text)

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_message"], df["label_num"], test_size=0.2, random_state=42, stratify=df["label_num"]
    )

    print("Vectorizing text with TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=3000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print("\nTraining Multinomial Naive Bayes...")
    nb_model = MultinomialNB()
    nb_model.fit(X_train_vec, y_train)
    nb_pred = nb_model.predict(X_test_vec)
    nb_acc = accuracy_score(y_test, nb_pred)
    nb_prec = precision_score(y_test, nb_pred)
    print(f"Naive Bayes  -> Accuracy: {nb_acc:.4f}  Precision: {nb_prec:.4f}")

    print("\nTraining Logistic Regression (for comparison)...")
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train_vec, y_train)
    lr_pred = lr_model.predict(X_test_vec)
    lr_acc = accuracy_score(y_test, lr_pred)
    lr_prec = precision_score(y_test, lr_pred)
    print(f"Logistic Reg -> Accuracy: {lr_acc:.4f}  Precision: {lr_prec:.4f}")

    # Pick the better performing model automatically (Naive Bayes usually wins here)
    if nb_acc >= lr_acc:
        best_model, best_name, best_pred = nb_model, "Multinomial Naive Bayes", nb_pred
    else:
        best_model, best_name, best_pred = lr_model, "Logistic Regression", lr_pred

    print(f"\n>>> Best model: {best_name} <<<")
    print("\nClassification report:\n", classification_report(y_test, best_pred, target_names=["ham", "spam"]))
    print("Confusion matrix:\n", confusion_matrix(y_test, best_pred))

    print("\nSaving model + vectorizer to model/ folder...")
    with open("model/model.pkl", "wb") as f:
        pickle.dump(best_model, f)
    with open("model/vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    with open("model/model_info.txt", "w") as f:
        f.write(f"Model: {best_name}\nAccuracy: {max(nb_acc, lr_acc):.4f}\n")

    print("Done! Model saved as model/model.pkl and model/vectorizer.pkl")


if __name__ == "__main__":
    main()
