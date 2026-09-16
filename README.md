# Spam Email/SMS Classifier

A machine learning web app that classifies text messages as **Spam** or **Not Spam**,
built with scikit-learn (Naive Bayes) and Flask.

## How it works

1. **Preprocessing** — each message is lowercased, stripped of punctuation/numbers,
   tokenized, cleaned of stopwords, and stemmed (`nltk`).
2. **Feature extraction** — cleaned text is converted into numeric features with
   **TF-IDF vectorization** (`TfidfVectorizer`, top 3000 features).
3. **Model** — a **Multinomial Naive Bayes** classifier is trained on the
   [SMS Spam Collection dataset](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)
   (5,169 labeled messages). Logistic Regression is trained alongside it for
   comparison, and the better-performing model is saved automatically.
4. **Web app** — a Flask app loads the saved model + vectorizer and serves a
   single-page UI where you paste a message and get an instant prediction with
   a confidence score.

**Result on held-out test data: ~97% accuracy.**

## Project structure

```
spam-classifier/
├── spam.csv              # dataset
├── train_model.py        # preprocessing + training script
├── app.py                # Flask web app
├── requirements.txt
├── model/
│   ├── model.pkl          # trained classifier (generated)
│   ├── vectorizer.pkl      # fitted TF-IDF vectorizer (generated)
│   └── model_info.txt
└── templates/
    └── index.html         # web UI
```

## Setup & run locally

```bash
# 1. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download NLTK data (one-time)
python -c "import nltk; nltk.download('stopwords')"

# 4. Train the model (creates model/model.pkl + model/vectorizer.pkl)
python train_model.py

# 5. Run the web app
python app.py
```

Then open **http://127.0.0.1:5000** in your browser, paste a message, and click
"Check message".

## Where to get the dataset

This project uses the classic **SMS Spam Collection** dataset (5,572 messages,
labeled ham/spam). It's included here as `spam.csv`. If you need it again, it's
mirrored in multiple places, including Kaggle ("SMS Spam Collection Dataset")
and the UCI Machine Learning Repository.

## Possible improvements (good talking points in an interview)

- Swap Naive Bayes for a fine-tuned DistilBERT for higher recall on tricky spam.
- Add a `/predict` JSON API endpoint so the model can be used outside the web UI.
- Deploy to Render/Railway/PythonAnywhere and link the live demo on your CV.
- Add a feedback loop: let users flag wrong predictions and retrain periodically.
- Track precision/recall per class, since spam is the minority class — accuracy
  alone can be misleading on imbalanced data like this.
