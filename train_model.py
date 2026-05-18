import os
import re
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

MODEL_PATH = "forensic_social_media_model.pkl"


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def limit_rows(df, max_rows):
    if len(df) > max_rows:
        return df.sample(n=max_rows, random_state=42)
    return df


def load_crime_dataset():
    path = "data/crime/labeled_data.csv"

    if not os.path.exists(path):
        return None

    print("Loading crime / hate speech dataset...")

    df = pd.read_csv(path)
    df = df.rename(columns={"tweet": "text"})

    if "text" not in df.columns or "class" not in df.columns:
        print("Crime dataset skipped because required columns were not found.")
        return None

    df = df[["text", "class"]].dropna()

    def map_label(value):
        value = int(value)

        if value == 0:
            return "hate"
        elif value == 1:
            return "offensive"
        return "normal"

    df["label"] = df["class"].apply(map_label)

    return limit_rows(df[["text", "label"]], 20000)


def load_twitter_hate_dataset():
    path = "data/twitter_hate/labeled_data.csv"

    if not os.path.exists(path):
        return None

    print("Loading additional Twitter hate dataset...")

    df = pd.read_csv(path)
    df = df.rename(columns={"tweet": "text"})

    if "text" not in df.columns or "class" not in df.columns:
        print("Twitter hate dataset skipped because required columns were not found.")
        return None

    df = df[["text", "class"]].dropna()

    def map_label(value):
        value = int(value)

        if value == 0:
            return "hate"
        elif value == 1:
            return "offensive"
        return "normal"

    df["label"] = df["class"].apply(map_label)

    return limit_rows(df[["text", "label"]], 15000)


def load_cyberbullying_dataset():
    path = "data/cyberbullying/cyberbullying_tweets.csv"

    if not os.path.exists(path):
        return None

    print("Loading cyberbullying dataset...")

    df = pd.read_csv(path)
    df = df.rename(columns={"tweet_text": "text"})

    if "text" not in df.columns or "cyberbullying_type" not in df.columns:
        print("Cyberbullying dataset skipped because required columns were not found.")
        return None

    df = df[["text", "cyberbullying_type"]].dropna()

    def map_label(value):
        value = str(value).lower()

        if value == "not_cyberbullying":
            return "normal"
        return "toxic"

    df["label"] = df["cyberbullying_type"].apply(map_label)

    return limit_rows(df[["text", "label"]], 20000)


def load_sentiment140_dataset():
    path = "data/sentiment140/training.1600000.processed.noemoticon.csv"

    if not os.path.exists(path):
        return None

    print("Loading Sentiment140 normal tweets...")

    df = pd.read_csv(
        path,
        encoding="latin-1",
        header=None,
        names=["target", "id", "date", "flag", "user", "text"]
    )

    df = df[["text"]].dropna()
    df["label"] = "normal"

    return limit_rows(df, 15000)


def load_spam_dataset():
    path = "data/spam/spam.csv"

    if not os.path.exists(path):
        return None

    print("Loading spam dataset...")

    df = pd.read_csv(path, encoding="latin-1")

    possible_text_cols = ["text", "message", "sms", "v2", "tweet", "content"]
    possible_label_cols = ["label", "class", "target", "v1", "category"]

    text_col = None
    label_col = None

    for col in df.columns:
        if col.lower() in possible_text_cols:
            text_col = col
        if col.lower() in possible_label_cols:
            label_col = col

    if text_col is None or label_col is None:
        if len(df.columns) >= 2:
            label_col = df.columns[0]
            text_col = df.columns[1]
        else:
            print("Spam dataset skipped because columns could not be detected.")
            return None

    df = df[[text_col, label_col]].dropna()
    df.columns = ["text", "raw_label"]

    def map_label(value):
        value = str(value).lower()
        if "spam" in value:
            return "spam"
        return "normal"

    df["label"] = df["raw_label"].apply(map_label)

    return limit_rows(df[["text", "label"]], 10000)


def load_fakenews_dataset():
    path = "data/fakenews/Fake.csv"

    if not os.path.exists(path):
        return None

    print("Loading fake news dataset...")

    df = pd.read_csv(path)

    if "text" not in df.columns:
        print("Fake news dataset skipped because text column was not found.")
        return None

    df = df[["text"]].dropna()
    df["label"] = "misinformation"

    return limit_rows(df, 10000)


def main():
    print("\n=== COS 783 SOCIAL MEDIA ANALYSIS MODEL TRAINING ===\n")

    loaders = [
        load_crime_dataset,
        load_twitter_hate_dataset,
        load_cyberbullying_dataset,
        load_sentiment140_dataset,
        load_spam_dataset,
        load_fakenews_dataset
    ]

    datasets = []

    for loader in loaders:
        data = loader()

        if data is not None and len(data) > 0:
            datasets.append(data)

    if len(datasets) == 0:
        print("No datasets found.")
        print("Please place datasets inside the data folders and run again.")
        return

    df = pd.concat(datasets, ignore_index=True)
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(str)
    df["clean_text"] = df["text"].apply(clean_text)
    df = df[df["clean_text"].str.len() > 3]

    print("\nCombined rows:", len(df))
    print("\nLabel distribution:")
    print(df["label"].value_counts())

    X = df["clean_text"]
    y = df["label"]

    stratify = y if y.value_counts().min() >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=stratify
    )

    custom_weights = {
        'normal': 1.0,
        'offensive': 1.1,
        'toxic': 1.2,
        'misinformation': 1.0,
        'spam': 1.8,
        'hate': 1.5  # Lower this slightly if it was too high under "balanced" automatically
    }

    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                stop_words="english",
                ngram_range=(1, 3),
                max_features=40000,
                min_df=2
            )
        ),
        (
            "classifier",
            LogisticRegression(
                penalty="l2",
                C=2.0,
                max_iter=1200,
                class_weight=custom_weights,
                random_state=42
            )
        )
    ])

    print("\nTraining model...")
    model.fit(X_train, y_train)

    print("Testing model...")
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print("\nAccuracy:", round(accuracy * 100, 2), "%")
    print("\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))

    bundle = {
        "model": model,
        "labels": sorted(df["label"].unique().tolist()),
        "rows_used": int(len(df)),
        "accuracy": float(accuracy)
    }

    joblib.dump(bundle, MODEL_PATH)

    print("\nModel saved as:", MODEL_PATH)


if __name__ == "__main__":
    main()
