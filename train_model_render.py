import os
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# -----------------------------------
# Load Dataset
# -----------------------------------
# Your CSV must have columns:
# URL , Label
# Label values: good / bad   (or 0 / 1)

DATA_PATH = "phishing_site_urls.csv"

df = pd.read_csv(DATA_PATH)

print("Dataset loaded:", df.shape)

# Normalize labels
df["Label"] = df["Label"].astype(str).str.lower()

# Map labels to 0/1 if needed
df["Label"] = df["Label"].replace({
    "good": 0,
    "bad": 1,
    "safe": 0,
    "malicious": 1
})

X = df["URL"]
y = df["Label"]

# -----------------------------------
# Train Test Split
# -----------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------------
# TF-IDF Vectorizer
# -----------------------------------
vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    max_features=50000
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# -----------------------------------
# Logistic Regression Model
# -----------------------------------
model = LogisticRegression(
    max_iter=2000
)

model.fit(X_train_vec, y_train)

# -----------------------------------
# Evaluate
# -----------------------------------
preds = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, preds)

print("Model Accuracy:", round(accuracy * 100, 2), "%")

# -----------------------------------
# Save Models
# -----------------------------------
os.makedirs("models", exist_ok=True)

with open("models/vectorizer_lr_new.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

with open("models/model_lr_new.pkl", "wb") as f:
    pickle.dump(model, f)

print("Models saved successfully inside /models folder")