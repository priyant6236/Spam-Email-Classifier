# main.py
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# ==========================
# Step 1: Load Dataset
# ==========================
csv_file = "spam.csv"

# Fallback: create sample dataset if spam.csv is not yet present
if not os.path.exists(csv_file):
    print(f"[!] '{csv_file}' not found in the current directory.")
    print("[*] Creating a sample 'spam.csv' so you can run the code immediately...")
    sample_data = {
        "label": [v
            "spam", "ham", "spam", "ham", "spam", "ham", "spam", "ham",
            "spam", "ham", "spam", "ham", "spam", "ham", "spam", "ham",
            "spam", "ham", "spam", "ham", "spam", "ham", "spam", "ham"
        ],
        "text": [
            "WINNER!! You have won a $1,000 cash prize! Text CLAIM to 88888 now!",
            "Hey, are you free this afternoon to study for the exam?",
            "URGENT: Your bank account has been flagged. Verify here: http://bit.ly/spam",
            "I'm on my way home, see you in 10 minutes.",
            "Free entry in 2 a weekly competition to win FA Cup final tickets!",
            "Can you pick up some milk on your way back?",
            "Exclusive offer! Call 09061743806 now to receive your secret holiday bonus!",
            "Sounds good, let me know when you arrive.",
            "Congratulations! Reply WIN to claim your guaranteed £5000 reward.",
            "Are we still meeting at Starbucks tomorrow morning?",
            "You have won a free iPhone 15! Click link to confirm delivery address.",
            "Don't forget to submit the assignment before midnight.",
            "Hot singles in your area want to chat! Send CHAT to 55555 now!",
            "Thanks for helping me yesterday, really appreciate it!",
            "Loan approved! Get up to $10,000 today with zero interest. Apply now.",
            "Can we reschedule our call to 3 PM?",
            "Alert: Your package delivery failed. Confirm info at spam-link.org",
            "Did you watch the match last night? It was amazing!",
            "You are selected for a free $500 gift card! Reply YES to claim.",
            "What time does the movie start?",
            "Double your money in 24 hours! Guaranteed crypto returns. Join now.",
            "I will be late for the meeting by 15 minutes, please start without me.",
            "Final notice: You have unclaimed rewards expiring today. Call 1-800-SPAM.",
            "Happy birthday! Wishing you a fantastic day and year ahead!"
        ] * 10
    }
    df_sample = pd.DataFrame(sample_data)
    df_sample.to_csv(csv_file, index=False)
    print(f"[+] Sample dataset saved to '{csv_file}'. Replace this file anytime with your real dataset.\n")

# Read CSV with encoding fallback
try:
    df = pd.read_csv(csv_file, encoding="latin-1")
except UnicodeDecodeError:
    df = pd.read_csv(csv_file, encoding="utf-8")

# Support both Kaggle ("Category"/"Message") and SMS Collection ("v1"/"v2") column formats
if "Category" in df.columns and "Message" in df.columns:
    df = df[["Category", "Message"]].rename(columns={"Category": "label", "Message": "text"})
elif "v1" in df.columns and "v2" in df.columns:
    df = df[["v1", "v2"]].rename(columns={"v1": "label", "v2": "text"})
elif "label" in df.columns and "text" in df.columns:
    df = df[["label", "text"]]
else:
    df = df.iloc[:, [0, 1]]
    df.columns = ["label", "text"]

df = df.dropna().reset_index(drop=True)

print("=" * 40)
print("Dataset Loaded Successfully!")
print("=" * 40)
print(df.head())
print("\nClass distribution:")
print(df["label"].value_counts())


# ==========================
# Step 2: Encode Labels
# ==========================
le = LabelEncoder()
df["label_num"] = le.fit_transform(df["label"])
print("\nLabel encoding mapping:", dict(zip(le.classes_, le.transform(le.classes_))))


# ==========================
# Step 3: Train/Test Split
# ==========================
X = df["text"]
y = df["label_num"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print(f"\nTraining samples: {len(X_train)} | Testing samples: {len(X_test)}")


# ==========================
# Step 4: Define Pipelines
# ==========================
pipeline_lr = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english")),
    ("clf", LogisticRegression(solver="liblinear"))
])

pipeline_svm = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english")),
    ("clf", SVC(kernel="linear", probability=True))
])

pipeline_rf = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english")),
    ("clf", RandomForestClassifier(n_estimators=200, random_state=42))
])

models = {
    "Logistic Regression": pipeline_lr,
    "SVM": pipeline_svm,
    "Random Forest": pipeline_rf
}


# ==========================
# Step 5: Train & Evaluate
# ==========================
results = []
os.makedirs("plots", exist_ok=True)

for name, model in models.items():
    print(f"\n{'=' * 20} {name} {'=' * 20}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))

    # Confusion matrix plot (saved to plots/ so script runs smoothly without manual window clicks)
    plt.figure(figsize=(5, 4))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plot_path = f"plots/cm_{name.lower().replace(' ', '_')}.png"
    plt.savefig(plot_path)
    plt.show(block=False)
    plt.pause(1)
    plt.close()
    print(f"Confusion matrix saved to '{plot_path}'")

    results.append([name, acc, prec, rec, f1])

results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "Precision", "Recall", "F1"])
print("\n" + "=" * 50)
print("Performance Comparison:")
print("=" * 50)
print(results_df.to_string(index=False))


# ==========================
# Step 6: Ensemble Model
# ==========================
print(f"\n{'=' * 20} Ensemble (Voting Classifier) {'=' * 20}")
ensemble = VotingClassifier(
    estimators=[("lr", pipeline_lr), ("svm", pipeline_svm), ("rf", pipeline_rf)],
    voting="soft"
)

ensemble.fit(X_train, y_train)
y_pred = ensemble.predict(X_test)

print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))

plt.figure(figsize=(5, 4))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title("Confusion Matrix - Ensemble")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plot_path = "plots/cm_ensemble.png"
plt.savefig(plot_path)
plt.show(block=False)
plt.pause(1)
plt.close()
print(f"Ensemble confusion matrix saved to '{plot_path}'")


# ==========================
# Step 7: Test Custom Inputs
# ==========================
print("\n" + "=" * 50)
print("Testing Custom Inputs:")
print("=" * 50)

test_msgs = [
    "Congratulations! You have won a free lottery ticket worth $1000. Claim now!",
    "Hey, are we still meeting for lunch today?",
    "URGENT! Your account will be suspended unless you verify your details immediately!"
]

preds = ensemble.predict(test_msgs)
probs = ensemble.predict_proba(test_msgs)

for msg, pred, prob in zip(test_msgs, preds, probs):
    pred_label = le.inverse_transform([pred])[0]
    confidence = np.max(prob) * 100
    print(f"\nMessage:    {msg}")
    print(f"Prediction: {pred_label.upper()} (Confidence: {confidence:.2f}%)")