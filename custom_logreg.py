'''
1684_HW2 Homework - Logistic Regression Classifier (Optimized)
Author: Your Name
Course: 1684
Description: Custom Logistic Regression classifier for offensive language detection.
'''

import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report
)
import joblib
from sklearn.model_selection import GridSearchCV

# =============================
# 🔧 Setup Paths
# =============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'civility_data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# =============================
# 📂 Load Data
# =============================
print("\n[1] Loading training and development data...")
train_path = os.path.join(DATA_DIR, 'train.tsv')
dev_path = os.path.join(DATA_DIR, 'dev.tsv')
demo_dev_path = os.path.join(DATA_DIR, 'mini_demographic_dev.tsv')
test_path = os.path.join(DATA_DIR, 'test.tsv')

train = pd.read_csv(train_path, sep='\t')
dev = pd.read_csv(dev_path, sep='\t')
demo_dev = pd.read_csv(demo_dev_path, sep='\t')
test = pd.read_csv(test_path, sep='\t')

print(f"Loaded {len(train)} training, {len(dev)} dev, {len(test)} test samples.")

# =============================
# ✂️ Preprocessing and Vectorization
# =============================
print("\n[2] Vectorizing text data (TF-IDF)...")

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words='english',
    ngram_range=(1, 2),      # unigrams + bigrams
    max_features=20000,      # slightly larger vocab
    sublinear_tf=True        # ✅ smoother weighting
)

X_train = vectorizer.fit_transform(train['text'])
y_train = train['label']

X_dev = vectorizer.transform(dev['text'])
y_dev = dev['label']

# =============================
# 🧠 Train Logistic Regression Model
# =============================
print("\n[3] Training Logistic Regression model...")

clf = LogisticRegression(
    max_iter=1000,
    class_weight={'NOT': 1.0, 'OFF': 1.3},  # ✅ heavier weight for OFF
    solver='liblinear'
)

clf.fit(X_train, y_train)

# Save model + vectorizer for reproducibility
joblib.dump(clf, os.path.join(RESULTS_DIR, 'logreg_model.pkl'))
joblib.dump(vectorizer, os.path.join(RESULTS_DIR, 'tfidf_vectorizer.pkl'))

# =============================
# 📊 Evaluate on dev.tsv (threshold tuning)
# =============================
print("\n[4] Evaluating on dev.tsv...")

# Default evaluation
probs = clf.predict_proba(X_dev)[:, list(clf.classes_).index('OFF')]

best_f1, best_t = 0, 0.5
for t in [0.35, 0.4, 0.45, 0.5, 0.55]:
    preds = ['OFF' if p > t else 'NOT' for p in probs]
    macro = f1_score(y_dev, preds, average='macro')
    print(f"Threshold {t:.2f}: macro F1 = {macro:.3f}")
    if macro > best_f1:
        best_f1, best_t = macro, t

print(f"\n✅ Best threshold = {best_t:.2f} → macro F1 ≈ {best_f1:.3f}")

# Final predictions using best threshold
y_pred = ['OFF' if p > best_t else 'NOT' for p in probs]

accuracy = accuracy_score(y_dev, y_pred)
macro_f1 = f1_score(y_dev, y_pred, average='macro')

print(f"\nFinal Accuracy: {accuracy:.4f}")
print(f"Final Macro F1: {macro_f1:.4f}")
print("\nDetailed classification report:")
print(classification_report(y_dev, y_pred))

# =============================
# 👥 Evaluate FPR on mini_demographic_dev.tsv
# =============================
print("\n[5] Evaluating FPR over demographic dev set...")

X_demo = vectorizer.transform(demo_dev['text'])
demo_probs = clf.predict_proba(X_demo)[:, list(clf.classes_).index('OFF')]
demo_dev['pred_label'] = ['OFF' if p > best_t else 'NOT' for p in demo_probs]

demo_fpr = demo_dev.groupby('demographic')['pred_label'].apply(lambda x: (x == 'OFF').mean())
print("\nFalse Positive Rate by Demographic:")
print(demo_fpr)

# =============================
# 💾 Generate Predictions for test.tsv
# =============================
print("\n[6] Generating predictions for test.tsv...")

X_test = vectorizer.transform(test['text'])
test_probs = clf.predict_proba(X_test)[:, list(clf.classes_).index('OFF')]
test['label'] = ['OFF' if p > best_t else 'NOT' for p in test_probs]

output_path = os.path.join(RESULTS_DIR, 'FirstName_LastName_test.tsv')
test[['text', 'label']].to_csv(output_path, sep='\t', index=False)

# =============================
# 🧾 Save metrics to file
# =============================
metrics_path = os.path.join(RESULTS_DIR, 'logreg_metrics.txt')
with open(metrics_path, 'w') as f:
    f.write(f"Best threshold: {best_t:.2f}\n")
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Macro F1: {macro_f1:.4f}\n\n")
    f.write(str(classification_report(y_dev, y_pred)))
    f.write("\n\nFalse Positive Rate by Demographic:\n")
    f.write(str(demo_fpr))

print(f"\n✅ Saved test predictions to {output_path}")
print(f"✅ Metrics saved to {metrics_path}\n")
