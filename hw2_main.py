'''
1684_HW2 Homework - Baseline Toxicity Classifier
Author: Jonah Smith
Course: 1684
Description: Baseline PerspectiveAPI rule-based classifier for offensive language detection.
'''

import os
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# =============================
# 🔧 Setup Paths
# =============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'civility_data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

os.makedirs(RESULTS_DIR, exist_ok=True)

# =============================
# Load Data
# =============================
print("\n[1] Loading data...")

dev_path = os.path.join(DATA_DIR, 'dev.tsv')
demo_dev_path = os.path.join(DATA_DIR, 'mini_demographic_dev.tsv')

dev = pd.read_csv(dev_path, sep='\t')
demo_dev = pd.read_csv(demo_dev_path, sep='\t')

print(f"Loaded {len(dev)} dev samples and {len(demo_dev)} demographic dev samples.")

# =============================
# Rule-based Classification
# =============================
print("\n[2] Running PerspectiveAPI threshold baseline...")

THRESHOLD = 0.8
dev['pred_label'] = dev['perspective_score'].apply(lambda x: 'OFF' if x > THRESHOLD else 'NOT')

y_true = dev['label']
y_pred = dev['pred_label']

# =============================
# Evaluation
# =============================
print("\n[3] Evaluating performance metrics...")

accuracy = accuracy_score(y_true, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None, labels=['OFF', 'NOT'])

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision (OFF, NOT): {precision}")
print(f"Recall (OFF, NOT): {recall}")
print(f"F1 (OFF, NOT): {f1}")

# =============================
# Demographic False Positive Rate (FPR)
# =============================
print("\n[4] Computing False Positive Rate (FPR) per demographic group...")

demo_dev['pred_label'] = demo_dev['perspective_score'].apply(lambda x: 'OFF' if x > THRESHOLD else 'NOT')
# FPR = predicted OFF / total, since true label = NOT for all
demo_fpr = demo_dev.groupby('demographic')['pred_label'].apply(lambda x: (x == 'OFF').mean())

print("\nFalse Positive Rate by Demographic:")
print(demo_fpr)

# =============================
# Save Results
# =============================
metrics_path = os.path.join(RESULTS_DIR, 'baseline_metrics.txt')

with open(metrics_path, 'w') as f:
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Precision (OFF, NOT): {precision}\n")
    f.write(f"Recall (OFF, NOT): {recall}\n")
    f.write(f"F1 (OFF, NOT): {f1}\n\n")
    f.write("False Positive Rate by Demographic:\n")
    f.write(str(demo_fpr))

print(f"\nResults saved to: {metrics_path}")
print("\nNext step → Implement your Logistic Regression model (custom_logreg.py).\n")
