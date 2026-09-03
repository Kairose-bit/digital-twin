import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# 1. Generate dummy evaluation metrics & confusion matrix data
y_true = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
y_pred = [0, 0, 0, 0, 1, 1, 1, 1, 0, 1]

cm = confusion_matrix(y_true, y_pred)

# 2. Create and save confusion_matrix.png
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('confusion_matrix.png')
plt.close()

# 3. Create and save feature_importance.png
features = ['EGT', 'RPM', 'Oil Pressure', 'MAP']
importance = [0.45, 0.25, 0.20, 0.10]

plt.figure(figsize=(6, 5))
plt.barh(features, importance, color='teal')
plt.title('Feature Importance')
plt.xlabel('Importance Score')
plt.savefig('feature_importance.png')
plt.close()

# 4. Create and save metrics_summary.json
metrics = {
    "accuracy": 0.85,
    "precision": 0.80,
    "recall": 0.80,
    "f1_score": 0.80
}

with open('metrics_summary.json', 'w') as f:
    json.dump(metrics, f, indent=4)

print("Successfully generated confusion_matrix.png, feature_importance.png, and metrics_summary.json!")