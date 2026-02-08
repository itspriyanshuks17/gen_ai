# Classification in Supervised Learning

## Overview
Classification predicts discrete categorical labels. It assigns data points to predefined classes or categories.

**Types of Classification:**
- **Binary Classification**: Two classes (Yes/No, Spam/Not Spam)
- **Multi-class Classification**: Multiple classes (Digits 0-9, Animal types)
- **Multi-label Classification**: Multiple labels per instance

## Classification Process
Input Features (x1, x2, x3)
           |
           | Decision Function
           v
  Classification Model
           |
           | Class Probabilities
           v
P(Class1), P(Class2), ...
           |
           | argmax(P)
           v
   Predicted Class
           |
           | Cross-Entropy Loss
           v
    Loss Function
           |
           | Update parameters
           v
   Optimization <-------- Training Data (Labeled examples)
           ^
           |
    (Backpropagation)
```

```
Email Classification Flow:
┌─────────────────────────────────┐
│ Email Features                  │
│ (Word count, sender, subject)   │
└─────────────┬───────────────────┘
              │
              v
┌─────────────────────────────────┐
│ Logistic Regression Model       │
└─────────────┬───────────────────┘
              │
              v
┌─────────────────────────────────┐
│ Probability of Spam             │
│ (0.0 to 1.0)                    │
└─────────────┬───────────────────┘
              │
              v
┌─────────────────────────────────┐
│ Threshold >= 0.5?               │
├─────────────┬───────────────────┤
│ Yes: Spam   │   No: Not Spam    │
└─────────────┴───────────────────┘

Decision Boundary:
┌─────────────────────────────────┐
│ Feature Space (2D projection)   │
│                                 │
│           ┌─────────────┐       │
│           │ Decision    │       │
│           │ Line        │       │
│           └─────────────┘       │
│                                 │
│ Class A         │         Class B │
└─────────────────────────────────┘
```

```python
# Comprehensive Classification Example
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Generate sample data
X, y = make_classification(n_samples=1000, n_features=2, n_informative=2,
                          n_redundant=0, n_clusters_per_class=1, random_state=42)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

# Metrics
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Sample predictions
print("\nSample Predictions:")
for i in range(5):
    print(f"True: {y_test[i]}, Predicted: {y_pred[i]}, Probability: {y_pred_proba[i][1]:.3f}")
```

### Hinglish Explanation
Classification discrete categorical labels predict karta hai. Yeh data points ko predefined classes ya categories mein assign karta hai.

