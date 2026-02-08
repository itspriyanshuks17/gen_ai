# Random Forest in Supervised Learning

## Overview
Random Forest is an ensemble learning method that constructs multiple decision trees and merges their results. It reduces overfitting and improves accuracy.

**Key Concepts:**
- **Bootstrap Sampling**: Random sampling with replacement
- **Feature Randomness**: Random feature subset at each split
- **Voting/Averaging**: Combine predictions from all trees

## Random Forest Architecture

```
┌─────────────────────────────────┐
│         Training Data           │
│     (Original Dataset)          │
└─────────────┬───────────────────┘
              │
              │ Bootstrap Sampling
              │ (Random sampling with replacement)
              ▼
    ┌─────────┼─────────┼─────────┐
    │         │         │         │
┌───▼───┐ ┌───▼───┐ ┌───▼───┐
│Bootstrap│ │Bootstrap│ │Bootstrap│
│Sample 1 │ │Sample 2 │ │Sample 3 │
└───┬────┘ └────┬──┘ └────┬──┘
    │            │         │
    │            │         │ Feature Randomness
    ▼            ▼         ▼ (Random subset of features)
┌───┴────┐ ┌────┴───┐ ┌────┴───┐
│Decision│ │Decision│ │Decision│
│Tree 1  │ │Tree 2  │ │Tree 3  │
│(Random │ │(Random │ │(Random │
│Features│ │Features│ │Features│
└───┬────┘ └────┬──┘ └────┬──┘
    │            │         │
    ▼            ▼         ▼
┌───┴────┐ ┌────┴───┐ ┌────┴───┐
│Pred 1  │ │Pred 2  │ │Pred 3  │
└───┬────┘ └────┬──┘ └────┬──┘
    │            │         │
    └────────────┼─────────┘
                 │
                 ▼
        ┌─────────────────┐
        │ Ensemble Voting │
        │ (Majority/Avg)  │
        └─────────┬───────┘
                  │
                  ▼
        ┌─────────────────┐
        │ Final           │
        │ Prediction      │
        └─────────────────┘
```


```
Random Forest Process:
┌─────────────────┐
│ Input Data      │
└─────┬───────────┘
      │
      v
┌─────────────────┐
│ Create Multiple │
│ Bootstrap       │
│ Samples         │
└─────┬───────────┘
      │
      v
┌─────────────────┐
│ Train Decision  │
│ Tree on each    │
│ sample          │
└─────┬───────────┘
      │
      v
┌─────────────────┐
│ At each split:  │
│ Random feature  │
│ subset          │
└─────┬───────────┘
      │
      v
┌─────────────────┐
│ Grow full trees │
│ without pruning │
└─────┬───────────┘
      │
      v
┌─────────────────┐
│ New data        │
│ prediction      │
└─────┬───────────┘
      │
      v
┌─────────────────┐
│ Each tree       │
│ predicts        │
└─────┬───────────┘
      │
      v
┌─────────────────┐
│ Majority voting │
│ for             │
│ classification  │
└─────┬───────────┘
      │
      v
┌─────────────────┐
│ Final           │
│ prediction      │
└─────────────────┘
```

```python
# Random Forest Classification Example
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report

# Load data
iris = load_iris()
X, y = iris.data, iris.target
feature_names = iris.feature_names
class_names = iris.target_names

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Random Forest
rf_model = RandomForestClassifier(
    n_estimators=100,      # Number of trees
    max_depth=3,          # Maximum depth of trees
    random_state=42,
    n_jobs=-1             # Use all processors
)

rf_model.fit(X_train, y_train)

# Predictions
y_pred = rf_model.predict(X_test)
y_pred_proba = rf_model.predict_proba(X_test)

# Metrics
accuracy = accuracy_score(y_test, y_pred)
print(f"Random Forest Accuracy: {accuracy:.2f}")

# Cross-validation scores
cv_scores = cross_val_score(rf_model, X, y, cv=5)
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean CV score: {cv_scores.mean():.2f} (+/- {cv_scores.std() * 2:.2f})")

# Feature importance
print("\nFeature Importance:")
for name, importance in zip(feature_names, rf_model.feature_importances_):
    print(f"{name}: {importance:.3f}")

# Individual tree analysis
print(f"\nNumber of trees: {len(rf_model.estimators_)}")
print(f"Sample tree depth: {rf_model.estimators_[0].get_depth()}")
```

### Hinglish Explanation
Random Forest ek ensemble learning method hai jo multiple decision trees construct karta hai aur unke results merge karta hai. Yeh overfitting reduce karta hai aur accuracy improve karta hai.

