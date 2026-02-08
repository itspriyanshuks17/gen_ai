# Decision Trees in Supervised Learning

## Overview
Decision trees are flowchart-like structures where each internal node represents a feature test, each branch represents an outcome, and each leaf node represents a class label.

**How Decision Trees Work:**
1. **Root Node**: Best feature to split on
2. **Internal Nodes**: Feature tests with branches
3. **Leaf Nodes**: Final class predictions
4. **Splitting Criteria**: Gini impurity, Information gain, etc.

## Decision Tree Structure

```
                    ┌─────────────┐
                    │ Root Node   │
                    │ Feature: Age│
                    └─────┬───────┘
                          │
                ┌─────────┴─────────┐
                │                   │
          ┌─────▼─────┐       ┌─────▼─────┐
          │ Age ≤ 30  │       │ Age > 30  │
          └─────┬─────┘       └─────┬─────┘
                │                   │
        ┌───────┴───────┐   ┌───────┴───────┐
        │               │   │               │
    ┌───▼───┐       ┌───▼───┐   ┌───▼───┐       ┌───▼───┐
    │Student│       │Credit │   │Income │       │Income │
    │?      │       │Rating │   │> 50k? │       │≤ 50k? │
    └───────┘       └───────┘   └───────┘       └───────┘
        │                   │           │                   │
   ┌────┴────┐       ┌──────┴─────┐   ┌──┴──┐       ┌─────┴─────┐
   │         │       │            │   │     │       │           │
┌──▼──┐  ┌──▼──┐  ┌──▼──┐  ┌────▼────┐ ┌─▼─┐ ┌───▼───┐ ┌────▼────┐ ┌──▼──┐
│High │  │Low  │  │Low  │  │Medium    │ │Low│ │High   │ │Medium    │ │High │
│Risk │  │Risk │  │Risk │  │Risk      │ │Risk│ │Risk   │ │Risk      │ │Risk │
└─────┘  └─────┘  └─────┘  └──────────┘ └────┘ └───────┘ └──────────┘ └─────┘
```

```

Decision Tree Algorithm:
┌─────────────────────────────────┐
│ Start with all training data    │
└─────────────┬───────────────────┘
              │
              v
┌─────────────────────────────────┐
│ Select best feature to split    │
└─────────────┬───────────────────┘
              │
              v
┌─────────────────────────────────┐
│ Calculate impurity reduction    │
│ (Gini/Entropy)                  │
└─────────────┬───────────────────┘
              │
              v
┌─────────────────────────────────┐
│ Create decision node            │
└─────────────┬───────────────────┘
              │
              v
┌─────────────────────────────────┐
│ Split data into subsets         │
└─────────────┬───────────────────┘
              │
              v
┌─────────────────────────────────┐
│ All subsets pure or             │
│ max depth reached?              │
├─────────────┬───────────────────┤
│ No          │         Yes       │
└─────┬───────┘         └─────┬────┘
      │                       │
      v                       v
┌─────────────┐     ┌─────────────────┐
│ Repeat for  │     │ Create leaf     │
│ each subset │     │ nodes with      │
│             │     │ class labels    │
│     ↑       │     └─────────────────┘
│     │       │
└─────┼───────┘
      │
      └─────────────────────────────┘

```

```python
# Decision Tree Classification Example
import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# Load data
iris = load_iris()
X, y = iris.data, iris.target
feature_names = iris.feature_names
class_names = iris.target_names

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Decision Tree
dt_model = DecisionTreeClassifier(max_depth=3, random_state=42)
dt_model.fit(X_train, y_train)

# Predictions
y_pred = dt_model.predict(X_test)

# Metrics
accuracy = accuracy_score(y_test, y_pred)
print(f"Decision Tree Accuracy: {accuracy:.2f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Feature importance
print("\nFeature Importance:")
for name, importance in zip(feature_names, dt_model.feature_importances_):
    print(f"{name}: {importance:.3f}")

# Visualize tree (conceptual)
# plt.figure(figsize=(20,10))
# plot_tree(dt_model, feature_names=feature_names, class_names=class_names,
#           filled=True, rounded=True)
# plt.show()
```

### Hinglish Explanation
Decision trees flowchart-like structures hain jismein har internal node ek feature test represent karta hai, har branch ek outcome, aur har leaf node ek class label.

