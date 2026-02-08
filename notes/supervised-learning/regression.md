# Regression in Supervised Learning

## Overview
Regression predicts continuous numerical values. It finds the relationship between variables to predict outcomes like prices, temperatures, or quantities.

**Types of Regression:**
- **Linear Regression**: Simple straight-line relationship
- **Polynomial Regression**: Curved relationships
- **Ridge/Lasso Regression**: Regularized linear regression

## Regression Process

```
┌─────────────────┐
│ Input Features  │
│ (x₁, x₂, x₃)    │
└─────────┬───────┘
          │
          │ f(x) = w₁*x₁ + w₂*x₂ + ... + b
          ▼
┌─────────────────┐
│ Regression      │
│ Model           │
└─────────┬───────┘
          │
          │ Continuous Output
          ▼
┌─────────────────┐
│ ŷ (Predicted    │
│ Value)          │
└─────────┬───────┘
          │
          │ MSE = Σ((y - ŷ)²)/n
          ▼
┌─────────────────┐
│ Loss Function   │
└─────────┬───────┘
          │
          │ Update weights
          ▼
┌─────────────────┐    ┌─────────────────┐
│ Gradient        │◄───│ Training Data   │
│ Descent         │    │ (Labeled        │
│                 │    │ examples)       │
│ Backpropagation │    └─────────────────┘
└─────────────────┘
```
<!-- ``` -->

```
┌─────────────────┐
│ House Size      │
│ (sq ft)         │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ Linear          │
│ Regression      │
│ Model           │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ Predicted Price │
│ ($)             │
└─────────────────┘

Sample Training Data:
• (1000 sq ft, $200k)
• (1500 sq ft, $280k)
• (2000 sq ft, $350k)
```

```python
# Comprehensive Regression Example
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Generate sample data
np.random.seed(42)
X = np.random.rand(100, 1) * 10
y = 2.5 * X.flatten() + 1.5 + np.random.randn(100) * 2

# Train model
model = LinearRegression()
model.fit(X, y)

# Predictions
y_pred = model.predict(X)

# Metrics
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")
print(f"Slope: {model.coef_[0]:.2f}")
print(f"Intercept: {model.intercept_:.2f}")

# Plotting (conceptual)
# plt.scatter(X, y, color='blue', label='Actual')
# plt.plot(X, y_pred, color='red', label='Predicted')
# plt.xlabel('X values')
# plt.ylabel('Y values')
# plt.legend()
# plt.show()
```

### Hinglish Explanation
Regression continuous numerical values predict karta hai. Yeh variables ke beech relationship dhundhta hai prices, temperatures, ya quantities jaise outcomes predict karne ke liye.

## Classification
Classification predicts discrete categorical labels. It assigns data points to predefined classes or categories.

**Types of Classification:**
- **Binary Classification**: Two classes (Yes/No, Spam/Not Spam)
- **Multi-class Classification**: Multiple classes (Digits 0-9, Animal types)
- **Multi-label Classification**: Multiple labels per instance

```
Input Features (x1, x2, x3)
           |
           | Decision Function
           v
  Classification Model
