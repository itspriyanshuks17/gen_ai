# Error Metrics in Supervised Learning

## Overview
Error metrics are evaluation measures used to assess the performance of supervised learning models, particularly in regression tasks like time series forecasting.

## Common Error Metrics

### 1. Mean Absolute Error (MAE)
- Average absolute differences between predictions and actual values
- Formula: MAE = Sum|yi - y_pred_i| / n
- Range: [0, ∞), lower is better

### 2. Mean Squared Error (MSE)
- Average squared differences between predictions and actual values
- Formula: MSE = Sum(yi - y_pred_i)^2 / n
- Penalizes large errors more heavily

3. **Root Mean Squared Error (RMSE)**
   - Square root of MSE
   - Formula: RMSE = sqrt(Sum(yi - y_pred_i)^2 / n)
   - Same units as original data

4. **Mean Absolute Percentage Error (MAPE)**
   - Percentage error relative to actual values
   - Formula: MAPE = (100/n) × Sum|(yi - y_pred_i)/yi|
   - Useful for comparing across different scales

```
Actual Values (y1, y2, ..., yn) ──┐
                                   │
Predicted Values (y_pred1, y_pred2, ..., y_predn) ──┐
                                   │
                                   └─ Errors (ei = yi - y_pred_i)
                                        │
                                        ├─── MAE: Sum(abs(ei)) / n
                                        │
                                        ├─── MSE: Sum((ei)^2) / n
                                        │
                                        ├─── RMSE: sqrt(MSE)
                                        │
                                        └─── MAPE: 100 * Sum(abs(ei/yi)) / n
                                             │
                                             └─ Error Metrics
```

```
Prediction Errors Scale:
Perfect Prediction (Error = 0) ──► Small Errors (MAE < 1) ──► Medium Errors (MAE 1-5) ──► Large Errors (MAE > 5)

Error Distribution Analysis:
┌─────────────────────────────────┐
│ Residual Plot                   │
│ (Errors vs Predictions)         │
├─────────────────────────────────┤
│ Q-Q Plot                        │
│ (Normality Check)               │
├─────────────────────────────────┤
│ Error Histogram                 │
│ (Distribution Shape)            │
└─────────────────────────────────┘
```

```python
# Sequence Error Metrics Example
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

def calculate_sequence_errors(y_true, y_pred):
    """
    Calculate various error metrics for sequence predictions
    """
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Basic metrics
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    # MAPE (avoid division by zero)
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    # Additional metrics
    max_error = np.max(np.abs(y_true - y_pred))
    median_abs_error = np.median(np.abs(y_true - y_pred))

    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'MAPE': mape,
        'Max_Error': max_error,
        'Median_Abs_Error': median_abs_error
    }

# Example usage
# Simulated time series data
np.random.seed(42)
true_values = np.sin(np.linspace(0, 4*np.pi, 100)) + np.random.normal(0, 0.1, 100)
predicted_values = true_values + np.random.normal(0, 0.2, 100)  # Add some noise

# Calculate errors
errors = calculate_sequence_errors(true_values, predicted_values)

print("Sequence Error Metrics:")
for metric, value in errors.items():
    print(f"{metric}: {value:.4f}")

# Error analysis
residuals = true_values - predicted_values
print(f"\nResidual Statistics:")
print(f"Mean residual: {np.mean(residuals):.4f}")
print(f"Std residual: {np.std(residuals):.4f}")
print(f"Residual range: [{np.min(residuals):.4f}, {np.max(residuals):.4f}]")

# Plotting concepts (would create visualizations)
# plt.figure(figsize=(12, 4))
# plt.subplot(1, 3, 1)
# plt.plot(true_values, label='True')
# plt.plot(predicted_values, label='Predicted')
# plt.title('Predictions vs True Values')
# plt.legend()

# plt.subplot(1, 3, 2)
# plt.scatter(predicted_values, residuals)
# plt.xlabel('Predicted Values')
# plt.ylabel('Residuals')
# plt.title('Residual Plot')

# plt.subplot(1, 3, 3)
# plt.hist(residuals, bins=20)
# plt.xlabel('Residual Value')
# plt.ylabel('Frequency')
# plt.title('Error Distribution')

# plt.tight_layout()
# plt.show()
```

### Hinglish Explanation
Sequence error evaluation metrics hai jo models ki performance measure karte hain, especially sequence prediction tasks mein jaise time series forecasting ya NLP.

