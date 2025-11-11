# Regression Metrics: mean_squared_error, mean_absolute_error, r2_score (and NumPy usage)

This document explains the two import lines shown in your notebook and how to use them when evaluating regression models.

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
```

## What these imports provide

- `mean_squared_error(y_true, y_pred)`
  - Computes the average of the squared differences between true and predicted values: \(\frac{1}{n}\sum_i (y_i - \hat{y}_i)^2\).
  - Output is in the squared units of `y`.

- `mean_absolute_error(y_true, y_pred)`
  - Computes the average absolute difference: \(\frac{1}{n}\sum_i |y_i - \hat{y}_i|\).
  - Output is in the same units as `y` and is easier to interpret than MSE.

- `r2_score(y_true, y_pred)`
  - Coefficient of determination. Measures how well predictions approximate the true values.
  - R² = 1 is perfect, 0 means the model predicts no better than using the mean of `y_true`, and values can be negative when the model is worse than that baseline.

- `numpy` (imported as `np`)
  - Used here primarily to compute the square root of MSE (to get RMSE) and for array manipulations.

## Common pattern: RMSE from mean_squared_error

Root Mean Squared Error (RMSE) is commonly reported because it has the same units as the target variable. You compute it using NumPy's `sqrt` on the MSE:

```python
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
```

Or in one line:

```python
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
```

## Example (typical usage with scaled data)

If you scaled the target during training (e.g., with `MinMaxScaler`), you must inverse-transform both the true values and the predictions back to the original scale before computing the metrics. Example pattern from your notebook:

```python
# assume scaler is a fitted MinMaxScaler for the target
# y_test and predictions are scaled arrays of shape (n_samples, 1)

y_test_actual = scaler.inverse_transform(y_test)
predictions_actual = scaler.inverse_transform(predictions)

rmse = np.sqrt(mean_squared_error(y_test_actual, predictions_actual))
mae = mean_absolute_error(y_test_actual, predictions_actual)
r2 = r2_score(y_test_actual, predictions_actual)

print(f"RMSE: {rmse:.2f}")
print(f"MAE:  {mae:.2f}")
print(f"R² Score: {r2:.4f}")
```

Note: If your arrays have shape `(n_samples, 1)` sklearn metric functions accept them, but you may also `ravel()` to 1-D arrays:

```python
rmse = np.sqrt(mean_squared_error(y_test_actual.ravel(), predictions_actual.ravel()))
```

## Interpretation

- RMSE (lower better): sensitive to large errors because of the squaring.
- MAE (lower better): robust to outliers relative to RMSE.
- R² (higher better): proportion of variance explained by the model. Negative values indicate poor model performance.

Use multiple metrics together — for example, MAE + RMSE + R² — to get a fuller picture.

## Edge cases and gotchas

- Shapes must match exactly: `y_true.shape == y_pred.shape`. Mismatched shapes raise exceptions.
- Missing values: metrics do not handle NaNs; drop or impute NaNs before calling the functions.
- Constant `y_true`: R² is not meaningful when `y_true` has zero variance (all identical values). scikit-learn may return `nan` or raise a warning/behaviour depending on version.
- Scaling: Always inverse-transform target values when computing metrics if you trained on scaled targets.
- Multioutput targets: these metric functions accept multioutput arrays; check sklearn docs for aggregation behavior ("uniform_average" by default).

## Quick check-list before computing metrics

- [ ] Do `y_true` and `y_pred` have the same length and compatible shapes?
- [ ] Are there NaNs or infinities? Clean them first.
- [ ] If data was scaled, did you inverse-transform both arrays back to original units?
- [ ] Use `.ravel()` if you prefer 1-D arrays for printing/formatting.

## Dependencies

- numpy
- scikit-learn

Install with pip if needed:

```powershell
pip install numpy scikit-learn
```

---
