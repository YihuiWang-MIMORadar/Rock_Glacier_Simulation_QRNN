"""
Compare XGBoost and QRNN predictions side by side.
Reads xgb_quantile_predictions.csv and qrnn_predictions.csv,
plots true velocity, predicted median, and 10th–90th percentile interval.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
xgb_df = pd.read_csv('xgb_quantile_predictions.csv')
qrnn_df = pd.read_csv('qrnn_predictions.csv')

# Extract true velocities (XGBoost has 'velocity', QRNN has 'true_velocity')
y_true_xgb = xgb_df['velocity'].values
y_true_qrnn = qrnn_df['true_velocity'].values

# Extract predictions
xgb_q10 = xgb_df['q10'].values
xgb_q50 = xgb_df['q50'].values
xgb_q90 = xgb_df['q90'].values

qrnn_q10 = qrnn_df['q10'].values
qrnn_q50 = qrnn_df['q50'].values
qrnn_q90 = qrnn_df['q90'].values

# Sort by true velocity for better visualization
idx_xgb = np.argsort(y_true_xgb)
idx_qrnn = np.argsort(y_true_qrnn)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Left: XGBoost
x_axis_xgb = np.arange(len(idx_xgb))
ax1.fill_between(x_axis_xgb, xgb_q10[idx_xgb], xgb_q90[idx_xgb], alpha=0.3, color='gray', label='10th–90th percentile')
ax1.plot(x_axis_xgb, y_true_xgb[idx_xgb], 'o', markersize=2, color='blue', label='True velocity')
ax1.plot(x_axis_xgb, xgb_q50[idx_xgb], 'r-', linewidth=1.5, label='Predicted median')
ax1.set_xlabel('Sample (sorted by true velocity)')
ax1.set_ylabel('Velocity (cm/yr)')
ax1.set_title('XGBoost Quantile Regression')
ax1.legend()

# Right: QRNN
x_axis_qrnn = np.arange(len(idx_qrnn))
ax2.fill_between(x_axis_qrnn, qrnn_q10[idx_qrnn], qrnn_q90[idx_qrnn], alpha=0.3, color='gray', label='10th–90th percentile')
ax2.plot(x_axis_qrnn, y_true_qrnn[idx_qrnn], 'o', markersize=2, color='blue', label='True velocity')
ax2.plot(x_axis_qrnn, qrnn_q50[idx_qrnn], 'r-', linewidth=1.5, label='Predicted median')
ax2.set_xlabel('Sample (sorted by true velocity)')
ax2.set_ylabel('Velocity (cm/yr)')
ax2.set_title('QRNN')
ax2.legend()

plt.tight_layout()
plt.savefig('comparison.png', dpi=150)
print("✅ Comparison figure saved as comparison.png")
plt.show()