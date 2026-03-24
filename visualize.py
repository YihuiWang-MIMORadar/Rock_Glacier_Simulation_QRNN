"""
Plot true velocity vs predicted intervals
Reads either xgb_quantile_predictions.csv or qrnn_predictions.csv
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Choose which result file to plot (modify as needed)
file = 'qrnn_predictions.csv'   # or 'qrnn_predictions.csv'

print(f"Loading data from {file}...")
df = pd.read_csv(file)

# Determine true velocity column
if 'true_velocity' in df.columns:
    y_true = df['true_velocity'].values
elif 'velocity' in df.columns:
    y_true = df['velocity'].values
else:
    print("Error: true velocity column not found")
    exit()

# Check if required prediction columns exist
for col in ['q10', 'q50', 'q90']:
    if col not in df.columns:
        print(f"Error: column {col} not found in {file}")
        exit()

q10 = df['q10'].values
q50 = df['q50'].values
q90 = df['q90'].values

# Sort by true velocity for a cleaner plot
idx = np.argsort(y_true)
y_true_sorted = y_true[idx]
q10_sorted = q10[idx]
q50_sorted = q50[idx]
q90_sorted = q90[idx]

plt.figure(figsize=(10, 6))
x_axis = np.arange(len(y_true_sorted))
plt.fill_between(x_axis, q10_sorted, q90_sorted, alpha=0.3, label='10th–90th percentile')
plt.plot(x_axis, y_true_sorted, 'o', markersize=2, label='True velocity')
plt.plot(x_axis, q50_sorted, 'r-', label='Predicted median')
plt.xlabel('Sample (sorted by true velocity)')
plt.ylabel('Velocity (cm/yr)')
plt.title('Rock Glacier Velocity Prediction with Uncertainty Intervals')
plt.legend()
plt.tight_layout()
plt.savefig('velocity_uncertainty.png', dpi=150)
print("✅ Figure saved: velocity_uncertainty.png")
plt.show()