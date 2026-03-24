"""
XGBoost quantile regression
Input: simulated_rock_glacier_data.csv
Output: predictions on test folds + evaluation metrics
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
import xgboost as xgb

# Load data
df = pd.read_csv('simulated_rock_glacier_data.csv')
feature_cols = [c for c in df.columns if c not in ['velocity', 'polygon_id']]
X = df[feature_cols].values
y = df['velocity'].values
groups = df['climate_domain'].values  # group by climate domain for grouped CV

# Quantiles to predict
quantiles = [0.1, 0.5, 0.9]
pred_all = {q: np.zeros(len(df)) for q in quantiles}

# Grouped 5-fold cross-validation
gkf = GroupKFold(n_splits=5)
for train_idx, test_idx in gkf.split(X, y, groups):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train = y[train_idx]

    for q in quantiles:
        model = xgb.XGBRegressor(
            objective='reg:quantileerror',
            quantile_alpha=q,
            n_estimators=300,
            learning_rate=0.03,
            max_depth=3,
            random_state=42
        )
        model.fit(X_train, y_train)
        pred_all[q][test_idx] = model.predict(X_test)

# Collect results
df['q10'] = pred_all[0.1]
df['q50'] = pred_all[0.5]
df['q90'] = pred_all[0.9]
df['interval_width'] = df['q90'] - df['q10']

# Evaluate median model
mae = mean_absolute_error(df['velocity'], df['q50'])
print(f"Median model MAE: {mae:.2f} cm/yr")
print(f"Average interval width: {df['interval_width'].mean():.2f} cm/yr")

# Summary by climate domain
summary = df.groupby('climate_domain')[['q50', 'interval_width']].median()
print("\nMedian predictions and uncertainty by climate domain:")
print(summary)

# Save results
df.to_csv('xgb_quantile_predictions.csv', index=False)
print("\n✅ Predictions saved: xgb_quantile_predictions.csv")