"""
Quantile Regression Neural Network (QRNN) training and evaluation
Note: tensorflow must be installed
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import tensorflow as tf
from tensorflow.keras import layers, models, Input, backend as K

# Load data
df = pd.read_csv('simulated_rock_glacier_data.csv')
feature_cols = [c for c in df.columns if c not in ['velocity', 'polygon_id']]
X = df[feature_cols].values
y = df['velocity'].values

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Quantile loss function
def quantile_loss(q):
    def loss(y_true, y_pred):
        error = y_true - y_pred
        return K.mean(K.maximum(q * error, (q - 1) * error), axis=-1)
    return loss

quantiles = [0.1, 0.5, 0.9]
y_pred_q = {}

for q in quantiles:
    print(f"\nTraining QRNN for quantile {q}")
    # Use Input layer to avoid warning
    inputs = Input(shape=(X_train.shape[1],))
    x = layers.Dense(64, activation='relu')(inputs)
    x = layers.Dense(32, activation='relu')(x)
    outputs = layers.Dense(1)(x)
    model = models.Model(inputs=inputs, outputs=outputs)
    
    model.compile(optimizer='adam', loss=quantile_loss(q))
    # Set verbose=1 to see progress
    model.fit(X_train, y_train, epochs=50, batch_size=64, verbose=1)
    y_pred_q[q] = model.predict(X_test).flatten()

# Evaluate
mae = mean_absolute_error(y_test, y_pred_q[0.5])
print(f"\nMedian model MAE: {mae:.2f} cm/yr")

# Build results DataFrame
results = pd.DataFrame({
    'true_velocity': y_test,
    'q10': y_pred_q[0.1],
    'q50': y_pred_q[0.5],
    'q90': y_pred_q[0.9]
})
results['interval_width'] = results['q90'] - results['q10']
print(f"Average interval width: {results['interval_width'].mean():.2f} cm/yr")
results.to_csv('qrnn_predictions.csv', index=False)
print("✅ Predictions saved: qrnn_predictions.csv")