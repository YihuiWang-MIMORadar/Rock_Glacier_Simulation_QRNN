"""
Generate synthetic rock glacier dataset.
Features: terrain, optical, radar, climate
Target: velocity (cm/yr) with heteroscedastic noise
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
n_samples = 5000

# Terrain features
elevation = np.random.uniform(4000, 5500, n_samples)
slope = np.random.uniform(5, 35, n_samples)
aspect_rad = np.random.uniform(0, 2 * np.pi, n_samples)
aspect_sin = np.sin(aspect_rad)
aspect_cos = np.cos(aspect_rad)
curvature = np.random.normal(0, 0.5, n_samples)

# Optical features (Sentinel-2)
ndvi = np.random.beta(0.2, 0.8, n_samples)
ndsi = np.random.beta(0.5, 0.5, n_samples)
swir_ratio = np.random.uniform(0.5, 2.0, n_samples)
texture = np.random.exponential(scale=0.1, size=n_samples)

# Radar features (Sentinel-1)
backscatter_vv = np.random.normal(-12, 3, n_samples)
backscatter_vh = backscatter_vv + np.random.normal(-3, 1.5, n_samples)
coherence = 0.7 - 0.01 * slope + np.random.normal(0, 0.05, n_samples)
coherence = np.clip(coherence, 0.2, 0.95)

# Climate features (ERA5-Land)
t2m_mean = np.random.uniform(-8, 2, n_samples)
precip_sum = np.random.uniform(100, 800, n_samples)
climate_domain = np.random.choice([0, 1], size=n_samples, p=[0.6, 0.4])
t2m_mean += climate_domain * 1.5
precip_sum += climate_domain * 200

# Build feature matrix
X_terrain = np.column_stack([elevation, slope, aspect_sin, aspect_cos, curvature])
X_optical = np.column_stack([ndvi, ndsi, swir_ratio, texture])
X_radar = np.column_stack([backscatter_vv, backscatter_vh, coherence])
X_climate = np.column_stack([t2m_mean, precip_sum, climate_domain])
X = np.column_stack([X_terrain, X_optical, X_radar, X_climate])

feature_names = [
    'elevation', 'slope', 'aspect_sin', 'aspect_cos', 'curvature',
    'ndvi', 'ndsi', 'swir_ratio', 'texture',
    'backscatter_vv', 'backscatter_vh', 'coherence',
    't2m_mean', 'precip_sum', 'climate_domain'
]

# Generate velocity labels (cm/yr)
base_speed = 5.0
beta_slope = 1.2
beta_elev = -0.003
beta_temp = 2.0
beta_precip = 0.01
beta_coh = 10.0
beta_ndvi = -5.0

linear = (base_speed +
          beta_slope * slope +
          beta_elev * elevation +
          beta_temp * t2m_mean +
          beta_precip * (precip_sum / 100) +
          beta_coh * (1 - coherence) +
          beta_ndvi * ndvi)
nonlinear = 0.05 * slope**2
noise_std = 2.0 + 0.1 * linear
noise = np.random.normal(0, noise_std)
velocity = linear + nonlinear + noise
velocity = np.maximum(velocity, 1.0)
velocity += climate_domain * 2.0

# Build DataFrame
df = pd.DataFrame(X, columns=feature_names)
df['velocity'] = velocity
df['polygon_id'] = np.arange(1, n_samples + 1)

df.to_csv('simulated_rock_glacier_data.csv', index=False)
print("✅ Synthetic data generated: simulated_rock_glacier_data.csv")
print(f"Samples: {n_samples}, Features: {len(feature_names)}")