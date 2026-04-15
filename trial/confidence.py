import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')
# ── 1. LOAD ───────────────────────────────────────────────────────────────────
df = pd.read_csv('trial/uk_weather_data.csv')
models = joblib.load('trial/rf_models.pkl')

feature_cols = [
    'min_temp °c', 'max_temp °c', 'rain mm', 'humidity %',
    'cloud_cover %', 'wind_speed km/h', 'wind_direction_numerical',
    'day_of_year', 'month', 'year', 'location_code'
]

target_cols = [
    'next_min_temp °c', 'next_max_temp °c', 'next_rain mm',
    'next_humidity %', 'next_cloud_cover %',
    'next_wind_speed km/h', 'next_wind_direction_numerical'
]

# ── 2. SAMPLE SOME ROWS ───────────────────────────────────────────────────────
sample = df.sample(n=1000, random_state=42)
X_sample = sample[feature_cols]
y_actual = sample[target_cols]

# ── 3. PREDICT + CONFIDENCE PER COLUMN ───────────────────────────────────────
predictions = {}
confidences = {}

for target in target_cols:
    model = models[target]
    
    # get predictions from each tree
    tree_preds = np.array([tree.predict(X_sample) for tree in model.estimators_])
    
    # mean prediction
    predictions[target] = tree_preds.mean(axis=0)
    
    # std dev across trees → raw confidence signal
    std = tree_preds.std(axis=0)
    
    # normalise to 0-100 (lower std = higher confidence)
    max_std = std.max()
    confidences[target] = 100 * (1 - std / max_std)

# ── 4. CALCULATE LABEL CONFIDENCE THREE WAYS ─────────────────────────────────
conf_df = pd.DataFrame(confidences)

conf_df['average']  = conf_df.mean(axis=1)
conf_df['minimum']  = conf_df.min(axis=1)
conf_df['geometric'] = conf_df.apply(
    lambda row: np.exp(np.log(row.clip(0.01)).mean()), axis=1
)

# ── 5. CALCULATE ACTUAL ERROR PER ROW ────────────────────────────────────────
errors = {}
for target in target_cols:
    errors[target] = np.abs(
        np.array(predictions[target]) - y_actual[target].values
    )

error_df = pd.DataFrame(errors)
conf_df['actual_mean_error'] = error_df.mean(axis=1)

# ── 6. COMPARE CONFIDENCE METHODS AGAINST ACTUAL ERROR ───────────────────────
print("Correlation with actual error (higher = better calibrated):")
for method in ['average', 'minimum', 'geometric']:
    # we expect high confidence to correlate with LOW error
    # so we flip the sign
    corr = conf_df[method].corr(-conf_df['actual_mean_error'])
    print(f"  {method:12s}: {corr:.4f}")