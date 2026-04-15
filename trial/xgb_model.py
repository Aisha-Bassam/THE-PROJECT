import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import shap
import joblib

# ── 1. LOAD ──────────────────────────────────────────────────────────────────
df = pd.read_csv('ML/uk_weather_data.csv')

# ── 2. DEFINE FEATURES AND TARGETS ───────────────────────────────────────────
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

X = df[feature_cols]
y = df[target_cols]

# ── 3. SAMPLE FOR SPEED ───────────────────────────────────────────────────────
X_sample = X.sample(n=200000, random_state=42)
y_sample = y.loc[X_sample.index]

# ── 4. TRAIN/TEST SPLIT ───────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_sample, y_sample, test_size=0.2, random_state=42
)

print(f"Training rows: {X_train.shape[0]}")
print(f"Test rows:     {X_test.shape[0]}")

# ── 5. TRAIN ONE XGBOOST PER TARGET ──────────────────────────────────────────
models = {}

for target in target_cols:
    print(f"\nTraining XGBoost for: {target}")
    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        n_jobs=-1,
        random_state=42,
        verbosity=0
    )
    model.fit(X_train, y_train[target])
    models[target] = model

    # quick evaluation
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test[target], preds)
    r2  = r2_score(y_test[target], preds)
    print(f"  MAE: {mae:.3f}  |  R²: {r2:.3f}")

# ── 6. SAVE MODELS ────────────────────────────────────────────────────────────
joblib.dump(models, 'ML/xgb_models.pkl')
print("\nModels saved to ML/xgb_models.pkl")

# ── 7. SHAP ───────────────────────────────────────────────────────────────────
shap_sample = X_test.sample(n=100, random_state=42)

print("\nRunning SHAP...")
shap_results = {}

for target in target_cols:
    print(f"  SHAP for: {target}")
    explainer   = shap.TreeExplainer(models[target])
    shap_values = explainer.shap_values(shap_sample)
    shap_results[target] = {
        'values':   shap_values,
        'expected': explainer.expected_value
    }

# ── 8. PRINT A SINGLE PREDICTION + SHAP EXPLANATION ──────────────────────────
print("\n── EXAMPLE PREDICTION ───────────────────────────────────────────────")
example = shap_sample.iloc[[0]]
print("Input (today's weather):")
print(example.to_string())

print("\nPredictions for tomorrow:")
for target in target_cols:
    pred = models[target].predict(example)[0]
    print(f"  {target}: {pred:.2f}")

print("\nSHAP feature attributions for next_min_temp °c:")
sv = shap_results['next_min_temp °c']['values'][0]
for feat, val in sorted(zip(feature_cols, sv), key=lambda x: abs(x[1]), reverse=True):
    print(f"  {feat:35s} {val:+.4f}")

print("\nDone.")