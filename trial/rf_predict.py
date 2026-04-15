# import joblib #loading trained ML
# import pandas as pd # tables

# import numpy as np # math

# import warnings # silences annoying warnings
# warnings.filterwarnings('ignore')

# # load the saved models
# models = joblib.load('ML/rf_models.pkl')
# ''' 
# The variable models is actually a dictionary 
# it has one mini-model per thing you want to predict 
# (e.g. one model for tomorrow's min temp, one for rain, etc.)
# '''

# # define today's weather for one location
# today = pd.DataFrame([{
#     'min_temp °c': 6.0,
#     'max_temp °c': 17.0,
#     'rain mm': 0.0,
#     'humidity %': 83.0,
#     'cloud_cover %': 26.0,
#     'wind_speed km/h': 4.0,
#     'wind_direction_numerical': 90.0,
#     'day_of_year': 158,
#     'month': 6,
#     'year': 2025,
#     'location_code': 245
# }])

# # generate predictions for tomorrow
# for target, model in models.items():
#     pred = model.predict(today)[0]
#     print(f"{target}: {pred:.2f}")

# # generate predictions + confidence for tomorrow
# for target, model in models.items():
#     # get prediction from every individual tree
#     tree_predictions = [tree.predict(today)[0] for tree in model.estimators_]
    
#     pred = np.mean(tree_predictions)        # final prediction (average of all trees)
#     std  = np.std(tree_predictions)         # spread of trees = uncertainty
    
#     print(f"{target}:")
#     print(f"  prediction: {pred:.2f}")
#     print(f"  std dev:    {std:.2f}  (lower = more confident)")


# # typical ranges per variable (based on UK weather)
# ranges = {
#     'next_min_temp °c':              20,   # roughly -10 to 30
#     'next_max_temp °c':              25,
#     'next_rain mm':                  20,
#     'next_humidity %':               60,
#     'next_cloud_cover %':            100,
#     'next_wind_speed km/h':          50,
#     'next_wind_direction_numerical': 180
# }

# print("\nConfidence scores:")
# for target, model in models.items():
#     tree_predictions = [tree.predict(today)[0] for tree in model.estimators_]
#     pred = np.mean(tree_predictions)
#     std  = np.std(tree_predictions)
    
#     # normalise: low std = high confidence
#     confidence = max(0, 100 - (std / ranges[target] * 100))
#     print(f"  {target}: {pred:.2f}  →  confidence: {confidence:.0f}%")





# #############################
# # COMPARE DIFF DATES
# #############################
# import pandas as pd
# import numpy as np
# import joblib
# import warnings
# warnings.filterwarnings('ignore')

# models = joblib.load('ML/rf_models.pkl')

# feature_cols = [
#     'min_temp °c', 'max_temp °c', 'rain mm', 'humidity %',
#     'cloud_cover %', 'wind_speed km/h', 'wind_direction_numerical',
#     'day_of_year', 'month', 'year', 'location_code'
# ]

# # same weather, different dates
# base = {
#     'min_temp °c': 6.0,
#     'max_temp °c': 17.0,
#     'rain mm': 0.0,
#     'humidity %': 83.0,
#     'cloud_cover %': 26.0,
#     'wind_speed km/h': 4.0,
#     'wind_direction_numerical': 90.0,
#     'location_code': 245
# }

# april = {**base, 'day_of_year': 105, 'month': 4, 'year': 2025}
# july  = {**base, 'day_of_year': 196, 'month': 7, 'year': 2025}

# april_df = pd.DataFrame([april])[feature_cols]
# july_df  = pd.DataFrame([july])[feature_cols]

# print(f"{'Target':<35} {'April':>10} {'July':>10} {'Difference':>12}")
# print("-" * 70)

# for target, model in models.items():
#     pred_april = model.predict(april_df)[0]
#     pred_july  = model.predict(july_df)[0]
#     diff       = pred_july - pred_april
#     print(f"{target:<35} {pred_april:>10.2f} {pred_july:>10.2f} {diff:>+12.2f}")



# ranges = {
#     'next_min_temp °c':              20,
#     'next_max_temp °c':              25,
#     'next_rain mm':                  20,
#     'next_humidity %':               60,
#     'next_cloud_cover %':            100,
#     'next_wind_speed km/h':          50,
#     'next_wind_direction_numerical': 180
# }

# print(f"\n{'Target':<35} {'April Conf':>12} {'July Conf':>12}")
# print("-" * 60)

# for target, model in models.items():
#     april_trees = [t.predict(april_df)[0] for t in model.estimators_]
#     july_trees  = [t.predict(july_df)[0]  for t in model.estimators_]

#     april_conf = max(0, 100 - (np.std(april_trees) / ranges[target] * 100))
#     july_conf  = max(0, 100 - (np.std(july_trees)  / ranges[target] * 100))

#     print(f"{target:<35} {april_conf:>11.0f}% {july_conf:>11.0f}%")



#############################
# SHAP ON A PREDICTION
#############################
import pandas as pd
import numpy as np
import joblib
import shap
import warnings
warnings.filterwarnings('ignore')

# ── LOAD MODELS ───────────────────────────────────────────────────────────────
models = joblib.load('trial/rf_models.pkl')

feature_cols = [
    'min_temp °c', 'max_temp °c', 'rain mm', 'humidity %',
    'cloud_cover %', 'wind_speed km/h', 'wind_direction_numerical',
    'day_of_year', 'month', 'year', 'location_code'
]

# ── DEFINE A DAY TO EXPLAIN ───────────────────────────────────────────────────
# change these numbers to explore different scenarios
day = pd.DataFrame([{
    'min_temp °c': 6.0,
    'max_temp °c': 17.0,
    'rain mm': 0.0,
    'humidity %': 83.0,
    'cloud_cover %': 26.0,
    'wind_speed km/h': 4.0,
    'wind_direction_numerical': 90.0,
    'day_of_year': 105,   # 105 = April, 196 = July, 355 = December
    'month': 4,
    'year': 2025,
    'location_code': 245
}])[feature_cols]

# ── PICK WHICH TARGET TO EXPLAIN ──────────────────────────────────────────────
# options:
#   'next_min_temp °c'
#   'next_max_temp °c'
#   'next_rain mm'
#   'next_humidity %'
#   'next_cloud_cover %'
#   'next_wind_speed km/h'
#   'next_wind_direction_numerical'

target = 'next_min_temp °c'

# ── RUN SHAP ──────────────────────────────────────────────────────────────────
model       = models[target]
explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(day)

# ── PRINT PREDICTION ──────────────────────────────────────────────────────────
pred = model.predict(day)[0]
base = float(explainer.expected_value)  # what the model predicts on average

print(f"\nTarget: {target}")
print(f"Base prediction (average across all training data): {base:.2f}")
print(f"Actual prediction for this day:                     {pred:.2f}")
print(f"Total SHAP shift from base:                         {pred - base:+.2f}")

# ── PRINT WHY ─────────────────────────────────────────────────────────────────
print(f"\nWhy did the model predict {pred:.2f}?")
print("-" * 60)
for feat, val in sorted(zip(feature_cols, shap_values[0]), key=lambda x: abs(x[1]), reverse=True):
    direction = "pushed UP  ↑" if val > 0 else "pushed DOWN↓"
    print(f"  {feat:<35} {val:+.4f}   {direction}")