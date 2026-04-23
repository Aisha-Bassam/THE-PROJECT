import pandas as pd

# ── 1. LOAD ──────────────────────────────────────────────────────────────────
df = pd.read_csv('trial/all_weather_data.csv')

# ── 2. FILTER TO UK ──────────────────────────────────────────────────────────
non_uk = [
    'Madrid', 'Oslo', 'Prague', 'Milan', 'Athens', 'Munich', 'Berlin',
    'Barcelona', 'Budapest', 'Rome', 'Stockholm', 'Amsterdam', 'Paris',
    'Warsaw', 'Lisbon', 'New York', 'Hamberg', 'Abengourou', 'Palermo'
]
df_uk = df[~df['location'].isin(non_uk)].copy()

# ── 3. PARSE DATE + EXTRACT FEATURES ─────────────────────────────────────────
df_uk['date'] = pd.to_datetime(df_uk['date'])
df_uk['day_of_year'] = df_uk['date'].dt.dayofyear
df_uk['month']       = df_uk['date'].dt.month
df_uk['year']        = df_uk['date'].dt.year

# ── 4. ENCODE LOCATION ───────────────────────────────────────────────────────
df_uk['location_code'] = pd.factorize(df_uk['location'])[0]

# ── 5. SORT + CREATE TOMORROW'S TARGETS ──────────────────────────────────────
df_uk = df_uk.sort_values(['location', 'date']).reset_index(drop=True)

target_cols = ['min_temp °c', 'max_temp °c', 'rain mm', 'humidity %',
               'cloud_cover %', 'wind_speed km/h', 'wind_direction_numerical']

for col in target_cols:
    df_uk[f'next_{col}'] = df_uk.groupby('location')[col].shift(-1)

# Drop last day per location (no tomorrow exists)
df_uk = df_uk.dropna(subset=[f'next_{col}' for col in target_cols])

# ── 6. SAVE ──────────────────────────────────────────────────────────────────
df_uk.to_csv('data/uk_weather_data.csv', index=False)

# ── 7. SANITY CHECK ──────────────────────────────────────────────────────────
print("Rows:", df_uk.shape[0])
print("Columns:", df_uk.columns.tolist())
print("\nSample:")
print(df_uk[['location', 'date', 'min_temp °c', 'next_min_temp °c']].head(8))