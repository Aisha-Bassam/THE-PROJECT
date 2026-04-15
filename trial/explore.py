import pandas as pd

df = pd.read_csv('ML/all_weather_data.csv')
# print(df.shape) # gives you (rows, columns)
# print(df.columns.tolist()) # lists all variable/column names.
# print(df['location'].unique()) # confirming unique locations
# print(df.head())

# # 1. data types and missing values
# print(df.dtypes)
# print("\nMissing values:")
# print(df.isnull().sum())

# # 2. date range
# df['date'] = pd.to_datetime(df['date'])
# print("\nDate range:", df['date'].min(), "to", df['date'].max())

# # 3. quick check on numeric columns
# print("\nNumeric summary:")
# print(df.describe())


# UK locations to exclude (non-UK identified in your data)
non_uk = [
    'Madrid', 'Oslo', 'Prague', 'Milan', 'Athens', 'Munich', 'Berlin',
    'Barcelona', 'Budapest', 'Rome', 'Stockholm', 'Amsterdam', 'Paris',
    'Warsaw', 'Lisbon', 'New York', 'Hamberg', 'Abengourou', 'Palermo'
]

# Dublin, Cork are Ireland — your call, keep or drop
# I'd keep them, close enough climatically and not far off

df_uk = df[~df['location'].isin(non_uk)].copy()
print("UK rows:", df_uk.shape[0])
print("Locations remaining:", df_uk['location'].nunique())

# Parse date and extract features for RF
df_uk['date'] = pd.to_datetime(df_uk['date'])
df_uk['day_of_year'] = df_uk['date'].dt.dayofyear
df_uk['month'] = df_uk['date'].dt.month
df_uk['year'] = df_uk['date'].dt.year

# Encode location as numeric (RF needs numbers)
df_uk['location_code'] = pd.factorize(df_uk['location'])[0]

print(df_uk.head())

df_uk.to_csv('ML/uk_weather_data.csv', index=False)