import pandas as pd

df = pd.read_csv("data/uk_weather_data.csv")

# Basic shape
print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())

# Stats for every numeric column
print("\n--- describe() ---")
print(df.describe())

# Check for nulls
print("\n--- Nulls per column ---")
print(df.isnull().sum())

# Peek at the data
print("\n--- First 5 rows ---")
print(df.head())