import pandas as pd
import geopandas as gpd
import numpy as np
from sksurv.util import Surv
from sksurv.ensemble import RandomSurvivalForest
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter

# Load the data
burn_areas = gpd.read_file('fire-ignitions/campania/campania_burn_areas_2007_2021.shp')
climate_data = pd.read_csv('fire-ignitions/campania/weather_data_2010_2019.csv')
climate_data_avg = pd.read_csv('fire-ignitions/campania/weather_data_avg_2010_2019.csv')


# Ensure 'data_inc' column is in datetime format
burn_areas['data_inc'] = pd.to_datetime(burn_areas['data_inc'], format="%d/%m/%Y", errors='coerce')

burn_areas = burn_areas.sort_values(by='data_inc', ascending=True)
# Drop rows with invalid date values (NaT)
burn_areas = burn_areas.dropna(subset=['data_inc'])



# Generate all dates in the range 2010-01-01 to 2019-12-31
dates = pd.date_range(start='2010/01/01', end='2019/12/31', freq='D')
daily_fires = pd.DataFrame({'date': dates})

# Convert fire dates, ensuring correct format
burn_areas['data_inc'] = pd.to_datetime(burn_areas['data_inc'], format="%d/%m/%Y", errors='coerce')
fire_dates = burn_areas['data_inc'].dropna().dt.normalize()

# Mark days with fires (1) and days without (0)
daily_fires['fire_occurred'] = daily_fires['date'].isin(fire_dates).astype(int)

# Compute days since last fire
# Initialize the column with NaN
daily_fires['days_since_last_fire'] = np.nan

# Create a Series of fire dates
fire_dates_series = daily_fires[daily_fires['fire_occurred'] == 1]['date']

# Calculate days since last fire using cumulative fire dates
last_fire_date = None
for i, row in daily_fires.iterrows():
    if row['fire_occurred'] == 1:
        last_fire_date = row['date']
    if last_fire_date is not None:
        daily_fires.at[i, 'days_since_last_fire'] = (row['date'] - last_fire_date).days

# Fill NaN values with 0 to indicate no previous fire recorded
daily_fires['days_since_last_fire'] = daily_fires['days_since_last_fire'].fillna(0)

daily_fires['cumulative_fires'] = daily_fires['fire_occurred'].cumsum()

# Group by cumulative fires and calculate days since last fire
daily_fires['days_since_last_fire'] = daily_fires.groupby('cumulative_fires').cumcount()

# Drop the temporary column
daily_fires.drop(columns=['cumulative_fires'], inplace=True)

# Display result
print(daily_fires)


climate_data = climate_data.iloc[2:].reset_index(drop=True)
climate_data['Date'] = pd.to_datetime(climate_data['Date'], format="%Y-%m-%d", errors='coerce')

# columns for the model
climate_data = climate_data[['Date', 'Temperature', 'Dew_Point', 'Humidity', 'Pressure', 'Wind_Speed', 'Precipitation']]

# merge the weather data with the survival data 
survival_df = pd.merge(daily_fires, climate_data, left_on='date', right_on='Date', how='left')

# 
survival_df.drop(columns=['Date'], inplace=True)

# removing first day of the year
survival_df = survival_df[survival_df['date'].dt.dayofyear != 1]

# list of columns to clean
columns_to_clean = ["Temperature", "Dew_Point", "Humidity", "Pressure", "Wind_Speed", "Precipitation"]

# cleaning the columns
for col in columns_to_clean:
    survival_df[col] = survival_df[col].astype(str).str.split('|').str[0]  # Keep only SI values

# converting the columns to numeric
survival_df[columns_to_clean] = survival_df[columns_to_clean].apply(pd.to_numeric, errors='coerce')

# cleaning the columns
for col in columns_to_clean:
    climate_data[col] = climate_data[col].astype(str).str.split('|').str[0]  # Keep only SI values

# converting the columns to numeric
climate_data[columns_to_clean] = climate_data[columns_to_clean].apply(pd.to_numeric, errors='coerce')

# time-to-event (duration) and event indicator
T = survival_df['days_since_last_fire']
E = survival_df['fire_occurred']

# covariates (features)
covariates = survival_df[['Temperature', 'Humidity', 'Wind_Speed', 'Dew_Point', 'Pressure', 'Precipitation']]

# initialize and fit the model
cph = CoxPHFitter()
cph.fit(survival_df, duration_col='days_since_last_fire', event_col='fire_occurred', formula="Temperature + Humidity + Wind_Speed + Dew_Point + Pressure + Precipitation")

# summary of results
cph.print_summary()