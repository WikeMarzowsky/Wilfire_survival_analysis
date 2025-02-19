import pandas as pd
import geopandas as gpd
import numpy as np
import fiona
import matplotlib.pyplot as plt
from tqdm import tqdm


# Load the data
burn_areas_campania = gpd.read_file('fire-ignitions/campania/campania_burn_areas_2007_2021.shp')
# with fiona.open('fire-ignitions/mediterranean/EFIS_BurntAreas_2008-2023.shp', encoding='MacRoman') as src:
#     burn_areas_mediterranean = gpd.GeoDataFrame.from_features(src)
# burn_areas_mediterranean = gpd.read_file('fire-ignitions/mediterranean/EFIS_BurntAreas_2008-2023.shp')
burn_areas_crete = pd.read_excel('fire-ignitions/crete/fires_crete_2000_2020.xlsx')


burn_areas_campania["data_inc"] = pd.to_datetime(burn_areas_campania["data_inc"], errors="coerce", dayfirst=True)
burn_areas_campania["year"] = burn_areas_campania["data_inc"].dt.year

burn_areas_campania = burn_areas_campania.dropna(subset=["data_inc"])
burn_areas_campania = burn_areas_campania.sort_values(by="year", ascending=True)

burn_areas_campania = burn_areas_campania[burn_areas_campania["year"] >= 2007]

# Count occurrences of each year
fires_per_year = burn_areas_campania["year"].value_counts().sort_index()

# Plot the trend
plt.figure(figsize=(10, 5))
plt.plot(fires_per_year.index, fires_per_year.values, marker="o", linestyle="-", color="red", label="Fires per year")

# Customize the plot
plt.xlabel("Year")
plt.ylabel("Number of Fires")
plt.title("Annual Fire Trends in Campania")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.show()


burn_areas_campania.loc[:, "area_hectares"] = burn_areas_campania.geometry.area / 10_000

# print(burn_areas_campania[["geometry", "area_hectares"]].head())  # Check results


burnt_area_per_year = burn_areas_campania.groupby("year")["area_hectares"].sum()

plt.figure(figsize=(12, 6))
plt.bar(burnt_area_per_year.index, burnt_area_per_year.values, color="darkred", alpha=0.8)

plt.xlabel("Year")
plt.ylabel("Total Burnt Area (hectares)")
plt.title("Total Burnt Area per Year in Campania (Calculated from Geometry)")
plt.xticks(rotation=45)
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.show()


# Plot the burned area polygons
fig, ax = plt.subplots(figsize=(10, 10))
burn_areas_campania.plot(ax=ax, color="orange", edgecolor="black", alpha=0.5)

ax.set_title("Burned Areas in Campania (2007-2021)")
plt.show()

burn_areas = burn_areas_campania.sort_values(by="data_inc")

# Load Campania region shapefile
campania_areas = gpd.read_file("fire-ignitions/campania/campania_region.shp")

unburned_records = []


for idx, row in tqdm(burn_areas.iterrows(), total=len(burn_areas), desc="Processing Unburned Areas"):
    burned_geometry = row.geometry
    date = row["data_inc"]

    # get unburned area
    unburned_area = gpd.overlay(campania_areas, gpd.GeoDataFrame(geometry=[burned_geometry]), how="difference")

    # result
    unburned_records.append({
        "date": date,
        "burned_geometry": burned_geometry,
        "unburned_geometry": unburned_area.geometry.union_all(),
    })

# GeoDataFrame
unburned_areas_df = gpd.GeoDataFrame(unburned_records, columns=["date", "burned_geometry", "unburned_geometry"])
unburned_areas_df.set_geometry("unburned_geometry", inplace=True)

# save in file
unburned_areas_df.to_file("unburned_areas_by_date.shp", driver="ESRI Shapefile")


