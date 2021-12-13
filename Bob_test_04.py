# Bob_test_03.py Test File
# 
# Import helpful libraries
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

# Load the data, and separate the target
# data_path = 'winemag_data.csv' # Original data file
data_path = 'winemag_data_first150k_edited.csv'
home_data = pd.read_csv(data_path)
# wine_reviews = pd.read_csv(data_path)
# print(home_data.shape) # (150930 rows, 11 columns)
# print(home_data.head())

## home_data.groupby(['country', 'province']).head() # .apply(lambda df: df.title.iloc[0])

countries_reviewed = home_data.groupby(['country', 'province']).description.agg([len])
print(countries_reviewed)

# mi = countries_reviewed.index
# print(type(mi))

print("Done!")
quit()
