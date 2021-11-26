import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
 
# Data Explanation / Exploration 

data_path = 'winemag_data.csv'
home_data = pd.read_csv(data_path)
home_data.describe()