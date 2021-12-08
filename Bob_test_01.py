# Bob_test_01.py Test File
# 
# Import helpful libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
 
# Load the data, and separate the target
# data_path = 'winemag_data.csv' # Original data file
data_path = 'winemag_data_first150k_edited.csv'
home_data = pd.read_csv(data_path)
 
# Create Y
y = home_data.points 
 
# Create X 
features = ['country', 'variety']
 
# Select Columns Corresponding to Feature and Preview Data
X = home_data[features]
 
# Split Into Validation and Training Data
X_train, X_valid, Y_train, Y_valid = train_test_split(X, y, random_state=1)
 
# Apply One-Hot Encoder
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[features]))
OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid[features]))
 
# # One-hot encoding Removed Index...Put It Back
OH_cols_train.index = X_train.index
OH_cols_valid.index = X_valid.index
 
# Define a Random Forest Model
rf_model = RandomForestRegressor(random_state=0) 
rf_model.fit(OH_cols_train, Y_train)
rf_val_predictions = rf_model.predict(OH_cols_valid)
rf_val_mae = mean_absolute_error(rf_val_predictions, Y_valid)
 
print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))
