from numpy.core.fromnumeric import var
import pandas as pd
import numpy as np
from pandas.core.algorithms import rank
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeRegressor

dataPath = 'csv/wineDataNew.csv' # This dataset has reduced country and variety cardinality
wineData = pd.read_csv(dataPath)

y = wineData.points
features = ['country','variety']
X = wineData[features]

# Split Into Validation and Training Data
X_train, X_valid, Y_train, Y_valid = train_test_split(X, y, random_state=1)
 
# Apply One-Hot Encoder
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[features]))
OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid[features]))
 
# One-hot encoding Removed Index...Put It Back
OH_cols_train.index = X_train.index
OH_cols_valid.index = X_valid.index

# ***************************************************************************************************************************
#                                                 Random Forest Model                                                       #
# ***************************************************************************************************************************

rf_model = RandomForestRegressor(random_state=0) 
rf_model.fit(OH_cols_train, Y_train)
rf_val_predictions = rf_model.predict(OH_cols_valid)
rf_val_mae = mean_absolute_error(rf_val_predictions, Y_valid)
 
# print(rf_val_mae) # MAE 2.364 with original country cardinality
# print(rf_val_mae) # MAE 2.370 with reduced country cardinality
print(rf_val_mae) # MAE of 2.465 with reduced country and variety cardinality


