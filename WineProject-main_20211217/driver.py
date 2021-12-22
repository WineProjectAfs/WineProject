from os import O_NOINHERIT
from numpy.core.fromnumeric import var
import pandas as pd
import numpy as np
from pandas.core.algorithms import rank
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeRegressor

from data_manipulation.input_manipulation import *

test = 'test'

# Data Set-Up
dataPath = 'source_csv/winemag_data.csv' 
wineData = setUpData(dataPath)

# Data Manipulation: Reduces Cardinality of 'Variety' and 'Countries' -> Outputs .csv and Returns New DataFrame
wineData = dataManipulation(wineData)
wineData.to_csv('data_output_csv/wineDataOutput.csv')

features = ['country', 'variety']
y = predictedFeature(wineData)
X = predictingFeatures(wineData, features)

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

# Uncomment All Between Lines  to Print .csv With Predictions For Random Forest Model #
# Beginning ###############################################################################################################
rf_model = RandomForestRegressor(random_state=0) 
rf_model.fit(OH_cols_train, Y_train)
rf_val_predictions = rf_model.predict(OH_cols_valid)
rf_val_mae = mean_absolute_error(rf_val_predictions, Y_valid)

# Output CSV
outputCSV(wineData, rf_val_predictions, "randomForestModelOutput", Y_valid)
# End ######################################################################################################################

# # Type Check
# print(type(wineData))
# print(type(Y_valid['predictions']))

# print(rf_val_predictions.tostring)

# Mean Average Error Results
# print(rf_val_mae) 
# MAE 2.364 w/ Reduced Country Cardinality
# MAE 2.370 w/ Reduced Country Cardinality
# MAE 2.465 w/ Reduced Country and Variety Cardinality
# MAE 2.472 w/ Duplicates Dropped
# print(rf_val_predictions)
# ***************************************************************************************************************************


# ***************************************************************************************************************************
#                                       Regression Forest Model w/ Leaf Node MAE Averages                                   #
# ***************************************************************************************************************************
# # Create Test Leaf Nodes
# candidate_max_leaf_nodes = [5, 10, 50, 100, 250, 500]

# # Functions
# def get_mae(max_leaf_nodes, OH_cols_train, OH_cols_valid, Y_train, Y_valid):
#     model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
#     model.fit(OH_cols_train, Y_train)
#     preds_val = model.predict(OH_cols_valid)
#     mae = mean_absolute_error(Y_valid, preds_val)
#     print(mae)
#     return(mae)

# scores = {leaf_size: get_mae(leaf_size, OH_cols_train, OH_cols_valid, Y_train, Y_valid) for leaf_size in candidate_max_leaf_nodes}

# best_tree_size = min(scores, key=scores.get)
# print(best_tree_size)  

# Best Tree Specifications: 500 best tree size | MAE 2.37
# ***************************************************************************************************************************


# ***************************************************************************************************************************
#                                        XGB Regressor Model w/ MAE Averages                                                #
# ***************************************************************************************************************************

# Uncomment All Between Lines  to Print .csv With Predictions For XGB Regressor Model #
# Beginning ###############################################################################################################
xgb_model = XGBRegressor()
xgb_model = XGBRegressor(n_estimators=500,learning_rate=0.05,n_jobs=4)
xgb_model.fit(OH_cols_train, Y_train, 
              early_stopping_rounds=5,
              eval_set=[(OH_cols_valid, Y_valid)],
              verbose=False)
xgbPredictions = xgb_model.predict(OH_cols_valid)
xgb_mae = mean_absolute_error(xgbPredictions, Y_valid)

# Output CSV
outputCSV(wineData, xgbPredictions, "xgbRegressorOutput", Y_valid)
# End ######################################################################################################################

# # Type Check
# print(type(wineData))
# print(type(Y_valid['preds']))

# Mean Average Error Results
# print(xgb_mae)
# MAE 2.39: w/ Reduced Country Cardinality Only
# MAE 2.466 w/ Reduced Country and Variety Cardinality | Runtime Greatly Reduced
# MAE 2.473 w/ Duplicates Dropped
# print(xgbPredictions)
# xgbDF = pd.read_csv('data_output_csv/xgbRegressorOutput.csv')
# print(xgbDF['0'].mean())      # 87.953 average predicted score - first 24455
# print(xgbDF['points'].mean()) # 87.957 average score
# ***************************************************************************************************************************

