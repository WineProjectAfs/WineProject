from itertools import count
from os import O_NOINHERIT
from typing import Counter
from numpy.core.fromnumeric import var
import pandas as pd
import numpy as np
from pandas.core.algorithms import rank
from pandas.core.frame import DataFrame
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeRegressor
from collections import OrderedDict

from data_manipulation.input_manipulation import *


# Data Set-Up
dataPath = 'source_csv/winemag_data.csv' 
wineData = setUpData(dataPath)

# Data Manipulation: Removes Wines That Score <= 85 Points -> Reduces Cardinality of 'Variety' and 'Countries' -> Outputs .csv and Returns New DataFrame
wineData = dataManipulation(wineData)
# wineData.to_csv('data_output_csv/wineDataOutputTest.csv')

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
# rf_model = RandomForestRegressor(random_state=0) 
# rf_model.fit(OH_cols_train, Y_train)
# rf_val_predictions = rf_model.predict(OH_cols_valid)
# rf_val_mae = mean_absolute_error(rf_val_predictions, Y_valid)

# # Output CSV
# outputCSV(wineData, rf_val_predictions, "randomForestModelOutput", Y_valid)
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
# xgb_model = XGBRegressor()
# xgb_model = XGBRegressor(n_estimators=500,learning_rate=0.05,n_jobs=4)
# xgb_model.fit(OH_cols_train, Y_train, 
#               early_stopping_rounds=5,
#               eval_set=[(OH_cols_valid, Y_valid)],
#               verbose=False)
# xgbPredictions = xgb_model.predict(OH_cols_valid)
# xgb_mae = mean_absolute_error(xgbPredictions, Y_valid)

# # Output CSV
# outputCSV(wineData, xgbPredictions, "xgbRegressorOutput2", Y_valid)
# End ######################################################################################################################

# test = pd.read_csv('data_output_csv/wineDataOutPut.csv')
# count = test[test.points < 85].sum()
# print(count)

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

# Testing out dropping entries with < 85 points

# allpoints = {}
# droppedpoints = {}
# point_data = pd.read_csv('data_output_csv/pointDropped.csv')

# for country in wineData['country'].unique():
#     str = f'{country}_df'
#     locals()[str] = wineData[wineData.country == country]
#     point_avg = locals()[str]['points'].mean()
#     allpoints.update({str:round(point_avg,2)})

# for country in point_data['country'].unique():
#     str = f'{country}_df'
#     locals()[str] = point_data[point_data.country == country]
#     point_avg = locals()[str]['points'].mean()
#     droppedpoints.update({str:round(point_avg,2)})

# print(allpoints)
# print(droppedpoints)

# output = pd.read_csv('data_output_csv/xgbRegressorOutput.csv')
# output2 = pd.read_csv('data_output_csv/xgbRegressorOutput2.csv')
# print(output['0'].mean())
# print(output2['0'].mean())

# This loop generates a dataframe for each unique country then stores those dataframes in a list
country_list = []

for country in wineData['country'].unique():
    country_list.append(f'{country}_df')

def countryFrame(country):
    globals()[str] = wineData[wineData.country == country]
    country_list.append(globals()[str])

for country in wineData['country'].unique():
    countryFrame(country)

US_df = wineData[wineData.country == 'US']
Spain_df = wineData[wineData.country == 'Spain']
France_df = wineData[wineData.country == 'France']
Italy_df = wineData[wineData.country == 'Italy']
New_Zealand_df = wineData[wineData.country == 'New Zealand']
Other_df = wineData[wineData.country == 'Other']
Argentina_df = wineData[wineData.country == 'Argentina']
Australia_df = wineData[wineData.country == 'Australia']
Portugal_df = wineData[wineData.country == 'Portugal']
Chile_df = wineData[wineData.country == 'Chile']
Austria_df = wineData[wineData.country == 'Austria']
countries_list = [US_df,Spain_df,France_df,Italy_df,New_Zealand_df,Other_df,
                  Argentina_df,Australia_df,Portugal_df]

# print(country_list)
# print(US_df)
# print(France_df)

allpoints = {}

for country in wineData['country'].unique():
    str = f'{country}_df'
    locals()[str] = wineData[wineData.country == country]
    point_avg = locals()[str]['points'].mean()
    allpoints.update({str:round(point_avg,2)})

print(sorted(allpoints.items(), key=lambda x: x[1], reverse=True))

# sorted_keys = sorted(dict1, key=dict1.get)  # [1, 3, 2]

# for w in sorted_keys:
#     sorted_dict[w] = dict1[w]

# print(output['0'].mean())
# print(allpoints)
# Average predicted score by country
# {'US_df': 87.91, 'Spain_df': 86.76, 'France_df': 88.92, 'Italy_df': 88.39,
#  'New Zealand_df': 87.64, 'Other_df': 87.56, 'Argentina_df': 86.08,
#  'Australia_df': 87.96, 'Portugal_df': 88.12, 'Chile_df': 86.28, 'Austria_df': 89.38}

# print(US_df['province'].unique().size)
# print(Spain_df['province'].unique().size)
# print(France_df['province'].unique().size)
# print(Italy_df['province'].unique().size)
# print(New_Zealand_df['province'].unique().size)
# print(Other_df['province'].unique().size)
# print(Argentina_df['province'].unique().size)
# print(Australia_df['province'].unique().size)
# print(Portugal_df['province'].unique().size)
# print(Chile_df['province'].unique().size)
# print(Austria_df['province'].unique().size)

# print(US_df['province'].unique().size)
# print(Spain_df['province'].unique().size)
# print(France_df['province'].unique().size)
# print(Italy_df['province'].unique().size)
# print(New_Zealand_df['province'].unique().size)
# print(Other_df['province'].unique().size)
# print(Argentina_df['province'].unique().size)
# print(Australia_df['province'].unique().size)
# print(Portugal_df['province'].unique().size)
# print(Chile_df['province'].unique().size)
# print(Austria_df['province'].unique().size)

# print(US_df['province'].unique())

# print(Austria_df['province'].unique())