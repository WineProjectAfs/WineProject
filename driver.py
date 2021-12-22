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
# xgb_model = XGBRegressor()
# xgb_model = XGBRegressor(n_estimators=500,learning_rate=0.05,n_jobs=4)
# xgb_model.fit(OH_cols_train, Y_train, 
#               early_stopping_rounds=5,
#               eval_set=[(OH_cols_valid, Y_valid)],
#               verbose=False)
# xgbPredictions = xgb_model.predict(OH_cols_valid)
# xgb_mae = mean_absolute_error(xgbPredictions, Y_valid)

# # Output CSV
# outputCSV(wineData, xgbPredictions, "xgbRegressorOutput", Y_valid)
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

# ***************************************************************************************************************************
#                                        Evaluating dropping < 85 point entries                                             #
# ***************************************************************************************************************************
# # Testing out dropping entries with < 85 points
# # dataset with predictions
# predicted_data = pd.read_csv('data_output_csv/xgbRegressorOutput.csv')

# # predicted points average
# predicted_points = {}

# # actual points average
# all_points = {}

# for country in predicted_data['country'].unique():
#     df = predicted_data[predicted_data.country == country]
#     point_avg = df['0'].mean()
#     predicted_points.update({country:round(point_avg,2)})

# for country in wineData['country'].unique():
#     df = wineData[wineData.country == country]
#     point_avg = df['points'].mean()
#     all_points.update({country:round(point_avg,2)})

# # # sort actual and predicted points dictionaries
# # all_points = sorted(all_points.items(), key=lambda x: x[1], reverse=True)
# # predicted_points = sorted(predicted_points.items(), key=lambda x: x[1], reverse=True)

# print(all_points)
# print(predicted_points)


# # Average actual score by country
# # {'US': 87.91, 'Spain': 86.76, 'France': 88.92, 'Italy': 88.39, 'New Zealand': 87.64,
# #  'Other': 87.56, 'Argentina': 86.08, 'Australia': 87.96, 'Portugal': 88.12, 'Chile': 86.28, 'Austria': 89.38}

# # Average predicted score by country
# # {'US': 87.96, 'Spain': 87.96, 'France': 87.93, 'Italy': 87.94, 'New Zealand': 88.07,
# #  'Other': 87.98, 'Argentina': 87.96, 'Australia': 87.98, 'Portugal': 87.91, 'Chile': 87.93, 'Austria': 87.99}

# # Dropping < 85 scores skews the data towards countries with more reviews and against countries with less
# ***************************************************************************************************************************

# ***************************************************************************************************************************
#                                      Grabbing Top Wine & Province Recommendations                                         #
# ***************************************************************************************************************************
# # Grabbing Top Wines
# # predicted data
# predicted_data = pd.read_csv('data_output_csv/xgbRegressorOutput.csv')

# variety_avgs = {}

# for variety in predicted_data['variety'].unique():
#     df = predicted_data[predicted_data['variety'] == variety]
#     point_avg = df['0'].mean()
#     variety_avgs.update({variety:round(point_avg,2)})

# variety_rank = sorted(variety_avgs.items(), key=lambda x: x[1], reverse=True)

# # Vartieties & their predicted scores
# # {'Cabernet Sauvignon': 87.96, 'Other': 87.95, 'Sauvignon Blanc': 87.92, 'Pinot Noir': 88.0, 'Chardonnay': 87.93, 'Syrah': 87.88,
# #  'Red Blend': 87.91, 'Riesling': 87.99, 'Zinfandel': 87.91, 'Bordeaux-style Red Blend': 87.96, 'Merlot': 87.98}

# # Varieties ranked by score
# print(variety_rank)
# # [('Pinot Noir', 88.0), ('Riesling', 87.99), ('Merlot', 87.98), ('Cabernet Sauvignon', 87.96), ('Bordeaux-style Red Blend', 87.96),
# #  ('Other', 87.95), ('Chardonnay', 87.93), ('Sauvignon Blanc', 87.92), ('Red Blend', 87.91), ('Zinfandel', 87.91), ('Syrah', 87.88)]

# # Pinot Noir & Riesling are the 2 top scoring wines
# # ***************************************************************************************************************************

# # This function grabs and ranks the top scoring provinces to grow the given variety 

# def topProvinces(variety):
#     predicted_data = pd.read_csv('data_output_csv/xgbRegressorOutput.csv')
#     province_scores ={}
#     variety_data = predicted_data[predicted_data['variety'] == variety]
#     province_data = variety_data['province']
#     province_series = province_data.value_counts(ascending=False)
#     province_list = province_series.index.tolist()
#     top_provinces = province_list[:10]

#     for province in top_provinces:
#         score = variety_data[variety_data.province == province]
#         predicted_avg = score['0'].mean()
#         province_scores.update({province:round(predicted_avg,2)})

#     sorted_provinces = sorted(province_scores.items(), key=lambda x: x[1], reverse=True)
#     print(sorted_provinces)

# print('Pinot Noir - Top Provinces:')
# print(topProvinces('Pinot Noir'))
# print('Bordeaux-style Red Blend - Top Provinces:')
# print(topProvinces('Bordeaux-style Red Blend'))


