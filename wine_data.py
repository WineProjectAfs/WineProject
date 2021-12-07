# ***************************************************************************************************************************
#                                                         Imports                                                           #
# ***************************************************************************************************************************
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


# ***************************************************************************************************************************
#                                                        Data Set-Up                                                        #
# ***************************************************************************************************************************
# Load the data, and separate the target

# dataPath = 'winemag_data.csv' # original un-altered dataset with 49 country count
# dataPath = 'wineData.csv' # csv with reduced countries cardinality to 10 and duplicate index column
dataPath = 'wineData10countries.csv'
wineData = pd.read_csv(dataPath)

# Create Y
y = wineData.points 
 
# Create X 
features = ['country', 'variety']
 
# Select Columns Corresponding to Feature and Preview Data
X = wineData[features]


# ***************************************************************************************************************************
#                                                            Notes                                                          #
# ***************************************************************************************************************************
# 1. What I'm trying to accomplish is to grab the 'Country' column, get all unique entries and store in a list
# 2. Then, I want to create a dictionary where the key is the country and the other side is the count per country
# 3. Next, I would like to create a loop at updates the count per country
# 4. Print the list 
# 5. Upon examination, determine which countries are the highest producers (to keep cardinality at 10...selecting only the top ten countries)
# 6. Set all other rows to "other"
# 7. The One-Hot Encoding should work properly, the perspective should be from a larger aggregate, and the MAE should be normal...not close to 2....which it should NOT be


countryData = wineData['country']
uniqueCountries = countryData.unique()
# print(uniqueCountries) # 49 unique countries

countrySeries = countryData.value_counts(ascending=False) 

rankedList = ['US','Italy','France','Spain','Chile','Argentina','Portugal','Australia','New Zealand','Austria','Germany',
              'South Africa','Greece','Israel','Hungary','Canada','Romania','Slovenia','Uruguay','Croatia','Bulgaria',
              'Moldova','Mexico','Turkey','Georgia','Lebanon','Cyprus','Brazil','Macedonia','Serbia','Morocco','England',
              'Luxembourg','Lithuania','India','Czech Republic','Ukraine','Switzerland','South Korea','Bosnia and Herzegovina',
              'China','Egypt','Slovakia','Tunisia','Albania','Montenegro','Japan','US-France']

exclusionList = rankedList[10:]

# wineData['country'] = wineData['country'].replace(exclusionList, 'Other')
# wineData['country'] = wineData['country'].fillna('Other')

# # Replacing the country column with the new country data adds an extra index column "Unnamed: 0", let's drop it
# wineData = wineData.drop('Unnamed: 0', axis='columns')

# wineData.to_csv('C:/Useres/charles.brant-stec/wine_project/csv/wineData.csv') # This dataset contains duplicate index columns
# wineData.to_csv('C:/Users/charles.brant-stec/wine_project/csv/wineData10countries.csv') # dataset with droppped double index

varietyData = wineData['variety']
uniqueVarieites = varietyData.unique()

varietySeries = varietyData.value_counts(ascending=False)
varietyList = varietySeries.index.tolist()
varietyExclusionList = varietyList[10:]
wineData['variety'] = wineData['variety'].replace(varietyExclusionList, 'Other')
wineData = wineData.drop('Unnamed: 0', axis='columns')

wineData.to_csv('C:/Users/charles.brant-stec/wine_project/csv/wineDataNew.csv')




# ***************************************************************************************************************************
#                                       Regression Forest Model w/ Leaf Node MAE Averages                                   #
# ***************************************************************************************************************************
# Split Into Validation and Training Data
X_train, X_valid, Y_train, Y_valid = train_test_split(X, y, random_state=1)
 
# Apply One-Hot Encoder
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[features]))
OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid[features]))
 
# One-hot encoding Removed Index...Put It Back
OH_cols_train.index = X_train.index
OH_cols_valid.index = X_valid.index
 
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

# print(best_tree_size) # 500 best tree size / MAE 2.37


# ***************************************************************************************************************************
#                                                 Random Forest Model                                                       #
# ***************************************************************************************************************************
rf_model = RandomForestRegressor(random_state=0) 
rf_model.fit(OH_cols_train, Y_train)
rf_val_predictions = rf_model.predict(OH_cols_valid)
rf_val_mae = mean_absolute_error(rf_val_predictions, Y_valid)
 
# print(rf_val_mae) # MAE 2.364 with original country cardinality
# print(rf_val_mae) # MAE 2.370 with reduced country cardinality


# ***************************************************************************************************************************
#                                        Working on XGB Regressor Model w/ MAE Averages                                     #
# ***************************************************************************************************************************

xgb_model = XGBRegressor()
xgb_model = XGBRegressor(n_estimators=500,learning_rate=0.05,n_jobs=4)
xgb_model.fit(OH_cols_train, Y_train, 
              early_stopping_rounds=5,
              eval_set=[(OH_cols_valid, Y_valid)],
              verbose=False)
predictions = xgb_model.predict(OH_cols_valid)
# xgb_mae = mean_absolute_error(predictions, Y_valid)
# print(xgb_mae) # MAE 2.39 










