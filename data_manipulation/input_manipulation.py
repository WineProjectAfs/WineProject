# ***************************************************************************************************************************
#                                                         Imports                                                           #
# ***************************************************************************************************************************
from numpy.core.fromnumeric import var
import pandas as pd
import numpy as np
from pandas.core.algorithms import rank, duplicated
from scipy.sparse import data
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

# Creates Dataframe
def setUpData(dataPath): 
    # Load the data, and separate the target
    wineData = pd.read_csv(dataPath)
    return wineData

# Creates Predicted Feature
def predictedFeature(wineData):
    # Create Predicted Feature
    predictedFeature = wineData.points 
    return predictedFeature

# Creates Predicting Features
def predictingFeatures(wineData, features):
    X = wineData[features]
    return X 


# ***************************************************************************************************************************
#                                                       Data Manipulation                                                   #
# ***************************************************************************************************************************

# Reduces Cardinality of 'Country' and 'Variety'
def dataManipulation(wineData):
    # ************************************************Remove Duplicates******************************************************
    wineData['is_duplicate'] = wineData.duplicated()
    wineData['is_duplicate'] = wineData.duplicated()
    wineData = wineData.drop_duplicates(subset='description')
    wineData.reset_index(drop=True,inplace=True)
    wineData = wineData.drop('is_duplicate',axis=1)
    
    # ************************************************Remove Wines Less than 85 Points***************************************
    # wineData = wineData[wineData.points >= 85]

    # **************************Remove Countries Outside of Top 10 Most Occurring and Label as 'Other'***********************
    countryData = wineData['country']
    uniqueCountries = countryData.unique()
    # # 49 Unique Countries
    # print(uniqueCountries) 

    # Get Counts For Each Country
    countrySeries = countryData.value_counts(ascending=False) 

    # Ranked List of Countries
    rankedList = ['US','Italy','France','Spain','Chile','Argentina','Portugal','Australia','New Zealand','Austria','Germany',
                'South Africa','Greece','Israel','Hungary','Canada','Romania','Slovenia','Uruguay','Croatia','Bulgaria',
                'Moldova','Mexico','Turkey','Georgia','Lebanon','Cyprus','Brazil','Macedonia','Serbia','Morocco','England',
                'Luxembourg','Lithuania','India','Czech Republic','Ukraine','Switzerland','South Korea','Bosnia and Herzegovina',
                'China','Egypt','Slovakia','Tunisia','Albania','Montenegro','Japan','US-France']

    #Grab Value Counts per Unique Entry -> Turn Into List -> Take Top 10 -> Replace Anything Not In Top 10
    exclusionList = rankedList[10:]
    wineData['country'] = wineData['country'].replace(exclusionList, 'Other')
    wineData['country'] = wineData['country'].fillna('Other')

    # Replacing the Country Column w/ the New Country Data Adds Index Column "Unnamed: 0" -> Dropped
    wineData = wineData.drop('Unnamed: 0', axis='columns')


    # ************************************************Remove Countries Outside of Top 10 Most Occurring and Label as 'Other'****************************************************
    # Get Column and Unique Entries
    varietyData = wineData['variety']
    uniqueVarieites = varietyData.unique()

    # Grab Value Counts per Unique Entry -> Turn Into List -> Take Top 10 -> Replace Anything Not In Top 10 w/ Other
    varietySeries = varietyData.value_counts(ascending=False)
    varietyList = varietySeries.index.tolist()
    varietyExclusionList = varietyList[10:]
    wineData['variety'] = wineData['variety'].replace(varietyExclusionList, 'Other')
    wineData['variety'] = wineData['variety'].fillna('Other')

    return wineData


# ***************************************************************************************************************************
#                                                       Results Output                                                      #
# ***************************************************************************************************************************

# Takes Original DataFrame, Predictions, and Model Name -> Outputs a .csv With Predictions Concatenated Based On Predicted Feature (Points)
def outputCSV(originalDataFrame, predictions, modelName, validationData):
    validationData['predictions'] = predictions
    validPredictions = pd.DataFrame(validationData['predictions'])
    modelPredictions = pd.merge(originalDataFrame, validPredictions, how = 'left', left_index = True, right_index = True)
    modelPredictions.to_csv(f"csv/{modelName}.csv")