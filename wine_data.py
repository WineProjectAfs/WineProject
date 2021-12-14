# ***************************************************************************************************************************
#                                                         Imports                                                           #
# ***************************************************************************************************************************
from numpy.core.fromnumeric import var
import pandas as pd
import numpy as np
from pandas.core.algorithms import duplicated, rank
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
# Load the data, and separate the target
dataPath = 'csv/winemag_data.csv'
wineData = pd.read_csv(dataPath)

print(wineData[wineData.duplicated(subset='description',keep='first')])


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


# countryData = wineData['country']
# uniqueCountries = countryData.unique()
# # print(uniqueCountries) # 49 unique countries

# countrySeries = countryData.value_counts(ascending=False) 

# rankedList = ['US','Italy','France','Spain','Chile','Argentina','Portugal','Australia','New Zealand','Austria','Germany',
#               'South Africa','Greece','Israel','Hungary','Canada','Romania','Slovenia','Uruguay','Croatia','Bulgaria',
#               'Moldova','Mexico','Turkey','Georgia','Lebanon','Cyprus','Brazil','Macedonia','Serbia','Morocco','England',
#               'Luxembourg','Lithuania','India','Czech Republic','Ukraine','Switzerland','South Korea','Bosnia and Herzegovina',
#               'China','Egypt'
#               ,'Slovakia','Tunisia','Albania','Montenegro','Japan','US-France']

# exclusionList = rankedList[10:]

# wineData['country'] = wineData['country'].replace(exclusionList, 'Other')
# wineData['country'] = wineData['country'].fillna('Other')

# # Replacing the country column with the new country data adds an extra index column "Unnamed: 0", let's drop it
# wineData = wineData.drop('Unnamed: 0', axis='columns')

# # reducing variety cardinality as we did with countries
# varietyData = wineData['variety']
# uniqueVarieites = varietyData.unique()

# varietySeries = varietyData.value_counts(ascending=False)
# varietyList = varietySeries.index.tolist()
# varietyExclusionList = varietyList[10:]
# wineData['variety'] = wineData['variety'].replace(varietyExclusionList, 'Other')

# # wineData.to_csv('csv/wineData.csv')


