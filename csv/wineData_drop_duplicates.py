# Import helpful libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
 
# Load the data, and separate the target
data_path = 'wineData.csv'
home_data = pd.read_csv(data_path, index_col=0)
 
Dup_Rows = home_data[home_data.duplicated()]
print("\n\nDuplicate Rows : \n {}".format(Dup_Rows))
 
DF_RM_DUP = home_data.drop_duplicates(keep=False)
 
# print('\n\nResult DataFrame after duplicate removal :\n', DF_RM_DUP.head(n=5))
DF_RM_DUP.to_csv('wineData_noDups.csv')
 
# # Create Y
# y = home_data.points 
 
# # Create X 
# features = ['country', 'variety', 'winery']
 
# # Select Columns Corresponding to Feature and Preview Data
# X = home_data[features]
 
# print(X)
