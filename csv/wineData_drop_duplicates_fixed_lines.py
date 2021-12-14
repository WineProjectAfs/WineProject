#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# wineData_drop_duplicates_fixed_lines.py
#
# Input: wineData_fixed_3_broken_lines.csv
#   Corrected three split lines in original wineData.csv
#   file (one data line split into two rows):
#   Rows 40427, 76769, 80904
#
# Output: wineData_noDups_1.csv
#   Has duplicate rows removed.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Import helpful libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Load the data, and separate the target
data_path = 'wineData_fixed_3_broken_lines.csv'
home_data = pd.read_csv(data_path, index_col=0)
 
# Dup_Rows = home_data[home_data.duplicated()]
# print("\n\nDuplicate Rows : \n {}".format(Dup_Rows))
 
DF_RM_DUP = home_data.drop_duplicates(keep=False)
 
# print('\n\nResult DataFrame after duplicate removal :\n', DF_RM_DUP.head(n=5))
DF_RM_DUP.to_csv('wineData_noDups_1.csv')
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
