# Bob_test_05.py
# from: winemag_scaling_test.py
# 
# Import helpful libraries
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

# for Box-Cox Transformation
from scipy import stats

# for min_max scaling
# import pypi
from mlxtend.preprocessing import minmax_scaling
# from pypi import minmax_scaling
# from sklearn.preprocessing import MinMaxScaler

# plotting modules
import seaborn as sns
import matplotlib.pyplot as plt

# set seed for reproducibility
np.random.seed(0)
#-------------------------------------------------------

# Load the data, and separate the target
data_path = 'winemag_data.csv' # Original data file
# data_path = 'winemag_data_first150k_edited.csv' # Edited to fix three split data rows.
home_data = pd.read_csv(data_path)
home_data_points = home_data.points

max_points = home_data_points.max()
min_points = home_data_points.min()

print(f"Max Points: {max_points} \nMin Points: {min_points}")

print(home_data.groupby('points').points.count() )

print()
#-------------------------------------------------------
# mix-max scale the data between 0 and 1
scaled_data = minmax_scaling(home_data_points, columns=[0])

#
# Note: 'distplot' is deprecated.
#       Need to replace it with 'displot' or 'histplot'.
#
# # plot both together to compare
fig, ax = plt.subplots(1,2)
sns.distplot(home_data_points, ax=ax[0])
ax[0].set_title("Original Data")
sns.distplot(scaled_data, ax=ax[1])
ax[1].set_title("Scaled data")
#
# sns.displot(data=home_data, x='points', kind='hist', legend=True)
### plt.subplots(1,2)
## plt.subplot(121)
## sns.kdeplot(data=home_data, x='points')
## plt.subplot(122)
## sns.kdeplot(data=scaled_data, x='points')

a = home_data_points
b = scaled_data
g = sns.displot(a)
g.map_dataframe(sns.histplot, b, color='red')


print("Done!")
quit()
