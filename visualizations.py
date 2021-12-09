from numpy.core.fromnumeric import var
import pandas as pd
import numpy as np
from pandas.core.algorithms import rank
import matplotlib.pyplot as plt
import seaborn as sns
from seaborn.matrix import heatmap
import itertools

data_path = 'csv/wineData.csv'
wine_data = pd.read_csv(data_path, index_col='country')

# Uncomment Plots you would like to show and comment those you do not 

# ***************************************************************************************************************************
#                                      Bar Plot for Average Score/Country                                                   #
# ***************************************************************************************************************************
# plt.figure(figsize=(10,6))
# plt.ylim(80,95)
# plt.title('Average Wine Score by Country')
# sns.barplot(x = wine_data.index, y = wine_data['points'])
# plt.ylabel('Average Score')
# plt.xlabel('Countries')
# plt.show()

# plt.savefig('graphs/BarPlot.png') # Save our graph

# ***************************************************************************************************************************
#                                      Scatter Plot for Average Score/Country                                               #
# ***************************************************************************************************************************
# plt.title('Scores by Country')
# sns.scatterplot(x=wine_data.index, y=wine_data['points'])
# plt.show()

# plt.savefig('graphs/scatterPlot.png') 

# Only US, France, Italy, and Australia have wines that have scored 100
# Scores of 100 may not be of great significance as for 100 there are only 24 entries?

# points_series = wine_data.points.value_counts().sort_index(ascending=False)
# print(points_series)
# 100       24
# 99        50
# 98       131
# 97       365
# 96       695
# 95      1716
# 94      3462
# 93      6017
# 92      9241
# 91     10536
# 90     15973

# High scores are >5% of the dataset, there is no practicality in doubting their accuracy

# ***************************************************************************************************************************
#                                      Heat Map for Average Score/Country                                                   #
# ***************************************************************************************************************************
# THIS IS NOT WORKING - DATASET NEED TO BE REFORMATTED FOR HEATMAP TO WORK
# THE DATA NEEDS TO BE REFORMATTED TO COUNTRY AS INDEX, POINTS AS COLUMNS, VALUES AS COUNTRY/POINT VALUE COUNTS

# plt.figure(figsize=(14,7))
# plt.title('Scores by Country')
# x = wine_data['country']
# y = wine_data['points']
# sns.heatmap(x,y)
# plt.show()

heat_data = pd.read_csv(data_path)
points = ['80','81','82','83','84','85','86','87','88','89','90','91','92','93','94','95','96','97','98','99','100'] # index
countries = ['US','Spain','France','Italy','New Zealand','Argentina','Australia','Portugal','Chile','Austria','Other Countries'] 
hdf = heat_data.groupby(['points','country']).size().to_frame()

hdf.to_csv('csv/hdf.csv')
d_pth='csv/hdf.csv'
hdfd=pd.read_csv(d_pth)
print(hdfd.loc[hdfd['country'] == 'US'])
US_pts = (hdfd.loc[hdfd['country'] == 'US']).iloc[:,2].to_list()

print(US_pts)
