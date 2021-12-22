
# ***************************************************************************************************************************
#                                                   Imports                                                                 #
# ***************************************************************************************************************************
from numpy.core.fromnumeric import var
import pandas as pd
import numpy as np
from pandas.core.algorithms import rank
import matplotlib.pyplot as plt
import seaborn as sns
from seaborn.matrix import heatmap
import itertools

data_path = 'data_output_csv/wineDataOutput.csv'
wine_data = pd.read_csv(data_path, index_col='country')
variety_data = pd.read_csv(data_path, index_col='variety')
point_data = pd.read_csv('data_output_csv/pointDropped.csv',index_col='country')

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

<<<<<<< HEAD
=======
# plt.figure(figsize=(10,6))
# plt.ylim(80,95)
# plt.title('Average Wine Score by Country')
# sns.barplot(x = point_data.index, y = point_data['points'])
# plt.ylabel('Average Score')
# plt.xlabel('Countries')
# plt.show()

>>>>>>> build
# plt.savefig('graphs/BarPlot.png') # Save our graph


# ***************************************************************************************************************************
#                                      Bar Plot for Average Score/Variety                                                   #
# ***************************************************************************************************************************
# plt.figure(figsize=(10,6))
# plt.ylim(80,100)
# plt.title('Average Wine Score by Variety')
# variety_plot = sns.barplot(x=variety_data.index,y=variety_data['points'])
# plt.ylabel('Average Score')
# plt.xlabel('Varieties')
# variety_plot.set_xticklabels(variety_plot.get_xticklabels(), rotation=40, ha='right')
# plt.tight_layout
# plt.show()

# # predicted data
# predicted_data = pd.read_csv('data_output_csv/xgbRegressorOutput.csv')
# plt.figure(figsize=(10,6))
# plt.ylim(80,100)
# plt.title('Average Wine Score by Variety')
# variety_plot = sns.barplot(x=predicted_data['variety'],y=predicted_data['points'])
# plt.ylabel('Predicted')
# plt.xlabel('Varieties')
# variety_plot.set_xticklabels(variety_plot.get_xticklabels(), rotation=40, ha='right')
# plt.tight_layout
# plt.show()


# ***************************************************************************************************************************
#                                      Scatter Plot for Average Score/Country                                               #
# ***************************************************************************************************************************
# plt.title('Scores by Country')
# sns.scatterplot(x=wine_data.index, y=wine_data['points'])
# plt.show()

# plt.savefig('graphs/scatterPlot.png') 

# Only US, France, Italy, and Australia have wines that have scored 100
# Scores of 100 may not be of great significance as for 100 there are only 24 entries?

# ***************************************************************************************************************************
#                                      Heat Map for Average Score/Country                                                   #
# ***************************************************************************************************************************
# THE DATA NEEDS TO BE REFORMATTED TO COUNTRY AS COLUMNS, POINTS AS INDEC, VALUES AS COUNTRY/POINT VALUE COUNTS

# heat_data = pd.read_csv(data_path)
# points = ['80','81','82','83','84','85','86','87','88','89','90','91','92','93','94','95','96','97','98','99','100'] # index
# points.reverse()
# countries = ['US','Spain','France','Italy','New Zealand','Argentina','Australia','Portugal','Chile','Austria','Other'] 
# hdf = heat_data.groupby(['points','country']).size().to_frame()

# # hdf.to_csv('csv/hdf.csv')
# d_pth='csv/hdf.csv'
# hdfd=pd.read_csv(d_pth)

# heat_df = pd.DataFrame(columns=countries)
# heat_df['points'] = points
# heat_df = heat_df.set_index('points',drop=True)

# countries = ['US','Spain','France','Italy','New Zealand','Argentina','Australia','Portugal','Chile','Austria','Other'] 
# points = ['80','81','82','83','84','85','86','87','88','89','90','91','92','93','94','95','96','97','98','99','100'] # index
# points1 = [80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100]

# # This loop adds 0 country/point value counts to the dataframe
# for point in points1:
#     pts = hdf.loc[[point]].reset_index()
#     pts_list = pts['country'].to_list()
#     for country in countries:
#         if pts_list.count(country) == 0:
#             hdfd.loc[len(hdfd.index)] = [point,country,0]

# hdfd = hdfd.set_index('points',drop=True)
# hdfd = hdfd.sort_index(ascending=False)
# print(hdfd.to_string())

# # This loop populates are new dataframe with value counts per country for 80-100
# for country in heat_df.columns:
#     country_pts = (hdfd.loc[hdfd['country'] == country]).iloc[:,1].to_list()
#     heat_df[country] = country_pts

# # print(heat_df.to_string)
# # heat_df.to_csv('csv/heatmap_data.csv')

# heatmap_filepath = 'csv/heatmap_data.csv'
# heatmap_data = pd.read_csv(heatmap_filepath, index_col='points')

# plt.figure(figsize=(14,7))

# plt.title('Wine Score Frequency by Country')

# # light heatmap
# # sns.heatmap(data=heatmap_data, annot=True,cmap='YlGnBu')
# # dark heatmap
# sns.heatmap(data=heatmap_data, annot=True)
# plt.xlabel('Countries')
# plt.show()

# ***************************************************************************************************************************
#                                            Displot for Score Volume                                                       #
# ***************************************************************************************************************************
# displotData = pd.read_csv(data_path)
# sns.displot(data=displotData,x='points',kind='kde')
# plt.show()

# ***************************************************************************************************************************
#                                                   Price/Score Plots                                                       #
# ***************************************************************************************************************************
# displotData = pd.read_csv(data_path)
# sns.displot(data=displotData,x='points',kind='kde')
# plt.show()

# hexData=pd.read_csv(data_path)
# hexData[hexData['price'] < 100].plot.hexbin(x='price', y='points', gridsize=15)
# plt.ylabel('Points')
# plt.xlabel('Price')
# plt.show()

# ***************************************************************************************************************************
#                                                  Predicted Variety Scores                                                 #
# ***************************************************************************************************************************
predicted_data = pd.read_csv('data_output_csv/xgbRegressorOutput.csv')

# Varieties & average predicted scores
varieties = {'Cabernet Sauvignon': 87.96, 'Other': 87.95, 'Sauvignon Blanc': 87.92,
             'Pinot Noir': 88.0, 'Chardonnay': 87.93, 'Syrah': 87.88, 'Red Blend': 87.91,
             'Riesling': 87.99, 'Zinfandel': 87.91, 'Bordeaux-style Red Blend': 87.96, 'Merlot': 87.98}

# Create a dataframe with the top 10 varieties and their predicted scores
df = pd.DataFrame(columns=['variety','predicted_score'])

for variety, score in varieties.items():
    new_row = {'variety':variety,'predicted_score':score} 
    df = df.append(new_row, ignore_index=True)

print(df)

plt.figure(figsize=(10,6))
plt.ylim(87.6,88.1)
plt.title('Wine Varieties by Predicted Score')
variety_plot = sns.barplot(x=df['variety'],y=df['predicted_score'],order=df.sort_values('predicted_score',ascending=False).variety)
plt.ylabel('Predicted Score')
plt.xlabel('Varieties')
variety_plot.set_xticklabels(variety_plot.get_xticklabels(), rotation=40, ha='right')
plt.tight_layout
plt.show()

# sns.barplot(x='Education', 
#             y="Salary", 
#             data=df, 
#             order=df.sort_values('Salary',ascending = False).Education)


    
            











