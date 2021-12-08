from numpy.core.fromnumeric import var
import pandas as pd
import numpy as np
from pandas.core.algorithms import rank
import matplotlib.pyplot as plt
import seaborn as sns

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

# ***************************************************************************************************************************
#                                      Scatter Plot for Average Score/Country                                               #
# ***************************************************************************************************************************
# Only US, France, Italy, and Australia have wines that have scored 100
# Scores of 100 may not be of great significance as for 100 there are only 24 entries, 

# plt.title('Scores by Country')
# sns.scatterplot(x=wine_data.index, y=wine_data['points'])
# plt.show()

points_series = wine_data.points.value_counts().sort_index(ascending=False)
print(points_series)



