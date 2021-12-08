from numpy.core.fromnumeric import var
import pandas as pd
import numpy as np
from pandas.core.algorithms import rank
import matplotlib.pyplot as plt
import seaborn as sns


data_path = 'csv/wineData.csv'
wine_data = pd.read_csv(data_path, index_col='country')

# ***************************************************************************************************************************
#                                      Bar Plot for Average Score/Country                                                   #
# ***************************************************************************************************************************
plt.figure(figsize=(10,6))
plt.ylim(80,95)
plt.title('Average Wine Score by Country')
sns.barplot(x = wine_data.index, y = wine_data['points'])
plt.ylabel('Average Score')
plt.xlabel('Countries')
plt.show()

