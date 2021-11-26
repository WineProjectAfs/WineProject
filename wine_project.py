#Import Libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

#Load Data and Separate Target
data_path = "winemag-data_first150k.csv"
wine_data = pd.read_csv(data_path)

Dup_Rows = wine_data[wine_data.duplicated()]
print("\n\nDuplicate Rows : \n {}".format(Dup_Rows))

DF_RM_DUP = wine_data.drop_duplicates(keep=False)

print('\n\nResult DataFrame after duplicate removal :\n', DF_RM_DUP.head(n=5))

#Create y
y = wine_data.points

#Create X
features = ['country','variety']

#Select columns corresponding to feature and Preview data
X = wine_data[features]

print(X)

#Split into Validation and Training Data
X_train,X_valid,y_train,y_valid = train_test_split(X,y,random_state=1)

#Apply OneHotEncoder
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[features]))
OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid[features]))

#OneHotEncoding removed index...put it back
OH_cols_train.index = X_train.index
OH_cols_valid.index = X_valid.index

#Define Random Forest Model
rf_model = RandomForestRegressor(random_state=0)
rf_model.fit(OH_cols_train, y_train)
rf_val_predictions = rf_model.predict(OH_cols_valid)
rf_val_mae = mean_absolute_error(rf_val_predictions, y_valid)

print("Validation MAE for Random Forest Model:{:,.0f}".format(rf_val_mae))


