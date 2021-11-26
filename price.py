import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

data_path = 'winemag_data.csv'
home_data = pd.read_csv(data_path)

y = home_data.points
features = ['price']
X = home_data[features]

X_train, X_valid, y_train, y_valid = train_test_split(X,y, random_state=1)

my_imputer = SimpleImputer()
imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))
imputed_X_valid = pd.DataFrame(my_imputer.transform(X_valid))
imputed_X_train.columns = X_train.columns
imputed_X_valid.columns = X_valid.columns

rf_model = RandomForestRegressor(random_state=0)
rf_model.fit(imputed_X_train, y_train)
rf_val_predictions = rf_model.predict(imputed_X_valid)
rf_val_mae = mean_absolute_error(rf_val_predictions, y_valid)

print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))
