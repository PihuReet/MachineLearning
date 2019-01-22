import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import  DecisionTreeRegressor

# save filepath to variable for easier access
train_file_path = '/home/manoj/PycharmProjects/MachineLearning/train.csv'

# read the data and store data in DataFrame titled melbourne_data
train_data = pd.read_csv(train_file_path)

# print a summary of the data in Melbourne data
train_data.describe()

train_y = train_data.SalePrice

predictor_cols = ['LotArea', 'OverallQual', 'YearBuilt', 'TotRmsAbvGrd']

train_X = train_data[predictor_cols]

my_model = RandomForestRegressor()

my_model.fit(train_X, train_y)

my_pred = my_model.predict(train_X)

print(my_pred)

test = pd.read_csv('/home/manoj/PycharmProjects/MachineLearning/test.csv')

test_X = test[predictor_cols]

predicted_prices = my_model.predict(test_X)

print(predicted_prices)

my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted_prices})

my_submission.to_csv('submission.csv', index=False)
