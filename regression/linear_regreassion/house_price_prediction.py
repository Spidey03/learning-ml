
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from linear_regression_scratch import ridge_regression

train_data = pd.read_csv('data/house_prices_data/train.csv')
test_data = pd.read_csv('data/house_prices_data/test.csv')

train_data.info()

train_Y = train_data.pop('SalePrice')

print(train_data.shape)
print(train_Y)
train_X = np.insert(train_data, 0, 1, axis=1)

# from regression.linear_regreassion.linear_regression_scratch import ridge_regression

theta = ridge_regression(train_X=train_data, train_Y=train_Y, Lambda=1, split_data=False, use_normal_eqn=False)
print(theta)