## Simple Linear Regression using Ordinary Least Squares[OLS] with Sci-kit

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the data
train_data = pd.read_csv('train_dataset.csv')
test_data = pd.read_csv('test_dataset.csv')

# Prepare the data to be compatible with scikit learn
train_data = train_data[['a', 'y']]
test_data = test_data[['a', 'y']]
x = train_data['a'].values
y = train_data['y'].values
x_test = test_data['a'].values
y_test = test_data['y'].values
# Reshape input; Cannot use Rank 1 matrix in scikit learn 
x = x.reshape((-1, 1))
x_test = x_test.reshape((-1, 1))

# Initialize Linear model
reg = LinearRegression()
# Fit and approximate the function/representation which models training data
reg = reg.fit(x, y)

## Model Evaluation

#Training set evaluation
print("Training-set Evaluation:")
y_pred_train = reg.predict(x)
train_mse = mean_squared_error(y, y_pred_train) # MSE
train_rmse = np.sqrt(train_mse) # RMSE
train_r2_score = reg.score(x, y) # R2 score
print(f"RMSE = {train_rmse}")
print(f"R2_score = {train_r2_score}")

#Testing set evaluation
print("Testing-set Evaluation:")
y_pred_test = reg.predict(x_test)
test_mse = mean_squared_error(y_test, y_pred_test)
test_rmse = np.sqrt(test_mse)
test_r2_score = reg.score(x_test, y_test)
print(f"RMSE = {test_rmse}")
print(f"R2_score = {test_r2_score}")