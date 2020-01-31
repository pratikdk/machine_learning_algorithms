### Simple Linear Regression using Ordinary Least Squares[OLS]

import numpy as np
import pandas as pd

train_data = pd.read_csv('train_dataset.csv')
test_data = pd.read_csv('test_dataset.csv')

# Keep only one column/variable as input variable (Simple Regression)
train_data = train_data[['a', 'y']]
test_data = test_data[['a', 'y']]
x = train_data['a'].values
y = train_data['y'].values

# Hypothesis structure(Linear Representation) y = theta0 + (theta1*x)
def hypothesis(theta0, theta1, x):
    return theta0 + (theta1*x)

# Method computes best theta0, theta1 using 'ols' -> Ordinary Least Squares
def find_params_using_ols(x, y):
    m = len(x) # Number of rows/examples
    # Compute/initialize pieces of OLS formula
    numerator = 0
    denominator = 0
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    for i in range(m): # Iterate over each example
        numerator += (x[i] - mean_x) * (y[i] - mean_y)
        denominator += (x[i] - mean_x) ** 2
    # Finally compute theta0 and theta1 with above pre-computations
    theta1 = numerator/denominator
    theta0 = mean_y - (theta1 * mean_x)
    return theta0, theta1

# Compute parameters to best approximate model of data with a linear representation
theta0, theta1 = find_params_using_ols(x, y)

## Model Evaluation
# RMSE
def rmse(y, y_pred):
    rmse = np.sqrt(np.sum((y - y_pred)**2) / len(y))
    return rmse

# R2 Score
# How much(%) of the total variation in y is explained by variation in x(fitted line)
def r2_score(y, y_pred):  
    mean_y = np.mean(y)
    SE_total_variation = np.sum((y - mean_y)**2) # Unexplained max possible variation in y wrt->Mean
    SE_line_variation = np.sum((y - y_pred)**2) # Unexplained variation in y wrt -> fitted line
    r2 = 1 - (SE_line_variation / SE_total_variation) # Expalined = 1 - Unexplained
    return r2

#Training set evaluation
print("Training-set Evaluation:")
y_pred_train = hypothesis(theta0, theta1, x)
print(f"RMSE = {rmse(y, y_pred_train)}")
print(f"R2_score = {r2_score(y, y_pred_train)}")

#Testing set evaluation
print("Testing-set Evaluation:")
y_pred_test = hypothesis(theta0, theta1, test_data['a'].values)
print(f"RMSE = {rmse(test_data['y'].values, y_pred_test)}")
print(f"R2_score = {r2_score(test_data['y'].values, y_pred_test)}")