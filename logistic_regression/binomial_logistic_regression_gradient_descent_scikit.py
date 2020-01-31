## Binomial Logistic Regression using Gradient Descent (Scikit)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score 

train_data = pd.read_csv('train_dataset.csv')
test_data = pd.read_csv('test_dataset.csv')
# Seperate input from output
x_train = train_data[['x1', 'x2']].values
y_train = train_data['y'].values
x_test = test_data[['x1', 'x2']].values
y_test = test_data['y'].values

print(x_train.shape, x_test.shape)

# Initialize Model
model = LogisticRegression(solver='lbfgs')
# Fit and approximate function
model.fit(x_train, y_train)

#### Model Evaluation
train_preds = model.predict(x_train)
test_preds = model.predict(x_test)

train_acc = accuracy_score(y_train.flatten(), train_preds)
test_acc = accuracy_score(y_test.flatten(), test_preds)

print(f"Training accuracy = {train_acc}")
print(f"Testing accuracy = {test_acc}")

parameters= model.coef_
print(f'Parameters: {parameters}')