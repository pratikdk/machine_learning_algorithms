## Multinominal Logistic Regression using Gradient Descent (Scikit)

from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load data
iris = datasets.load_iris()
x = iris.data
y = iris.target

print(x.shape, y.shape)

# Total classes
print(f'Total Classes: {np.unique(y)}')

# Standarize features
scaler = StandardScaler()
x_std = scaler.fit_transform(x)

# Initialize one vs rest Logistic regression classifier object
clf = LogisticRegression(random_state=0, multi_class='multinomial', solver='newton-cg')
# Train the initialized model using iris training data
model = clf.fit(x_std, y)

# New observation
new_observation = [[.5, .5, .5, .5]]

# Predict new observation class using the trained model
class_predicted = model.predict(new_observation)
print(f'Class predicted: {class_predicted}')

# View predicted probabilities
prediciton_probs = model.predict_proba(new_observation)
print(f'Prediction probabilities for each class: {prediciton_probs}')