### Decision Tree using Scikit

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree.export import export_text

df = pd.read_csv('Iris.csv')
df = df.drop("Id", axis=1)
df = df.rename(columns={"species": "label"})

# Train test split
def train_test_split(df, test_size=0.8, random_state=None):
    train_df = df.sample(frac=test_size, random_state=random_state)
    test_df = df[~df.index.isin(train_df.index)]
    return train_df.sort_index(), test_df.sort_index()

train_df, test_df = train_test_split(df, 0.8, 100)

decision_tree = DecisionTreeClassifier(random_state=0, max_depth=3)

decision_tree = decision_tree.fit(train_df.iloc[:, :-1], train_df.iloc[:, -1])

formated_tree = export_text(decision_tree, feature_names=df.iloc[:, :-1].columns.tolist())

print(formated_tree)

#### Evaluate
decision_tree.score(test_df.iloc[:, :-1], test_df.iloc[:, -1]) * 100

