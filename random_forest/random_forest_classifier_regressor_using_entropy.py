## Random Forest algorithm using entropy (Classifier+Regressor)

### Import Statements
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

import random
from pprint import pprint


### Load and Prepare Data

df = pd.read_csv("sonar.csv", header=None)
df = df.add_prefix('f_')

print(df.head())
print(f'Input Shape: {df.shape[1]-1}')

### Decision Tree - Helper Functions

#### Data pure?

def check_purity(data):
    
    label_column = data[:, -1]
    unique_classes = np.unique(label_column)

    if len(unique_classes) == 1:
        return True
    else:
        return False


#### Create Leaf

def create_leaf(data, ml_task):
    
    label_column = data[:, -1]
    if ml_task == "regression":
        leaf = np.mean(label_column)
        
    # classfication    
    else:
        unique_classes, counts_unique_classes = np.unique(label_column, return_counts=True)
        index = counts_unique_classes.argmax()
        leaf = unique_classes[index]
    
    return leaf


#### Potential splits? ..... Random Forest adaption added below

def get_potential_splits(data, num_features_split):

    potential_splits = {}
    _, n_columns = data.shape
    
    # Random Forest Adaption: Random subspacing of feature space
    feature_indices = np.random.choice((n_columns-1), num_features_split, replace=False) # excluding the last column which is the label
        
    for column_index in feature_indices:          
        values = data[:, column_index]
        unique_values = np.unique(values)
        
        potential_splits[column_index] = unique_values
    
    return potential_splits


#### Split Data

def split_data(data, split_column, split_value):
    
    split_column_values = data[:, split_column]

    type_of_feature = FEATURE_TYPES[split_column]
    if type_of_feature == "continuous":
        data_below = data[split_column_values <= split_value]
        data_above = data[split_column_values >  split_value]
    
    # feature is categorical   
    else:
        data_below = data[split_column_values == split_value]
        data_above = data[split_column_values != split_value]
    
    return data_below, data_above


#### Determine Best Split

def calculate_mse(data):
    actual_values = data[:, -1]
    if len(actual_values) == 0:   # empty data
        mse = 0
        
    else:
        prediction = np.mean(actual_values)
        mse = np.mean((actual_values - prediction) **2)
    
    return mse


def calculate_entropy(data):
    
    label_column = data[:, -1]
    _, counts = np.unique(label_column, return_counts=True)

    probabilities = counts / counts.sum()
    entropy = sum(probabilities * -np.log2(probabilities))
     
    return entropy


def calculate_overall_metric(data_below, data_above, metric_function):
    
    n = len(data_below) + len(data_above)
    p_data_below = len(data_below) / n
    p_data_above = len(data_above) / n

    overall_metric =  (p_data_below * metric_function(data_below) 
                     + p_data_above * metric_function(data_above))
    
    return overall_metric


def determine_best_split(data, potential_splits, ml_task):
    
    first_iteration = True
    for column_index in potential_splits:
        for value in potential_splits[column_index]:
            data_below, data_above = split_data(data, split_column=column_index, split_value=value)
            
            if ml_task == "regression":
                current_overall_metric = calculate_overall_metric(data_below, data_above, metric_function=calculate_mse)
            
            # classification
            else:
                current_overall_metric = calculate_overall_metric(data_below, data_above, metric_function=calculate_entropy)

            if first_iteration or current_overall_metric <= best_overall_metric:
                first_iteration = False
                
                best_overall_metric = current_overall_metric
                best_split_column = column_index
                best_split_value = value
    
    return best_split_column, best_split_value


### Decision Tree Algorithm

#### Determine Type of Feature

def determine_type_of_feature(df):
    
    feature_types = []
    n_unique_values_treshold = 15
    for feature in df.columns:
        if feature != "label":
            unique_values = df[feature].unique()
            example_value = unique_values[0]

            if (isinstance(example_value, str)) or (len(unique_values) <= n_unique_values_treshold):
                feature_types.append("categorical")
            else:
                feature_types.append("continuous")
    
    return feature_types


#### Algorithm

def decision_tree_algorithm(df, ml_task, min_samples=2, max_depth=5, num_features_split=3, counter=0):
    # data preparations
    if counter == 0:
        global COLUMN_HEADERS, FEATURE_TYPES
        COLUMN_HEADERS = df.columns
        FEATURE_TYPES = determine_type_of_feature(df)
        data = df.values
    else:
        data = df           
    
    
    # base cases
    if (check_purity(data)) or (len(data) < min_samples) or (counter == max_depth):
        leaf = create_leaf(data, ml_task)
        return leaf

    
    # recursive part
    else:    
        counter += 1

        # helper functions 
        potential_splits = get_potential_splits(data, num_features_split)
        split_column, split_value = determine_best_split(data, potential_splits, ml_task)
        data_below, data_above = split_data(data, split_column, split_value)
        
        # check for empty data
        if len(data_below) == 0 or len(data_above) == 0:
            leaf = create_leaf(data, ml_task)
            return leaf
        
        # determine question
        feature_name = COLUMN_HEADERS[split_column]
        type_of_feature = FEATURE_TYPES[split_column]
        if type_of_feature == "continuous":
            question = "{} <= {}".format(feature_name, split_value)
            
        # feature is categorical
        else:
            question = "{} = {}".format(feature_name, split_value)
        
        # instantiate sub-tree
        sub_tree = {question: []}
        
        # find answers (recursion)
        yes_answer = decision_tree_algorithm(data_below, ml_task, min_samples, max_depth, num_features_split, counter)
        no_answer = decision_tree_algorithm(data_above, ml_task, min_samples, max_depth, num_features_split, counter)
        
        # If the answers are the same, then there is no point in asking the qestion.
        # This could happen when the data is classified even though it is not pure
        # yet (min_samples or max_depth base case).
        if yes_answer == no_answer:
            sub_tree = yes_answer
        else:
            sub_tree[question].append(yes_answer)
            sub_tree[question].append(no_answer)
        
        return sub_tree


# tree = decision_tree_algorithm(train_df, ml_task="regression", max_depth=3)
# pprint(tree)


#### Prediction

def predict_example(example, tree):
    question = list(tree.keys())[0]
    feature_name, comparison_operator, value = question.split(" ")
    
    # ask question
    if comparison_operator == "<=":
        if example[feature_name] <= float(value):
            answer = tree[question][0]
        else:
            answer = tree[question][1]
    
    # feature is categorical
    else:
        if str(example[feature_name]) == value:
            answer = tree[question][0]
        else:
            answer = tree[question][1]

    # base case
    if not isinstance(answer, dict):
        return answer
    
    # recursive part
    else:
        residual_tree = answer
        return predict_example(example, residual_tree)


### Bootstrap helper functions

def bootstrap_sampling(train_df, sample_size):
    return train_df.sample(frac=sample_size, replace=True)


def bagging_predict(trees, test_df, ml_task):
    df_tree_preds = pd.DataFrame()
    for i, tree in enumerate(trees):
        df_tree_preds[i] = test_df.apply(predict_example, args=(tree,), axis=1)
    if ml_task == "regression":
        preds = df_tree_preds.mean(axis=1)
    else:
        preds = df_tree_preds.mode(axis=1).iloc[:,0]
    return preds


def bagging(train_df, test_df, ml_task, min_samples, max_depth, sample_size, n_trees, num_features_split):
    
    trees = []
    for i in range(n_trees):
        bootstrap_sample = bootstrap_sampling(train_df, sample_size)
        tree = decision_tree_algorithm(train_df, ml_task, min_samples, max_depth, num_features_split)
        trees.append(tree)
    preds = bagging_predict(trees, test_df, ml_task)
    
    return preds


def cross_validation_folds(df, n_folds):
    
    # Decode n_folds to n_rows to n_fraction(fraction of rows for df)
    df_copy = df.copy()
    n_rows = int(df_copy.shape[0] / n_folds)
    n_fraction = n_rows / df_copy.shape[0]
    folds = []
    # Make n_folds samples
    for fold in range(n_folds):
        n_sample = df_copy.sample(n=n_rows, replace=False)
        df_copy.drop(n_sample.index)
        folds.append(n_sample)
        
    return folds


def rmse(y, y_pred):
    rmse = np.sqrt(np.sum((y - y_pred)**2) / len(y))
    return rmse


def calculate_bag_score(actual, predicted, ml_task):
    if ml_task == "regression":
        score = rmse(actual, predicted)
    else:
#         score = (calculate_accuracy(predicted, actual))
        score = (actual == predicted).mean()
    return score


def evaluate_algorithm(df, algorithm, n_folds, ml_task, min_samples, max_depth, sample_size, n_trees, num_features_split):
    folds = cross_validation_folds(df, n_folds)
    scores = []
    for i, fold in enumerate(folds):
        train_df = pd.concat(folds, axis=0)
        train_df = train_df.drop(fold.index, axis=0)
        test_df = fold
        predicted = algorithm(train_df, test_df, ml_task, min_samples, max_depth, sample_size, n_trees, num_features_split)
        actual = test_df.iloc[:,-1]
        score = calculate_bag_score(actual, predicted, ml_task)
        scores.append(score)
    return scores


# Number of features to operate(greedy search) on for performing each split.
num_features_split = int(np.sqrt(df.shape[1]-1))

scores = evaluate_algorithm(df, bagging, 5, "classification", 2, 3, 0.50, 10, num_features_split)

print(np.mean(scores))