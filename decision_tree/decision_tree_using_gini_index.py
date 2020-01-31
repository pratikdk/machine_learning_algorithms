### Decision Tree using gini-index as spliting metric

#### Import Statements
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
from pprint import pprint

#### Load and Prepare Data
df = pd.read_csv('Iris.csv')
df = df.drop("Id", axis=1)
df = df.rename(columns={"species": "label"})

#### Helper Functions

# Train test split
def train_test_split(df, test_size=0.8, random_state=None):
    train_df = df.sample(frac=test_size, random_state=random_state)
    test_df = df[~df.index.isin(train_df.index)]
    return train_df.sort_index(), test_df.sort_index()

### Classify with the most frequent value
def classify_data(data_group):
    
    (values, counts) = np.unique(data_group[:, -1], return_counts=True)
    most_common_value_indx = np.argmax(counts)
    
    return values[most_common_value_indx]


### Get potential splits for each feature
def get_potential_splits(data_group): # Split on each unique value(or each value)
    potential_splits = {} # Can essentially make a split at each unique value
    
    for column_index in range(data_group.shape[1] - 1):
        potential_splits[column_index] = np.unique(data_group[:, column_index])
    
    return potential_splits


def split_data(data, split_column_index, splitting_value):
    
    split_column_values = data[:, split_column_index]

    type_of_feature = FEATURE_TYPES[split_column_index]
    if type_of_feature == "continuous":
        data_below = data[split_column_values <= splitting_value]
        data_above = data[split_column_values >  splitting_value]
    
    # feature is categorical   
    else:
        data_below = data[split_column_values == splitting_value]
        data_above = data[split_column_values != splitting_value]
    
    return data_below, data_above


#### Gini impurity
# **Gini(feature_question)** = 1 - (no. of rows as yes)/((no. of rows as yes) + (no. of rows as no))^2 - (no. of rows as no)/((no. of rows as yes) + (no. of rows as no))^2
# ### Weighted Gini Index
# **Weighted_Gini_Index(feature_question)** = (((no. of examples in left child node) / (total no. of examples in parent node)) * (gini_index of left node) + ((no. of examples in right child node) / (total no. of examples in parent node)) * (gini_index of right node))


### Calculate gini index
def calculate_gini_index(data_group):
    _, counts = np.unique(data_group[:, -1], return_counts=True)
    
    probabilities = counts / counts.sum()
    gini_impurity = - (1 + sum(probabilities**2))
    
    return gini_impurity


def calculate_weigthed_gini_index(parent_data_group, left_data_group, right_data_group):
    
    # Calculate gini index for parent and left, right nodes
    left_group_gini_index = calculate_gini_index(left_data_group)
    right_group_gini_index = calculate_gini_index(right_data_group)
    
    # Probabilities of left and right node
    num_examples_parent = len(parent_data_group)
    left_group_probability = len(left_data_group) / num_examples_parent
    right_group_probability = len(right_data_group) / num_examples_parent
    
    # Final equation for weigthed_gini_index computation
    weigthed_gini_index = (left_group_probability * left_group_gini_index) + (right_group_probability * right_group_gini_index)
    
    return weigthed_gini_index


# Determines which column/feature to split on
def determine_best_split(data_group, potential_splits):
    
    max_weighted_gini_index = 9999
    for column_index in potential_splits:
        for value in potential_splits[column_index]:
            # Go for a split 
            left_data_group, right_group_data = split_data(data_group, split_column_index=column_index, splitting_value=value)
            current_weighted_gini_index = calculate_weigthed_gini_index(data_group, left_data_group, right_group_data)
            
            if current_weighted_gini_index <= max_weighted_gini_index:
                max_weighted_gini_index = current_weighted_gini_index
                best_split_column_index = column_index
                best_split_value = value
    
    return best_split_column_index, best_split_value


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


#### Decision Tree Algorithm

def build_Decision_Tree(df, counter=0, min_samples=2, max_depth=5):
    ### Data preparation
    if counter == 0: # Declare before tree building(to skip re-declarations when doing recursions)
        global COLUMN_HEADERS, FEATURE_TYPES
        COLUMN_HEADERS = df.columns
        FEATURE_TYPES = determine_type_of_feature(df)
        data = df.values
    else:
        data = df # this is basically a numpy array ()
        
    ### Base cases
    ## Check if, # To conclude as a Leaf node
    # data(rows) are of same label or
    # data(rows) have length < min_samples
    # if tree has crossed a depth threshold (manitored using counter)
    if (np.unique(data[:, -1]).size == 1) or (len(data) < min_samples) or (counter == max_depth):
        classification = classify_data(data)
        
        return classification
    
    ### Mighty-Greedy Recursion
    # The tree is constructed in a depth first on Left, back traversal on Right fashion.
    else: # Its a dictionary (haven't exploited our base case)/ hence ask Question and split
        counter += 1
        
        # Call for help using our helper functions
        potential_splits = get_potential_splits(data)
        split_column_index, split_value = determine_best_split(data, potential_splits)
        left_data_group, right_data_group = split_data(data, split_column_index, split_value)
        
        # Check for empty data
        if len(left_data_group) == 0 or len(right_data_group) == 0: # Either nothing or a pure group
            classification = classify_data(data)
            return classification # Classify as leaf 
        
        # determine question
        feature_name = COLUMN_HEADERS[split_column_index]
        type_of_feature = FEATURE_TYPES[split_column_index]
        if type_of_feature == "continuous":
            question = "{} <= {}".format(feature_name, split_value)
        else: # feature is categorical
            question = "{} = {}".format(feature_name, split_value)
            
        # Instantiate sub-tree
        sub_tree = {question: []}
        
        # Find answers: Find(by questioning) or conclude(classify with leaf)
        # Recursion starts here.
        # First completely expand tree on left(yes) questions/answers, followed by answering right while traversing back until root is reached.
        yes_answer = build_Decision_Tree(left_data_group, counter, min_samples, max_depth)
        no_answer = build_Decision_Tree(right_data_group, counter, min_samples, max_depth)
        
        # If the answers are the same, then there is no point in asking the qestion.
        # This could happen when the data is classified even though it is not pure
        # yet (min_samples or max_depth base case).
        if yes_answer == no_answer:
            sub_tree = yes_answer
        else:
            sub_tree[question].append(yes_answer)
            sub_tree[question].append(no_answer)
        
        return sub_tree



train_df, test_df = train_test_split(df, 0.8, 100)

tree = build_Decision_Tree(train_df, max_depth=3)
pprint(tree)


#### Evaluate

def classify_example(example, tree):
    question = list(tree.keys())[0]
    feature_name, comparison_operator, value = question.split(" ")
    
    # ask question
    if comparison_operator == "<=":
        if example[feature_name] <= float(value):
            answer = tree[question][0] # Yes
        else:
            answer = tree[question][1] # No
            
    # feature is categorical
    else:
        if str(example[feature_name]) == value:
            answer = tree[question][0] # Yes
        else:
            answer = tree[question][1] # No
    
    # base case
    if not isinstance(answer, dict): # Answer found
        return answer
    # recursive part
    else: # Follow along the tree
        residual_tree = answer
        return classify_example(example, residual_tree)


def calculate_accuracy(df, tree):

    df["classification"] = df.apply(classify_example, args=(tree,), axis=1)
    df["classification_correct"] = df["classification"] == df["label"]
    
    accuracy = df["classification_correct"].mean()
    
    return accuracy


accuracy = calculate_accuracy(test_df, tree)
print(accuracy)





